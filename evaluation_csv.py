import argparse
import json
import os
import re
import evaluate
import torch
import numpy as np
import random
from collections import defaultdict
import csv


torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

def load_predictions(pred_path):
    preds = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj and "output" in obj:
                preds[obj["id"]] = obj["output"].strip()
    return preds

def load_references(gt_path, ref_key="tgt"):
    refs = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            obj = json.loads(line)
            if "id" in obj:
                refs[obj["id"]] = [obj.get(ref_key, "").strip()]
    return refs

def align_preds_and_refs(preds_dict, refs_dict):
    ids = sorted(refs_dict.keys())
    preds_list, refs_list = [], []
    missing = []
    for i in ids:
        p = preds_dict.get(i, "").strip()
        if not p:
            missing.append(i)
        preds_list.append(p)
        refs_list.append(refs_dict[i])
    if missing:
        print(f"[WARN] Missing predictions for {len(missing)} IDs")
    print(f"[INFO] Total examples evaluated: {len(ids)}")
    return preds_list, refs_list

def compute_metrics(task, preds, refs_list_of_lists):
    results = {}
    flat_refs = [r[0] for r in refs_list_of_lists]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # BERTScore
    bert = evaluate.load("bertscore", device=device)
    bs = bert.compute(predictions=preds, references=flat_refs, lang='en', batch_size=8)
    results["BERTScore(F1)"] = np.mean(bs["f1"]) if bs["f1"] else None

    # ROUGE
    rouge = evaluate.load("rouge")
    r = rouge.compute(predictions=preds, references=flat_refs)
    results["ROUGE-L"] = r.get("rougeL", None)  # rename key to match how it's referenced later

    # Task-specific metrics
    t = task.lower()
    if t == "translation":
        bleu = evaluate.load("bleu")
        b = bleu.compute(predictions=preds, references=refs_list_of_lists)
        results["BLEU"] = b.get("bleu", None)
    elif t == "dialogue":
        mauve = evaluate.load("mauve")
        m = mauve.compute(predictions=preds, references=flat_refs)
        results["MAUVE"] = m.mauve

    return results

# exact mapping of method → nice name + hyperparam regex
STRATEGY_INFO = [
    ("diverse_beam_search", "diverse beam search", r"divbeams(\d+)_groups(\d+)"),
    ("beam_search",          "beam search",         r"beams(\d+)"),
    ("dola",                 "DoLa",                r"layers(low|high)"),
    ("eta",                  "eta",                 r"eps([0-9]*\.?[0-9]+)"),
    ("min_p",                "min p",               r"min_p([0-9]*\.?[0-9]+)"),
    ("temperature",          "Temperature",         r"temp([0-9]*\.?[0-9]+)"),
    ("greedy",               "greedy",              None),
    ("top_k",                "top k",               r"topk(\d+)"),
    ("top_p",                "top p",               r"topp([0-9]*\.?[0-9]+)"),
    ("typical",              "typical",             r"typical([0-9]*\.?[0-9]+)"),
    ("contrastive",          "contrastive search",  r"csα([0-9]*\.?[0-9]+)"),

]

def extract_row_key(filepath: str) -> str:
    base = os.path.basename(filepath).rsplit(".jsonl",1)[0]
    for key, nice_name, pattern in STRATEGY_INFO:
        if key in base:
            if not pattern:
                return nice_name
            m = re.search(pattern, base)
            if not m:
                return nice_name
            param = ", ".join(m.groups())
            return f"{nice_name} {param}"
    return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True,
                        help="Path to one .jsonl or folder of .jsonl predictions")
    parser.add_argument("--gt_file", required=True,
                        help="Ground-truth JSONL file (with 'id' + 'tgt')")
    parser.add_argument("--task", required=True,
                        choices=["qa","images","translation","summarization","dialogue"])
    args = parser.parse_args()

    print(f"[INFO] Loading ground-truth from: {args.gt_file}")
    refs = load_references(args.gt_file, ref_key="tgt")

    # discover prediction files
    if os.path.isdir(args.pred_path):
        files = sorted(os.listdir(args.pred_path))
        pred_files = [os.path.join(args.pred_path, f) for f in files if f.endswith(".jsonl")]
    elif os.path.isfile(args.pred_path):
        pred_files = [args.pred_path]
    else:
        raise ValueError(f"Not a file or folder: {args.pred_path}")
    if not pred_files:
        raise ValueError("No .jsonl prediction files found.")

    # matrix[row][col] = {metric_name: value}
    matrix = defaultdict(lambda: defaultdict(dict))
    cols = set()

    task_pretty = {
        "translation":"Translation",
        "summarization":"Summarization",
        "dialogue":"Dialogue",
        "qa":"QA",
        "images":"Images"
    }[args.task]

    for pf in pred_files:
        print(f"[INFO] Evaluating {pf} …")
        preds = load_predictions(pf)
        pred_list, ref_list = align_preds_and_refs(preds, refs)
        mets = compute_metrics(args.task, pred_list, ref_list)

        # identify primary metric
        if args.task == "translation":
            primary = mets.get("BLEU")
            primary_name = "BLEU"
        elif args.task == "summarization":
            primary = mets.get("rougeL")
            primary_name = "ROUGE-L"
        elif args.task == "images":
            primary = mets.get("rougeL")
            primary_name = "ROUGE-L"
        else:  # qa or dialogue
            primary = mets.get("MAUVE")
            primary_name = "MAUVE"

        bscore = mets.get("BERTScore(F1)")

        rouge_l = mets.get("ROUGE-L")

        if primary is None and bscore is None and rouge_l is None:
            continue

        row = extract_row_key(pf)
        model = os.path.basename(pf).split("_",1)[0]
        col = f"{task_pretty} {model}"

        # store both
        if primary is not None:
            matrix[row][col][primary_name] = primary
        if bscore is not None:
            matrix[row][col]["BERTScore(F1)"] = bscore
        if rouge_l is not None:
            matrix[row][col]["ROUGE-L"] = rouge_l
        cols.add(col)

    # write CSV with 2 columns per model
    out_csv = os.path.join(
        os.path.dirname(pred_files[0]),
        f"{args.task}_results_summary.csv"
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf, delimiter="\t")

        # header: for each col, Primary then BERTScore
        header = ["Decoding Strategy + Hyperparam"]
        for c in sorted(cols):
            header += [f"{c} {primary_name}", f"{c} ROUGE-L", f"{c} BERTScore(F1)"]
        writer.writerow(header)

        for r in sorted(matrix):
            row = [r]
            for c in sorted(cols):
                cell = matrix[r].get(c, {})
                # format numbers or blank
                prim_val = f"{cell.get(primary_name, ''):.4f}" if primary_name in cell else ""
                rouge_val = f"{cell.get('ROUGE-L', ''):.4f}" if "ROUGE-L" in cell else ""
                bs_val = f"{cell.get('BERTScore(F1)', ''):.4f}" if "BERTScore(F1)" in cell else ""
                row += [prim_val, rouge_val, bs_val]
            writer.writerow(row)

    print(f"[INFO] CSV with two metrics written to {out_csv}")

if __name__ == "__main__":
    main()
