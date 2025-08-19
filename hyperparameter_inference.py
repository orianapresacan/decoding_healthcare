import argparse
import json
import os
import time
import numpy as np
import random
import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, Gemma3ForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, Gemma3ForConditionalGeneration
from PIL import Image


def load_model_and_tokenizer(model_id):
    print(f"[INFO] Loading tokenizer and model '{model_id}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        print("[INFO] → No pad_token found; set pad_token = eos_token or manually added.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()

    return model, tokenizer

def load_llava_model(model_id):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    return model, processor

def load_gemma_1b_model(model_id):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Gemma3ForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def load_gemma_12b_model(model_id):
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def load_medgemma_model(model_id):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def build_generation_params(args, tokenizer=None):
    strat = args.decoding_strategy.lower()
    print(f"[INFO] Building generation parameters for strategy: '{strat}'")

    gen_params = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 1,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 0,
        "do_sample": False
    }

    if tokenizer and "gemma" not in args.model.lower() and "medgemma" not in args.model.lower():
        gen_params.update({
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        })

    # ─── 1) GREEDY ───────────────────────────────────────────────────────
    if strat == "greedy":
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params
    
    # ─── 2) DIVERSE BEAM SEARCH ─────────────────────────────────────────────────
    if strat == "diverse_beam_search":
        num_beams = args.num_beams 
        num_beam_groups = args.num_beam_groups
        gen_params["num_beams"] = num_beams
        gen_params["diversity_penalty"] = 1.0

        if args.num_beam_groups is not None:
            if num_beams % args.num_beam_groups != 0:
                raise ValueError("`--num_beam_groups` must divide `--num_beams` evenly")
            gen_params["num_beam_groups"] = num_beam_groups
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 2) BEAM SEARCH ─────────────────────────────────────────────────
    if strat == "beam_search":
        num_beams = args.num_beams 
        gen_params["num_beams"] = num_beams
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 3) TOP-K SAMPLING ───────────────────────────────────────────────
    if strat == "top_k":
        top_k = args.top_k 
        gen_params["do_sample"] = True
        gen_params["top_k"] = top_k
        gen_params["temperature"] = 1.0
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 4) TOP-P (NUCLEUS) SAMPLING ─────────────────────────────────────
    if strat == "top_p":
        top_p = args.top_p 
        gen_params["do_sample"] = True
        gen_params["temperature"] = 1.0
        gen_params["top_p"] = top_p
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 5) TEMPERATURE SAMPLING ─────────────────────────────────────────
    if strat == "temperature":
        temp = args.temperature 
        gen_params["do_sample"] = True
        gen_params["temperature"] = temp
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 6) TYPICAL SAMPLING ─────────────────────────────────────────────
    if strat == "typical":
        typical_p = args.typical_p 
        gen_params["do_sample"] = True
        gen_params["typical_p"] = typical_p
        gen_params["temperature"] = 1.0
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 7) EPSILON SAMPLING ─────────────────────────────────────────────
    if strat == "eta":
        eps = args.eta_cutoff 
        gen_params["do_sample"] = True
        gen_params["temperature"] = 1.0
        gen_params["eta_cutoff"] = eps
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params
    
     # ─── 7) MIN P ─────────────────────────────────────────────
    if strat == "min_p":
        minp = args.min_p 
        gen_params["do_sample"] = True
        gen_params["temperature"] = 1.0
        gen_params["min_p"] = minp
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 8) CONTRASTIVE SEARCH ────────────────────────────────────────────
    if strat == "contrastive":
        alpha = args.cs_alpha 
        kval  = args.cs_k  
        gen_params["penalty_alpha"] = alpha
        gen_params["top_k"] = kval
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 9) DOLA (model‐specific) ─────────────────────────────────────────
    if strat == "dola":
        layers = args.dola_layers 
        gen_params["dola_layers"] = layers
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── 10) ASSISTED (prompt lookup) ─────────────────────────────────────
    if strat == "assisted":
        lookup_n = args.prompt_lookup_num_tokens 
        gen_params["prompt_lookup_num_tokens"] = lookup_n
        # gen_params["use_cache"] = False
        print(f"[INFO] Final gen_params = {gen_params}\n")
        return gen_params

    # ─── UNKNOWN STRATEGY ─────────────────────────────────────────────────
    raise ValueError(f"[ERROR] Unrecognized decoding_strategy: {args.decoding_strategy}")

def format_output_filename(args):
    """
    Build a unique output JSONL path based on task/model/strategy and relevant hyperparameters.
    """
    base_name = args.model.split("/")[-1].replace("-", "")
    suffix = [args.decoding_strategy]

    strat = args.decoding_strategy.lower()
    if strat == "beam_search":
        beams = args.num_beams
        suffix.append(f"beams{beams}")
       
    elif strat == "diverse_beam_search":
        beams = args.num_beams 
        suffix.append(f"divbeams{beams}")
        if args.num_beam_groups is not None:
            suffix.append(f"groups{args.num_beam_groups}")

    elif strat == "top_k":
        k = args.top_k
        suffix.append(f"topk{k}")

    elif strat == "top_p":
        p = args.top_p
        suffix.append(f"topp{p}")

    elif strat == "temperature":
        t = args.temperature 
        suffix.append(f"temp{t}")

    elif strat == "typical":
        tp = args.typical_p
        suffix.append(f"typical{tp}")

    elif strat == "eta":
        e = args.eta_cutoff 
        suffix.append(f"eps{e}")

    elif strat == "contrastive":
        a = args.cs_alpha 
        k = args.cs_k   
        suffix.append(f"csα{a}_k{k}")

    elif strat == "dola":
        layers = args.dola_layers 
        suffix.append(f"layers{layers}")

    elif strat == "assisted":
        lookup_n = args.prompt_lookup_num_tokens 
        suffix.append(f"lookup{lookup_n}")
    
    elif strat == "min_p":
        p = args.min_p
        suffix.append(f"min_p{p}")

    final_suffix = "_".join(suffix)
    out_dir = os.path.join("data", args.task, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base_name}_{final_suffix}_oneshot.jsonl")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA not available. Using CPU.")
        
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(0)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ─── Required ─────────────────────────────────────────────────────────────
    parser.add_argument("--task", required=True,
                        help="Name of task (used for data/<task>/data.jsonl)")
    parser.add_argument("--model", required=True,
                        help="Model identifier")
    parser.add_argument("--decoding_strategy", required=True,
                        choices=[
                            "greedy", "beam_search", "diverse_beam_search", "top_k", "top_p",
                            "temperature", "typical", "eta",
                            "contrastive", "dola", "assisted", "min_p"
                        ],
                        help="Decoding method to use. Hyperparameters default if omitted.")

    # ─── Optional overrides ─────────────────────────────────────────────────────
    parser.add_argument("--input_file",  type=str, default=None,
                        help="If not set, uses data/<task>/data.jsonl")
    parser.add_argument("--output_file", type=str, default=None,
                        help="If not set, uses data/<task>/outputs/<…>.jsonl")
    parser.add_argument("-f", "--image_folder", type=str, default=None,
                        help="Folder where the images are.")

    # ─── Common generation flags (only used if your strategy needs them) ───────
    parser.add_argument("--max_new_tokens", type=int, default=1500,
                        help="How many tokens to generate after the prompt")

    # ─── Beam‐search flags ─────────────────────────────────────────────────────
    parser.add_argument("--num_beams", type=int, help="Number of beams if using beam_search")
    parser.add_argument("--num_beam_groups", type=int,
                        help="Number of beam groups (for diverse‐beam, must divide num_beams)")
    parser.add_argument("--diversity_penalty", type=float,
                        help="Diversity penalty for diverse‐beam search")

    # ─── Sampling flags ─────────────────────────────────────────────────────────
    parser.add_argument("--top_k", type=int, help="If using top_k sampling: k")
    parser.add_argument("--top_p", type=float, help="If using top_p sampling: p")
    parser.add_argument("--temperature", type=float, help="If using temperature sampling: temp")
    parser.add_argument("--min_p", type=float)

    # ─── Advanced sampling ─────────────────────────────────────────────────────
    parser.add_argument("--typical_p", type=float, help="If using typical sampling: typical_p")
    parser.add_argument("--eta_cutoff", type=float,
                        help="If using eta sampling: eta_cutoff")

    # ─── Contrastive search ─────────────────────────────────────────────────────
    parser.add_argument("--cs_alpha", type=float,
                        help="If using contrastive: penalty_alpha (default 0.6)")
    parser.add_argument("--cs_k", type=int,
                        help="If using contrastive: top_k (default 4)")

    # ─── DOLA & Assisted (model‐specific) ───────────────────────────────────────
    parser.add_argument("--dola_layers", type=str,
                        help="If using DOLA: which layers (comma‐sep) (default 'Low')")
    parser.add_argument("--prompt_lookup_num_tokens", type=int,
                        help="If using 'assisted': how many tokens to look up (default 5)")

    args = parser.parse_args()

    is_llava = "llava-hf/llava-1.5-7b-hf" in args.model
    is_gemma_12b = "google/gemma-3-12b-it" in args.model
    is_gemma_1b = "google/gemma-3-1b-it" in args.model
    is_biomistral = "BioMistral/BioMistral-7B" in args.model
    is_medgemma = "google/medgemma-4b-it" in args.model
    is_meditron = "epfl-llm/meditron-7b" in args.model
    llama_13b = "meta-llama/Llama-2-13b-chat-hf" in args.model

    if args.task == "images" and not args.image_folder:
        parser.error("--image_folder is required when using multimodal models")

    if is_llava:
        model, processor = load_llava_model(args.model)
        gen_params = build_generation_params(
            args,
            tokenizer=processor.tokenizer, 
        )

    elif is_gemma_1b:
        model, tokenizer = load_gemma_1b_model(args.model)
        processor = tokenizer  
        gen_params = build_generation_params(args, tokenizer)
    
    elif is_gemma_12b:
        model, processor = load_gemma_12b_model(args.model)
        gen_params = build_generation_params(args,tokenizer=processor.tokenizer)

    elif is_medgemma:
        model, processor = load_medgemma_model(args.model)
        gen_params = build_generation_params(args,tokenizer=processor.tokenizer)

    else:
        model, tokenizer = load_model_and_tokenizer(args.model)
        gen_params = build_generation_params(args,tokenizer)

    # ─── Determine input & output paths and print them ─────────────────────────
    input_file = args.input_file or f"data/{args.task}/data.jsonl"
    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"[ERROR] Input file not found: {input_file!r}\n"
            f"Either create data/{args.task}/data.jsonl or pass --input_file."
        )
    print(f"[INFO] Using input file:  {input_file}")

    output_file = args.output_file or format_output_filename(args)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[INFO] Will write outputs to:  {output_file}\n")

    # Load all samples and extract one-shot example
    with open(input_file, encoding='utf-8') as f:
        all_samples = [json.loads(line) for line in f]

    example = next(sample for sample in all_samples if sample["id"] == 99)
    example_prompt = example["prompt"].strip()
    example_tgt = example["tgt"].strip()

    # ex_img_path = os.path.join(args.image_folder, example["src"])
    # ex_image = Image.open(ex_img_path).convert("RGB")

    with open(input_file, encoding='utf-8') as in_f, open(output_file, 'w', encoding='utf-8') as out_f:
        for line in in_f:
            data = json.loads(line)
            if data["id"] > 98:
                continue

            instruction = data["instruction"].strip()
            prompt = data["prompt"].strip()
            rec_id = data["id"]

            # Build messages depending on whether one-shot example is used
            if llama_13b:
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": example_prompt},
                    {"role": "assistant", "content": example_tgt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ]

            # Handle each model type
            if is_llava:
                img_path = os.path.join(args.image_folder, data["src"])
                image = Image.open(img_path).convert("RGB")

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": f"{instruction}. {prompt}"}
                        ]
                    }
                ]

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors='pt'
                )

                inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

                with torch.no_grad():
                    start = time.time()
                    gen_ids = model.generate(**inputs, **gen_params)
                    elapsed = time.time() - start

                out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                if "ASSISTANT:" in out_text:
                    out_text = out_text.split("ASSISTANT:")[-1].strip()

            elif is_gemma_1b or is_gemma_12b or is_medgemma:
                if args.task == "images":
                    img_path = os.path.join(args.image_folder, data["src"])
                    image = Image.open(img_path).convert("RGB")

                    messages = [
                        {"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text",  "text": instruction + prompt}
                        ]}
                    ]
                    
                else:
                    content_type = [{"type": "text", "text": prompt}]
                    if is_gemma_12b or is_medgemma:
                        content_type = [{"type": "text", "text": prompt}]

                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": instruction}]},
                        {"role": "user", "content": content_type}
                    ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    start = time.time()
                    generation = model.generate(**inputs, **gen_params)
                    elapsed = time.time() - start
                    generation = generation[0][input_len:]

                out_text = processor.decode(generation, skip_special_tokens=True).strip()

            else:
                # Handle standard chat template or meditron-style input
                if is_biomistral:
                    messages = [{"role": "user", "content": f"{instruction}{prompt}"}]

                if getattr(tokenizer, 'chat_template', None) and not is_meditron:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                elif is_meditron:
                    text = (
                        f"<|im_start|>system\n{instruction}<|im_end|>\n"
                        f"<|im_start|>question\n{example_prompt}<|im_end|>\n"
                        f"{example_tgt}\n"
                        f"<|im_start|>question\n{prompt}<|im_end|>"
                    )
                else:
                    text = f"{instruction}\n\n{example_prompt}\n{example_tgt}\n\n{prompt}"

                inputs = tokenizer([text], return_tensors="pt").to(model.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                start = time.time()
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_params
                )
                elapsed = time.time() - start
                new_tokens = outputs[0, input_ids.size(1):]
                out_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            result = {
                "id": rec_id,
                "output": out_text,
                "duration_sec": round(elapsed, 3)
            }
            json.dump(result, out_f, ensure_ascii=False)
            out_f.write("\n")

        
        print(f"Done ⏱  Outputs written to {output_file}")


if __name__ == "__main__":
    main()