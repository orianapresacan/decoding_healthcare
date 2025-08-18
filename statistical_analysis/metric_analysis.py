# %%
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

DATA_DIR = "./"
FILE_PATTERN = "results_*.csv"
CI_METHOD = "normal"  # choose: "bootstrap" or "normal"
ALPHA = 0.05
N_BOOT = 10000
RNG_SEED = 0

METRIC_SUFFIXES = {
    "ROUGE": "ROUGE",
    "BERTSCORE": "BERTScore",
    "BLEU": "BLEU",
    "MAUVE": "MAUVE",
}

TASK_LABELS = {
    "imagecaptioning": "Captioning",
    "dialogue": "Dialogue",
    "qa": "QA",
    "summarization": "Summarization",
    "translation": "Translation",
}

TASK_ORDER_P1_DISPLAY = ["Captioning", "Dialogue", "QA", "Summarization", "Translation"]


def parse_model_and_metric(col: str):
    for raw, canon in METRIC_SUFFIXES.items():
        if re.search(rf"_{raw}$", col, flags=re.IGNORECASE):
            model = re.sub(rf"_{raw}$", "", col, flags=re.IGNORECASE)
            return model, canon
    return None, None

def load_long_df(data_dir=DATA_DIR, pattern=FILE_PATTERN):
    rows = []
    for fp in sorted(glob.glob(os.path.join(data_dir, pattern))):
        task_key = os.path.basename(fp).replace("results_", "").replace(".csv", "")
        df = pd.read_csv(fp)
        if "strategy" not in df.columns:
            continue
        df = df.rename(columns={"strategy": "Strategy"})
        for col in df.columns:
            if col == "Strategy":
                continue
            model, metric = parse_model_and_metric(col)
            if model is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            for strat, v in zip(df["Strategy"], vals):
                if pd.notna(v):
                    rows.append({
                        "Task": task_key,
                        "Model": model,
                        "Strategy": strat,
                        "Metric": metric,
                        "Value": float(v),
                    })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError("No metric columns parsed. Check your CSV headers and METRIC_SUFFIXES.")
    long_df["Block"] = long_df["Task"] + " | " + long_df["Model"]
    return long_df

df_long = load_long_df()


def block_pair_tau(pivot_block, m1, m2, min_strats=3):
    if m1 not in pivot_block or m2 not in pivot_block:
        return np.nan
    sb = pivot_block[[m1, m2]].dropna()
    if len(sb) < min_strats:
        return np.nan
    tau, _ = kendalltau(sb[m1], sb[m2])  # tau-b
    return tau

def summarize_mean_ci_normal(values, alpha=0.05):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = arr.size
    if n == 0:
        return np.nan, (np.nan, np.nan), 0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    z = 1 - alpha/2
    # 1.96 is fine for alpha=0.05, but keep general:
    from scipy.stats import norm
    crit = norm.ppf(z)
    lo, hi = mean - crit * se, mean + crit * se
    return mean, (float(lo), float(hi)), int(n)

def summarize_mean_ci_boot(values, alpha=0.05, n_boot=10000, seed=0):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = arr.size
    if n == 0:
        return np.nan, (np.nan, np.nan), 0
    if n == 1:
        return float(arr[0]), (np.nan, np.nan), 1
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
    return float(arr.mean()), (float(lo), float(hi)), int(n)

def summarize(values, method="bootstrap", alpha=0.05, n_boot=10000, seed=0):
    if method == "normal":
        return summarize_mean_ci_normal(values, alpha=alpha)
    return summarize_mean_ci_boot(values, alpha=alpha, n_boot=n_boot, seed=seed)

def mean_tau_over_blocks(sub_df: pd.DataFrame, metricA: str, metricB: str):
    taus = []
    for _, sb in sub_df.groupby("Block"):
        pivot = sb.pivot_table(index="Strategy", columns="Metric", values="Value", aggfunc="mean")
        tau = block_pair_tau(pivot, metricA, metricB)
        if np.isfinite(tau):
            taus.append(tau)
    return summarize(taus, method=CI_METHOD, alpha=ALPHA, n_boot=N_BOOT, seed=RNG_SEED)

# %% Plot 1: ROUGE ↔ BERTScore (overall + per task)

overall_mean, (overall_lo, overall_hi), overall_n = mean_tau_over_blocks(df_long, "ROUGE", "BERTScore")

rows_p1 = [{
    "Scope": "Overall",
    "Pair":  "ROUGE↔BERTScore",
    "Tau": overall_mean, "CI_lo": overall_lo, "CI_hi": overall_hi, "n_blocks": overall_n
}]

for task_key, sub_t in df_long.groupby("Task"):
    mean_t, (lo_t, hi_t), n_t = mean_tau_over_blocks(sub_t, "ROUGE", "BERTScore")
    scope_label = TASK_LABELS.get(task_key, task_key)
    rows_p1.append({
        "Scope": scope_label,
        "Pair":  "ROUGE↔BERTScore",
        "Tau": mean_t, "CI_lo": lo_t, "CI_hi": hi_t, "n_blocks": n_t
    })

df_p1 = pd.DataFrame(rows_p1)
present = set(df_p1["Scope"])
order = ["Overall"] + [lab for lab in TASK_ORDER_P1_DISPLAY if lab in present]
dfp = df_p1.copy()
dfp["Scope"] = pd.Categorical(dfp["Scope"], categories=order, ordered=True)
dfp = dfp.sort_values("Scope").reset_index(drop=True)

y = np.arange(len(dfp))
xerr_lo = dfp["Tau"] - dfp["CI_lo"]
xerr_hi = dfp["CI_hi"] - dfp["Tau"]

plt.figure(figsize=(8, 0.55*len(dfp) + 0.4))
plt.axvline(0.0, linestyle="--", linewidth=1, color='rosybrown')
plt.errorbar(dfp["Tau"], y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3, color='rosybrown')
plt.yticks(y, [f"{sc}" for sc in dfp["Scope"]])
xmax = max(dfp["CI_hi"].max(), 0.0)
for i, n in enumerate(dfp["n_blocks"]):
    plt.text(xmax + 0.05, i, f"n={n}", va="center")
plt.xlim(-1.0, 1.0)
plt.xlabel("Kendall's τ (rank correlation)")
plt.title("Metric agreement: ROUGE ↔ BERTScore")
plt.tight_layout()
plt.show()

# %% Plot 2: BLEU/MAUVE pairs
rows_p2 = []

if "translation" in df_long["Task"].unique():
    sub_tr = df_long[df_long["Task"] == "translation"]
    for (a, b) in [("BLEU","ROUGE"), ("BLEU","BERTScore")]:
        mean_t, (lo_t, hi_t), n_t = mean_tau_over_blocks(sub_tr, a, b)
        rows_p2.append({
            "Scope": "Translation",
            "Pair": f"{a}↔{b}",
            "Tau": mean_t, "CI_lo": lo_t, "CI_hi": hi_t, "n_blocks": n_t
        })

sub_dq = df_long[df_long["Task"].isin(["dialogue","qa"])]
if not sub_dq.empty:
    for (a, b) in [("MAUVE","ROUGE"), ("MAUVE","BERTScore")]:
        mean_t, (lo_t, hi_t), n_t = mean_tau_over_blocks(sub_dq, a, b)
        rows_p2.append({
            "Scope": "Dialogue+QA",
            "Pair": f"{a}↔{b}",
            "Tau": mean_t, "CI_lo": lo_t, "CI_hi": hi_t, "n_blocks": n_t
        })

df_p2 = pd.DataFrame(rows_p2).sort_values("Tau", ascending=True).reset_index(drop=True)

y = np.arange(len(df_p2))
xerr_lo = df_p2["Tau"] - df_p2["CI_lo"]
xerr_hi = df_p2["CI_hi"] - df_p2["Tau"]

plt.figure(figsize=(8, 0.55*len(df_p2) + 0.4))
plt.axvline(0.0, linestyle="--", linewidth=1, color='steelblue')
plt.errorbar(df_p2["Tau"], y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3, color='steelblue')
plt.yticks(y, [f"{sc} · {pa}" for sc, pa in zip(df_p2["Scope"], df_p2["Pair"])])
xmax = max(df_p2["CI_hi"].max(), 0.0)
for i, n in enumerate(df_p2["n_blocks"]):
    plt.text(xmax + 0.05, i, f"n={n}", va="center")
plt.xlim(-1.0, 1.0)
plt.xlabel("Kendall's τ (rank correlation)")
plt.title("Metric agreement: BLEU and MAUVE")
plt.tight_layout()
plt.show()

print("\nPlot 1 rows (ROUGE↔BERTScore):\n", df_p1.to_string(index=False))
print("\nPlot 2 rows (BLEU/MAUVE pairs):\n", df_p2.to_string(index=False))

# %% Metric sensitivity to decoding (CV)

def block_metric_cv(values: pd.Series, eps=1e-12, robust=False):
    """
    Compute CV across strategies within a block for a single metric.
    values: 1D numeric pd.Series of metric values across strategies (dropna beforehand)
    - Classic CV: std / mean
    - Robust CV (optional): 1.4826*MAD / median
    """
    x = pd.to_numeric(values, errors="coerce").dropna().values
    if x.size == 0:
        return np.nan
    if robust:
        med = np.median(x)
        if med == 0:
            return np.nan
        mad = np.median(np.abs(x - med))
        return 1.4826 * mad / (np.abs(med) + eps)
    else:
        mu = np.mean(x)
        sd = np.std(x, ddof=1) if x.size > 1 else 0.0
        return sd / (np.abs(mu) + eps)

def mean_cv_over_blocks(sub_df: pd.DataFrame, metric: str, min_strats=3, robust=False):
    """
    For a given sub-dataframe (e.g., a task subset) and a metric,
    compute the CV across strategies within each Block,
    then summarize the mean CV (with CI) across Blocks.
    """
    cvs = []
    for _, sb in sub_df[sub_df["Metric"] == metric].groupby("Block"):
        # pivot to strategies -> values for this metric
        pv = sb.pivot_table(index="Strategy", values="Value", aggfunc="mean")
        pv = pv.dropna()
        if len(pv) < min_strats:
            continue
        cv = block_metric_cv(pv["Value"], robust=robust)
        if np.isfinite(cv):
            cvs.append(cv)
    return summarize(cvs, method=CI_METHOD, alpha=ALPHA, n_boot=N_BOOT, seed=RNG_SEED)

rows_cv = []
metrics_present = sorted(df_long["Metric"].unique())

# Overall
for m in metrics_present:
    mean_cv, (lo, hi), n = mean_cv_over_blocks(df_long, m, min_strats=3, robust=False)
    rows_cv.append({
        "Scope": "Overall",
        "Metric": m,
        "Mean_CV": mean_cv, "CI_lo": lo, "CI_hi": hi, "n_blocks": n
    })

# Per task
for task_key, sub_t in df_long.groupby("Task"):
    scope_label = TASK_LABELS.get(task_key, task_key)
    for m in sorted(sub_t["Metric"].unique()):
        mean_cv, (lo, hi), n = mean_cv_over_blocks(sub_t, m, min_strats=3, robust=False)
        rows_cv.append({
            "Scope": scope_label,
            "Metric": m,
            "Mean_CV": mean_cv, "CI_lo": lo, "CI_hi": hi, "n_blocks": n
        })

df_cv = pd.DataFrame(rows_cv).dropna(subset=["Mean_CV"]).reset_index(drop=True)



print("\nMetric sensitivity to decoding (CV):")
print(df_cv.to_string(index=False))

# %%
