#%%
import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, norm
from itertools import combinations


METRIC = "ROUGE"         

rows = []
for fp in glob.glob(os.path.join("./", "results_*.csv")):
    task = os.path.basename(fp).replace("results_", "").replace(".csv", "")
    df = pd.read_csv(fp)
    if "strategy" not in df.columns:
        continue
    df = df.set_index("strategy")
    metric_cols = [c for c in df.columns if METRIC in c.upper()]
    for col in metric_cols:
        model = re.sub(rf"_{METRIC}$", "", col, flags=re.IGNORECASE)
        for strat, score in df[col].items():
            rows.append({"Task": task, "Model": model, "Strategy": strat, "Metric": score})

long_df = pd.DataFrame(rows)

df = long_df.copy()

# normalize metrics within each task
df["Metric"] = df.groupby("Task")["Metric"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
)

#%% FRIEDMAN OVERALL 

# Rank within each task × model block 
df["Rank"] = df.groupby(["Task", "Model"])["Metric"].rank(
    ascending=False, method="average"
)

df["Block"] = df["Task"] + " | " + df["Model"]

# Pivot to Block × Strategy ranks
all_strats = sorted(df["Strategy"].unique())     

P = df.pivot_table(index="Block", columns="Strategy", values="Rank")
P = P.reindex(columns=all_strats)                 

# Filter out blocks with missing values 
mask = P.notna().all(axis=1)
P_full = P.loc[mask]

# Friedman 
N, k = P_full.shape
stat, p = friedmanchisquare(*[P_full[c].values for c in P_full.columns])
mean_ranks = P_full.mean(axis=0).sort_values()
names = mean_ranks.index.tolist()
x = mean_ranks.values

print(f"Using {N} blocks and {k} strategies.")
print(f"Friedman χ²={stat:.2f}, p={p:.3g}")

mean_rank_table = mean_ranks.reset_index()
mean_rank_table.columns = ["Strategy", "Mean Rank"]
print(mean_rank_table)

#%% WILCOXON OVERALL

rows = []
for a, b in combinations(P.columns, 2):
    sub = P[[a, b]].dropna()
    d = sub[a] - sub[b] # ranks: negative → A better (lower)
    n_eff = np.count_nonzero(d != 0)
    if n_eff < 5:
        continue
    try:
        stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = (np.nan, 1.0)
    sgn = np.sign((d > 0).sum() - (d < 0).sum())
    z = sgn * (norm.isf(p/2) if p > 0 else 0.0)
    r = z / np.sqrt(n_eff)

    rows.append([
        a, b, len(sub), np.median(d),
        (d < 0).sum(), (d == 0).sum(), (d > 0).sum(),
        stat, p, z, r
    ])

details = pd.DataFrame(rows, columns=[
    "A","B","N_used","median_diff","A_wins","ties","B_wins","W","p_raw","z","r"
])

# Holm step-down on the raw p-values
details = details.sort_values("p_raw").reset_index(drop=True)
m = len(details)
details["p_holm"] = ((m - details.index) * details["p_raw"]).clip(upper=1.0)
details["p_holm"] = details["p_holm"].cummax()
details["signif_0.05"] = details["p_holm"] <= 0.05

def _dir(md):
    return "A better" if md < 0 else ("B better" if md > 0 else "tie")
details["direction"] = details["median_diff"].apply(_dir)

cols = ["A","B","N_used","median_diff","A_wins","ties","B_wins","W",
        "r","p_raw","p_holm","signif_0.05","direction"]
details[["median_diff","r","p_raw","p_holm"]] = (
    details[["median_diff","r","p_raw","p_holm"]].round(4)
)

print("\n Wilcoxon pairwise (Holm-adjusted)")
print(details[cols].to_string(index=False))

# %% FRIEDMAN PER TASK

task_results = []
# Loop through each task and do Friedman 
for task, sub_df in df.groupby("Task"):
    pivot = sub_df.pivot_table(index='Model', columns='Strategy', values='Rank')

    complete = pivot.dropna()
    
    if complete.shape[0] < 2 or complete.shape[1] < 2:
        continue

    stat, p = friedmanchisquare(*[complete[col].values for col in complete.columns])
    task_results.append({'Task': task, 'Friedman_stat': stat, 'Friedman_p': p})

task_results_df = pd.DataFrame(task_results)
print(task_results_df.sort_values("Friedman_p"))

# %% Just to see the best strategy per task 

best_strategy_df = (
    df.groupby(["Task", "Strategy"])["Rank"]
    .mean()
    .reset_index()
    .sort_values(["Task", "Rank"])
    .groupby("Task")
    .first()
    .reset_index()
)

print("\n Best strategy per task (by mean rank)")
print(best_strategy_df)

# %%  WILCOXON PER TASK

all_task_wilcoxon = []

for task, sub_df in df.groupby("Task"):
    pivot = sub_df.pivot_table(index="Model", columns="Strategy", values="Rank")

    # Need at least 2 strategies and at least 2 models overall to attempt pairs
    if pivot.shape[1] < 2 or pivot.shape[0] < 2:
        continue

    print(f"\n=== Task: {task} ===")

    comparisons = []
    for a, b in combinations(pivot.columns, 2):
        sub = pivot[[a, b]].dropna()
        if sub.shape[0] < 2:
            continue

        d = sub[a] - sub[b]  
        n_eff = (d != 0).sum()
        if n_eff < 5:
            continue

        try:
            stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = (np.nan, 1.0)

        sgn = np.sign((d > 0).sum() - (d < 0).sum())
        z = sgn * (norm.isf(p/2) if p > 0 else 0.0)
        r = z / np.sqrt(n_eff)

        comparisons.append({
            "Task": task,
            "A": a,
            "B": b,
            "N": len(d),                     
            "N_eff": int(n_eff),               # non-tied pairs actually used by Wilcoxon
            "median_diff": float(np.median(d)),
            "A_wins": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
            "B_wins": int((d > 0).sum()),
            "W": stat,
            "p_raw": p,
            "z": z,
            "r": r
        })

    if not comparisons:
        continue

    pairwise_df = pd.DataFrame(comparisons).sort_values("p_raw").reset_index(drop=True)

    # Holm 
    m = len(pairwise_df)
    pairwise_df["p_holm"] = ((m - pairwise_df.index) * pairwise_df["p_raw"]).clip(upper=1.0)
    pairwise_df["p_holm"] = pairwise_df["p_holm"].cummax()
    pairwise_df["signif_0.05"] = pairwise_df["p_holm"] <= 0.05

    pairwise_df["direction"] = pairwise_df["median_diff"].apply(
        lambda md: "A better" if md < 0 else ("B better" if md > 0 else "tie")
    )

    # Print significant results
    sig = pairwise_df[pairwise_df["signif_0.05"]]
    if not sig.empty:
        print("Significant pairwise differences (Wilcoxon, Holm-corrected):")
        for _, row in sig.iterrows():
            better = row["A"] if row["median_diff"] < 0 else row["B"]
            worse  = row["B"] if row["median_diff"] < 0 else row["A"]
            gap    = abs(row["median_diff"])
            print(f"- {better} > {worse} (median rank gap {gap:.2f}, "
                  f"r={abs(row['r']):.2f}, N={row['N']}, p_Holm={row['p_holm']:.3g})")

    all_task_wilcoxon.append(pairwise_df)

# %% DETERMINISTIC vs STOCHASTIC STRATEGIES

det_list = ["greedy","beam search","diverse beam search","contrastive search","DoLa"]
sto_list = ["Temperature","top p","top k","min p sampling","eta","typical"]
det_set = {s for s in det_list}
sto_set = {s for s in sto_list}

block_rows = []
for (task, model), g in long_df.groupby(["Task","Model"]):
    det_vals = g[g["Strategy"].isin(det_set)]["Metric"]
    sto_vals = g[g["Strategy"].isin(sto_set)]["Metric"]
    if det_vals.empty or sto_vals.empty:
        continue
    diff_mean = float(det_vals.mean() - sto_vals.mean())
    block_rows.append({"Task": task, "Model": model, "diff_mean": diff_mean})

blocks = pd.DataFrame(block_rows)
d = blocks["diff_mean"].values

W, p = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
median_diff = float(np.median(d))
n_nonzero = int(np.sum(d != 0))  # number of non-zero pairs used by Wilcoxon

print(f"DET vs STOCH Wilcoxon (mean-of): W={W}, n={n_nonzero}, p={p:.4g}; "
      f"median ΔROUGE_mean={median_diff:.4f}")

# PER TASK Wilcoxon det vs stoch
per_task_rows = []
for task, sub in blocks.groupby("Task"):
    d_task = sub["diff_mean"].values
    try:
        W_t, p_t = wilcoxon(d_task, alternative="two-sided", zero_method="wilcox")
        n_nz_t = int(np.sum(d_task != 0))
        med_t = float(np.median(d_task)) if len(d_task) else np.nan
    except ValueError:
        W_t, p_t, n_nz_t, med_t = np.nan, np.nan, 0, np.nan

    per_task_rows.append({
        "Task": task,
        "N_blocks": len(d_task),
        "n_nonzero": n_nz_t,
        "W": W_t,
        "p": p_t,
        "median_diff_mean": med_t
    })

per_task = pd.DataFrame(per_task_rows).sort_values("Task")
print("\n DET vs STOCH per task Wilcoxon (mean-of)")
print(per_task.to_string(index=False))

# %%
