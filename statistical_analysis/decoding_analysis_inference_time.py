#%%
import os, glob, re
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import spearmanr, kendalltau

DATA_DIR = "./"
METRIC_SUBSTR = "ROUGE"         
DURATION_SUBSTR = "sec/tok"     

metric_rows = []
duration_rows = []

for fp in glob.glob(os.path.join(DATA_DIR, "results_*.csv")):
    task = os.path.basename(fp).replace("results_", "").replace(".csv", "")
    df_file = pd.read_csv(fp)
    if "strategy" not in df_file.columns:
        continue
    df_file = df_file.set_index("strategy")

    metric_cols = [c for c in df_file.columns if METRIC_SUBSTR in c.upper()]
    for col in metric_cols:
        model = re.sub(rf"_{re.escape(METRIC_SUBSTR)}$", "", col, flags=re.IGNORECASE)
        for strat, score in df_file[col].items():
            metric_rows.append({"Task": task, "Model": model, "Strategy": strat, "Metric": score})

    dur_cols = [c for c in df_file.columns if DURATION_SUBSTR in c.lower()]
    for col in dur_cols:
        model = re.sub(r"_sec/tok$", "", col, flags=re.IGNORECASE)
        for strat, duration in df_file[col].items():
            duration_rows.append({
                "Task": task,
                "Model": model,
                "Strategy": strat,
                "Duration_sec_per_tok": duration
            })

# normalize per task
long_df = pd.DataFrame(metric_rows)
if not long_df.empty:
    long_df["Metric"] = long_df.groupby("Task")["Metric"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )

duration_df = pd.DataFrame(duration_rows)

df = long_df.merge(duration_df, on=["Task", "Model", "Strategy"], how="left")
# %% Average Decoding Time per Strategy 

if duration_df.empty:
    print("No duration columns found (matching 'sec/tok').")
else:
    df["Duration_norm"] = df.groupby("Task")["Duration_sec_per_tok"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )

    strategy_durations_norm = (
        df.groupby("Strategy", dropna=False)["Duration_norm"]
          .mean()
          .sort_values()
          .reset_index()
    )

    strategy_durations_raw = (
        df.groupby("Strategy", dropna=False)["Duration_sec_per_tok"]
          .mean()
          .sort_values()
          .reset_index()
    )

    print("\n=== Average Decoding Time per Strategy (normalized 0–1, per task) ===")
    print(strategy_durations_norm)

    print("\n=== Average Decoding Time per Strategy (sec/token, raw mean) ===")
    print(strategy_durations_raw)

# %% Friedman test for decoding speed across strategies
df["Block"] = df["Task"] + " | " + df["Model"]

duration_pivot = df.pivot_table(index="Block", columns="Strategy", values="Duration_sec_per_tok")

duration_pivot = duration_pivot.dropna()

if duration_pivot.shape[1] > 1:
    stat, p = friedmanchisquare(*[duration_pivot[col].values for col in duration_pivot.columns])
    print(f"\nFriedman test on decoding speed:")
    print(f"χ² = {stat:.2f}, p = {p:.4f}")
else:
    print("Not enough complete duration data across strategies.")

# %%
performance_by_strategy = (
    df.groupby("Strategy", dropna=False)["Metric"]
      .mean()
      .reset_index()
      .rename(columns={"Metric": "Mean_ROUGE"})
)

duration_by_strategy = (
    df.groupby("Strategy", dropna=False)[["Duration_sec_per_tok", "Duration_norm"]]
      .mean()
      .reset_index()
      .rename(columns={
          "Duration_sec_per_tok": "Mean_sec_per_tok",
          "Duration_norm": "Mean_Duration_norm"
      })
)

strategy_summary = performance_by_strategy.merge(duration_by_strategy, on="Strategy", how="inner")

strategy_summary = strategy_summary.dropna(subset=["Mean_ROUGE", "Mean_sec_per_tok", "Mean_Duration_norm"])

rho_raw, p_rho_raw = spearmanr(strategy_summary["Mean_sec_per_tok"], strategy_summary["Mean_ROUGE"])
tau_raw, p_tau_raw = kendalltau(strategy_summary["Mean_sec_per_tok"], strategy_summary["Mean_ROUGE"])

rho_norm, p_rho_norm = spearmanr(strategy_summary["Mean_Duration_norm"], strategy_summary["Mean_ROUGE"])
tau_norm, p_tau_norm = kendalltau(strategy_summary["Mean_Duration_norm"], strategy_summary["Mean_ROUGE"])

print("\nCorrelation between decoding time and strategy performance:")
print(f"Spearman ρ (raw sec/token) = {rho_raw:.4f}, p = {p_rho_raw:.4f}")
print(f"Kendall’s τ (raw sec/token) = {tau_raw:.4f}, p = {p_tau_raw:.4f}")
print(f"Spearman ρ (normalized 0–1) = {rho_norm:.4f}, p = {p_rho_norm:.4f}")
print(f"Kendall’s τ (normalized 0–1) = {tau_norm:.4f}, p = {p_tau_norm:.4f}")# %%


# %%
