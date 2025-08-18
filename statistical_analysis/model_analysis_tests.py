#%%
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, kendalltau, levene
import re

data_dir = "./"
file_pattern = os.path.join(data_dir, "results_*.csv")

long_data = []

for filepath in glob.glob(file_pattern):
    task = os.path.basename(filepath).replace("results_", "").replace(".csv", "")
    df = pd.read_csv(filepath)

    if "strategy" not in df.columns:
        continue

    df = df.set_index("strategy")

    duration_cols = [col for col in df.columns if "sec/tok" in col.lower()]
    rouge_cols = [col for col in df.columns if "ROUGE" in col.upper()]

    for model_col in rouge_cols:
        for strategy, score in df[model_col].items():
            long_data.append({
                "Task": task,
                "Model": model_col,
                "Strategy": strategy,
                "ROUGE": score
            })

    for model_col in duration_cols:
        model_rouge_name = model_col.replace("_sec/tok", "_ROUGE")
        for strategy, sec_per_tok in df[model_col].items():
            long_data.append({
                "Task": task,
                "Model": model_rouge_name,
                "Strategy": strategy,
                "sec_per_tok": sec_per_tok
            })

long_df = pd.DataFrame(long_data)

#%%
# Add model type and model size 
def extract_size(model_name):
    match = re.search(r'(\d+(\.\d+)?)[Bb]', model_name)
    return float(match.group(1)) if match else None

def is_medical_model(model_name):
    return any(tag in model_name.lower() for tag in ["bio", "med"])

long_df["ModelType"] = long_df["Model"].apply(lambda x: "Medical" if is_medical_model(x) else "General")
long_df["ModelSize"] = long_df["Model"].apply(extract_size)

# Normalize ROUGE per task ===
long_df["Normalized_ROUGE"] = long_df.groupby("Task")["ROUGE"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# %%  Mann–Whitney U test (Medical vs. General models - Normalized ROUGE)

model_summary = long_df.groupby('Model')[['Normalized_ROUGE']].mean().reset_index()

model_summary['ModelType'] = model_summary['Model'].apply(
    lambda x: 'Medical' if is_medical_model(x) else 'General'
)
model_summary['ModelSize'] = model_summary['Model'].apply(extract_size)

general_vals = model_summary[model_summary['ModelType'] == 'General']['Normalized_ROUGE']
medical_vals = model_summary[model_summary['ModelType'] == 'Medical']['Normalized_ROUGE']

stat, p = mannwhitneyu(medical_vals, general_vals, alternative='two-sided')

print("Mann–Whitney U test (Medical vs. General models):")
print(f"stat = {stat:.4f}, p = {p:.4f}")
print(f"General mean = {general_vals.mean():.4f}")
print(f"Medical mean = {medical_vals.mean():.4f}")

# %% Model Size vs. Performance (Normalized ROUGE)

filtered_df = long_df[~long_df['Model'].str.contains("llava", case=False)]

filtered_general = filtered_df[filtered_df['ModelType'] == 'General']

model_avg = filtered_general.groupby('Model')[['ModelSize', 'Normalized_ROUGE']].mean().dropna()
rho, p_rho = spearmanr(model_avg["ModelSize"], model_avg["Normalized_ROUGE"])
tau, p_tau = kendalltau(model_avg["ModelSize"], model_avg["Normalized_ROUGE"])

print("Per-model average (General only, LLaVA excluded):")
print(f"Spearman ρ = {rho:.4f}, p = {p_rho:.4f}")
print(f"Kendall’s τ = {tau:.4f}, p = {p_tau:.4f}")
# %% Model Size vs. Sensitivity (Normalized ROUGE)

agg = (
    filtered_general
    .groupby(['Task', 'Model'], sort=False)
    .agg(mean_performance=('ROUGE', 'mean'),
         std=('ROUGE', 'std'),
         count=('ROUGE', 'size'))
    .reset_index()
)

agg['cv'] = agg['std'] / agg['mean_performance']
agg.loc[agg['mean_performance'] <= 0, 'cv'] = np.nan
agg.loc[agg['count'] < 2, 'cv'] = np.nan
agg.replace([np.inf, -np.inf], np.nan, inplace=True)

cv_per_model = (
    agg.groupby('Model', as_index=False)['cv']
       .mean()
       .rename(columns={'cv': 'avg_cv'})
)

size_per_model = (
    filtered_general.groupby('Model', as_index=False)['ModelSize']
                    .mean()
)

cv_model_df = (
    cv_per_model.merge(size_per_model, on='Model', how='inner')
                .dropna(subset=['ModelSize', 'avg_cv'])
)

rho_cv, p_rho_cv = spearmanr(cv_model_df['ModelSize'], cv_model_df['avg_cv'])
tau_cv, p_tau_cv = kendalltau(cv_model_df['ModelSize'], cv_model_df['avg_cv'])

print("\nModel size vs average CV across strategies (General only, LLaVA excluded):")
print(f"Spearman ρ = {rho_cv:.4f}, p = {p_rho_cv:.4f}")
print(f"Kendall’s τ = {tau_cv:.4f}, p = {p_tau_cv:.4f}")
 
# %% General vs. Medical SENSITIVITY (CV) Levene's test

agg_all = (
    long_df
    .groupby(['Task', 'Model'], sort=False)
    .agg(mean_perf=('ROUGE', 'mean'),
         sd=('ROUGE', 'std'),
         n=('ROUGE', 'size'))
    .reset_index()
)

agg_all['cv'] = agg_all['sd'] / agg_all['mean_perf']
agg_all.loc[(agg_all['mean_perf'] <= 0) | (agg_all['n'] < 2), 'cv'] = np.nan

agg_all['ModelType'] = agg_all['Model'].apply(
    lambda x: 'Medical' if is_medical_model(x) else 'General'
)

clean = agg_all.dropna(subset=['cv'])         

group_sizes = clean['ModelType'].value_counts()
print("CV samples per group:")
print(group_sizes)

general_cv = clean.loc[clean['ModelType']=='General', 'cv'].values
medical_cv = clean.loc[clean['ModelType']=='Medical', 'cv'].values

if len(general_cv) < 2 or len(medical_cv) < 2:
    raise ValueError("Need at least two CV values in EACH group for Levene’s test.")

F_mean,  p_mean = levene(general_cv, medical_cv, center='mean')
F_median, p_median = levene(general_cv, medical_cv, center='median')  # Brown–Forsythe

df1, df2 = 1, len(general_cv) + len(medical_cv) - 2
print(f"\nLevene (mean):      F({df1}, {df2}) = {F_mean:.4f},  p = {p_mean:.4f}")
print(f"Brown–Forsythe:     F({df1}, {df2}) = {F_median:.4f}, p = {p_median:.4f}")
vr = max(np.var(general_cv, ddof=1), np.var(medical_cv, ddof=1)) / \
     min(np.var(general_cv, ddof=1), np.var(medical_cv, ddof=1))
print(f"Variance ratio = {vr:.2f}")

# partial eta-squared for Brown–Forsythe
eta2 = F_median / (F_median + df2 + 1)     # df1 = 1
print(f"partial η² = {eta2:.3f}")

# %% Correlations DURATION

model_perf_dur = (
    long_df.groupby("Model")[["Normalized_ROUGE", "sec_per_tok"]]
    .mean()
    .reset_index()
)

model_perf_dur["ModelType"] = model_perf_dur["Model"].apply(
    lambda x: "Medical" if is_medical_model(x) else "General"
)
model_perf_dur["ModelSize"] = model_perf_dur["Model"].apply(extract_size)

model_perf_dur.dropna(subset=["Normalized_ROUGE", "sec_per_tok", "ModelSize"], inplace=True)

# Model Size vs Duration
rho_dur, p_dur = spearmanr(model_perf_dur["ModelSize"], model_perf_dur["sec_per_tok"])
tau_dur, p_tau_dur = kendalltau(model_perf_dur["ModelSize"], model_perf_dur["sec_per_tok"])

print("\nCorrelation: Model Size vs Duration (sec/tok)")
print(f"Spearman ρ = {rho_dur:.4f}, p = {p_dur:.4f}")
print(f"Kendall’s τ = {tau_dur:.4f}, p = {p_tau_dur:.4f}")
