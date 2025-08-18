import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import re

bar_width       = 0.22            
mean_color      = 'rosybrown'      
robust_color    = 'lightsteelblue'   
max_tasks       = 5               
mean_ylim_pad   = 0.18               # +5% headroom above max mean ROUGE
robust_ylim_pad = 0.18               # +5% headroom above max sensitivity
mean_better   = 'up'   
duration_better = 'down'
sensitive_better = 'down'

def clean_model_label(name: str) -> str:
    label = re.sub(r'(_)?ROUGE.*$', '', str(name), flags=re.IGNORECASE)
    return label.rstrip('_')

# 1) Load all per-strategy ROUGE scores (and remember CSV model order)
data_dir = "./" 
file_pattern = os.path.join(data_dir, "results_*.csv")

long_data = []
task_model_order = {}

for filepath in glob.glob(file_pattern):
    task = os.path.basename(filepath).replace("results_", "").replace(".csv", "")
    df = pd.read_csv(filepath)

    if "strategy" not in df.columns:
        continue

    df = df.set_index("strategy")
    rouge_cols = [col for col in df.columns if "ROUGE" in col.upper()]
    task_model_order[task] = rouge_cols[:]  

    for model_col in rouge_cols:
        for strategy, score in df[model_col].items():
            long_data.append({
                "Task": task,
                "Model": model_col,
                "Strategy": strategy,
                "ROUGE": score
            })

long_df = pd.DataFrame(long_data)

def arrow(dir_): return '↑' if dir_ == 'up' else '↓'

# 2) Aggregate mean performance and sensitivity (CV) per Task × Model
if long_df.empty:
    print("No data loaded. Ensure files matching 'results_*.csv' exist,",
          "with a 'strategy' column and ROUGE* columns.")
else:
    long_duration_data = []
    for filepath in glob.glob(file_pattern):
        task = os.path.basename(filepath).replace("results_", "").replace(".csv", "")
        df = pd.read_csv(filepath)

        if "strategy" not in df.columns:
            continue

        df = df.set_index("strategy")
        sec_per_tok_cols = [col for col in df.columns if "sec/tok" in col.lower()]

        for model_col in sec_per_tok_cols:
            for strategy, duration in df[model_col].items():
                long_duration_data.append({
                    "Task": task,
                    "Model": model_col.replace("_sec/tok", "_ROUGE"),  # align naming with ROUGE
                    "Strategy": strategy,
                    "sec_per_tok": duration
                })

    duration_df = pd.DataFrame(long_duration_data)

    combined_df = pd.merge(long_df, duration_df, on=["Task", "Model", "Strategy"], how="left")

    agg = (combined_df
           .groupby(['Task', 'Model'], sort=False)
           .agg(mean_performance=('ROUGE', 'mean'),
                std=('ROUGE', 'std'),
                count=('ROUGE', 'size'),
                mean_duration=('sec_per_tok', 'mean'))
           .reset_index())

    agg['cv'] = agg['std'] / agg['mean_performance']
    agg.loc[agg['mean_performance'] == 0, 'cv'] = np.nan
    agg.loc[~np.isfinite(agg['cv']), 'cv'] = np.nan

    # 3) Separate plots (one per task), preserving CSV model order and cleaning labels
    tasks = [t for t in agg['Task'].unique() if t in task_model_order][:max_tasks]
    if len(tasks) == 0:
        print("No tasks found after aggregation.")
    else:
        for task in tasks:
            order = [m for m in task_model_order[task]
                     if m in agg.loc[agg['Task'] == task, 'Model'].unique()]
            df_task = (agg[agg['Task'] == task]
                       .set_index('Model')
                       .reindex(order))

            models = df_task.index.tolist()
            labels = [clean_model_label(m) for m in models]  # cleaned x labels
            x = np.arange(len(models))
            mean_vals = df_task['mean_performance'].to_numpy()
            robust_vals = df_task['cv'].to_numpy()
            duration_vals = df_task['mean_duration'].to_numpy()  # 🔄 new

            fig, ax = plt.subplots(figsize=(8, 6))

            # Bar 1: Mean ROUGE (left y-axis)
            ax.bar(x - bar_width, mean_vals, bar_width,
                label=f'Performance (ROUGE) {arrow(mean_better)}',
                color=mean_color)
            ax.set_xlabel('Model', fontsize=15)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=15)

            if np.isfinite(mean_vals).any():
                y_max = np.nanmax(mean_vals)
                ax.set_ylim(0, y_max * (1 + mean_ylim_pad) if y_max > 0 else 1)

            # === Shared right y-axis for CV and Duration ===
            ax2 = ax.twinx()

            # Bar 2: Sensitivity (CV)
            ax2.bar(x, robust_vals, bar_width,
                    label=f'Sensitivity (CV) {arrow(sensitive_better)}',
                    color=robust_color)

            # Bar 3: Duration (tokens/s)
            duration_color = 'darkgrey'
            ax2.bar(x + bar_width, duration_vals, bar_width,
                    label=f'Inference Time (s/token) {arrow(duration_better)}', color=duration_color)

            # Set same Y-limit for both CV and Duration
            combined = np.concatenate([robust_vals, duration_vals])
            if np.isfinite(combined).any():
                ymax = np.nanmax(combined)
                ax2.set_ylim(0, ymax * (1 + robust_ylim_pad) if ymax > 0 else 1)

            ax.set_ylabel('ROUGE', fontsize=14)
            ax2.set_ylabel('Sensitivity (CV) and Time (s/token)', fontsize=14)

            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)
            ax2.tick_params(axis='y', labelsize=13)
            
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=13)

            # plt.title(f'Task: {task}', fontsize=16)
            plt.tight_layout()
            plt.show()