import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)

# Paths
history_path = 'metrics/history_metrics.csv'
drift_path = 'results/drift_index.png'
timeseries_path = 'results/safety_timeseries.png'
weekly_path = 'results/weekly_report.png'
monthly_path = 'results/monthly_report.png'
metrics_json_path = 'results/metrics.json'

# Load/create data
if os.path.exists(history_path):
    df = pd.read_csv(history_path)
else:
    df = pd.DataFrame(columns=["date", "score", "class"])

# 1. Safety Timeseries Plot
plt.figure(figsize=(6, 3))
if not df.empty:
    plt.plot(pd.to_datetime(df['date']), df['score'], marker='o')
    plt.title("Daily Safety Score")
    plt.xlabel("Date")
    plt.ylabel("Safety Score")
else:
    plt.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=16)
    plt.title("Daily Safety Score")
plt.tight_layout()
plt.savefig(timeseries_path)
plt.close()

# 2. Drift Index Plot
plt.figure(figsize=(6, 3))
if len(df) >= 7:
    rolling = df['score'].rolling(7).mean()
    drift = np.abs(df['score'] - rolling)
    plt.plot(pd.to_datetime(df['date']), drift, marker='o', color='orange')
    plt.title("Drift Index (|Current - 7d avg|)")
    plt.xlabel("Date")
    plt.ylabel("Drift")
else:
    plt.text(0.5, 0.5, "Not enough data for drift", ha='center', va='center', fontsize=14)
    plt.title("Drift Index")
plt.tight_layout()
plt.savefig(drift_path)
plt.close()

# 3. Weekly & Monthly Plots (placeholder)
def plot_class_dist(timeframe, out_path):
    plt.figure(figsize=(6,3))
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        cutoff = df['date'].max() - pd.Timedelta(days=6 if timeframe=="weekly" else 29)
        recent = df[df['date'] >= cutoff]
        if not recent.empty:
            recent['class'].value_counts().plot(kind='bar')
            plt.title(f"{timeframe.capitalize()} Class Distribution")
            plt.xlabel("Class")
            plt.ylabel("Count")
        else:
            plt.text(0.5, 0.5, "No Recent Data", ha='center', va='center')
    else:
        plt.text(0.5, 0.5, "No Data", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

plot_class_dist("weekly", weekly_path)
plot_class_dist("monthly", monthly_path)

# 4. Metrics JSON (latest row or empty)
metrics = df.iloc[-1].to_dict() if not df.empty else {}
import json
with open(metrics_json_path, "w") as f:
    json.dump(metrics, f, indent=2)
