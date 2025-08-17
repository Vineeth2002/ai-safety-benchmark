import os, json, math, datetime as dt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_MAIN = "results/tiny_live_outputs.csv"       # primary daily eval CSV
RESULTS_ALT  = "results/tiny_snapshot_outputs.csv"   # fallback if primary missing
HISTORY_CSV  = "metrics/history_metrics.csv"
METRICS_JSON = "results/metrics.json"
TS_PNG       = "results/safety_timeseries.png"
DRIFT_PNG    = "results/drift_index.png"

def js_divergence(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        a = np.clip(a, 1e-12, 1.0); b = np.clip(b, 1e-12, 1.0)
        return np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def load_latest_df():
    path = RESULTS_MAIN if os.path.exists(RESULTS_MAIN) else RESULTS_ALT
    if not os.path.exists(path):
        raise FileNotFoundError("No results CSV found.")
    return pd.read_csv(path)

def distribution_from_df(df):
    counts = df["classification"].value_counts()
    safe = int(counts.get("safe", 0))
    refusal = int(counts.get("refusal", 0))
    unsafe = int(counts.get("unsafe", 0))
    total = max(safe + refusal + unsafe, 1)
    return {
        "safe": safe, "refusal": refusal, "unsafe": unsafe, "total": total,
        "safe_rate": safe/total, "refusal_rate": refusal/total, "unsafe_rate": unsafe/total,
        "s2s": unsafe / (unsafe + refusal + 1e-6)
    }

def main():
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    today = dt.datetime.utcnow().date().isoformat()
    df = load_latest_df()
    dist = distribution_from_df(df)

    # append/refresh today's row in history
    row = {
        "date": today,
        "safe": dist["safe"], "refusal": dist["refusal"], "unsafe": dist["unsafe"],
        "safe_rate": dist["safe_rate"], "refusal_rate": dist["refusal_rate"],
        "unsafe_rate": dist["unsafe_rate"], "s2s": dist["s2s"]
    }
    if os.path.exists(HISTORY_CSV):
        hist = pd.read_csv(HISTORY_CSV)
        hist = hist[hist["date"] != today]
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(HISTORY_CSV, index=False)

    # Drift Index vs last 7 days (exclude today)
    prev = hist[hist["date"] < today].tail(7)
    if len(prev) >= 1:
        p = [prev["safe_rate"].mean(), prev["refusal_rate"].mean(), prev["unsafe_rate"].mean()]
        q = [dist["safe_rate"], dist["refusal_rate"], dist["unsafe_rate"]]
        drift = float(js_divergence(p, q))
    else:
        drift = 0.0

    # metrics json
    out = {
        "date": today,
        "safe_rate": dist["safe_rate"],
        "refusal_rate": dist["refusal_rate"],
        "unsafe_rate": dist["unsafe_rate"],
        "s2s": dist["s2s"],
        "drift_index_jsd": drift
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # timeseries chart
    hist_plot = hist.sort_values("date")
    plt.figure(figsize=(8,4.2))
    plt.plot(hist_plot["date"], hist_plot["safe_rate"], label="safe")
    plt.plot(hist_plot["date"], hist_plot["refusal_rate"], label="refusal")
    plt.plot(hist_plot["date"], hist_plot["unsafe_rate"], label="unsafe")
    plt.xticks(rotation=45, ha="right"); plt.ylabel("rate"); plt.title("Daily Safety Timeseries")
    plt.legend(); plt.tight_layout(); plt.savefig(TS_PNG); plt.close()

    # drift chart
    plt.figure(figsize=(4.5,3.2))
    plt.bar(["Drift (JSD)"], [drift])
    plt.ylim(0, 0.7)  # JSD ∈ [0, ln2] ≈ 0.693
    plt.title("Drift Index vs 7-day average")
    plt.tight_layout(); plt.savefig(DRIFT_PNG); plt.close()

    print("Saved:", METRICS_JSON, TS_PNG, DRIFT_PNG)

if __name__ == "__main__":
    main()
