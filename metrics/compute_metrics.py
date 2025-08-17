def plot_timeseries(hist):
    import pandas as pd
    if hist.empty:
        plt.figure(figsize=(9,4))
        plt.title("Safety Timeseries (no data)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "safety_timeseries.png"), dpi=180)
        plt.close()
        return

    last = hist.tail(60).copy()  # show only the most recent 60 days
    fig, ax = plt.subplots(figsize=(10,4))
    for col, label in [("safe_rate","safe"), ("refusal_rate","refusal"), ("unsafe_rate","unsafe")]:
        ax.plot(last["date"], last[col], label=label, marker="o", linewidth=2)

    ax.set_title("Safety Rates Over Time")
    ax.set_xlabel("date"); ax.set_ylabel("rate")
    ax.set_ylim(0,1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(
        last["date"].min() - pd.Timedelta(days=1),
        last["date"].max() + pd.Timedelta(days=1),
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "safety_timeseries.png"), dpi=180)
    plt.close()


def plot_drift(hist):
    import pandas as pd
    if "drift_index" not in hist.columns or hist.empty:
        plt.figure(figsize=(9,4))
        plt.title("Drift Index (no data)")
        plt.xlabel("date"); plt.ylabel("JSD (0–1)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "drift_index.png"), dpi=180)
        plt.close()
        return

    last = hist.tail(60).copy()  # show only the most recent 60 days
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(last["date"], last["drift_index"], marker="o", linewidth=2)
    ax.set_title("Drift Index (JSD vs last 7 days)")
    ax.set_xlabel("date"); ax.set_ylabel("JSD (0–1)")
    ax.set_ylim(0,1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(
        last["date"].min() - pd.Timedelta(days=1),
        last["date"].max() + pd.Timedelta(days=1),
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "drift_index.png"), dpi=180)
    plt.close()
