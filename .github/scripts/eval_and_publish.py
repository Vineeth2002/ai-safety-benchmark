import os, json, glob, pandas as pd, matplotlib.pyplot as plt
from benchmark.evaluate import classify_response, evaluate_results

def summarize(csv_path):
    df = pd.read_csv(csv_path)
    if "classification" not in df.columns:
        df["classification"] = df["response"].apply(classify_response)
        df.to_csv(csv_path, index=False)
    # ensure toxicity columns exist
    df = evaluate_results(csv_path)

    counts = df["classification"].value_counts().to_dict()
    tox_counts = df["tox_label"].value_counts(dropna=False).to_dict()
    mean_tox = float(df["toxicity"].dropna().mean()) if "toxicity" in df else None
    return {"counts": counts, "toxicity": {"counts": tox_counts, "mean": mean_tox}}

def chart_from_counts(counts, title, out_png):
    s = pd.Series(counts)
    ax = s.plot(kind="bar", title=title)
    ax.set_xlabel("class"); ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def main():
    os.makedirs("results", exist_ok=True)
    outputs = sorted(glob.glob("results/*_outputs.csv"))
    summary_all = {}
    for csv in outputs:
        model = os.path.basename(csv).replace("_outputs.csv","")
        s = summarize(csv)
        summary_all[model] = s
        chart_from_counts(s["counts"], f"Safety Summary ({model})", f"results/{model}_summary.png")

    # comparison table & json
    models = sorted(summary_all.keys())
    rows=[]
    for m in models:
        c = summary_all[m]["counts"]
        rows.append({"model": m,
                     "safe": int(c.get("safe",0)),
                     "refusal": int(c.get("refusal",0)),
                     "unsafe": int(c.get("unsafe",0))})
    pd.DataFrame(rows).to_csv("results/summary_comparison.csv", index=False)
    with open("results/summary_all.json","w") as f:
        json.dump(summary_all, f, indent=2)
    print("Wrote results: summary_comparison.csv + summary_all.json + per-model charts")

if __name__ == "__main__":
    main()
