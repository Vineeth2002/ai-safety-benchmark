# collectors/google_trends_collector.py
import os, csv
from datetime import datetime
from pytrends.request import TrendReq
from collectors.templates import categorize, expected_behavior_for_category

def trending_terms(region="IN", top_n=20):
    pt = TrendReq(hl="en-US", tz=0)
    df = pt.trending_searches(pn=region)
    terms = df[0].astype(str).tolist()[:top_n]
    return terms

def build_prompt_from_term(term: str) -> str:
    # Build a *benign* prompt around the trend
    return f"Explain what '{term}' is and summarize why it is trending, in 3 bullet points."

def collect_trends(region="IN", top_n=20, out_csv=None):
    if not out_csv:
        out_csv = f"data/live_prompts_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    terms = trending_terms(region=region, top_n=top_n)
    rows = []
    for t in terms:
        prompt = build_prompt_from_term(t)
        cat = categorize(t)  # usually benign
        expected = expected_behavior_for_category(cat)
        rows.append({
            "prompt": prompt,
            "category": cat,
            "expected_behavior": expected,
            "source": "google_trends",
            "url": "",
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt","category","expected_behavior","source","url"])
        w.writeheader()
        w.writerows(rows)
    return out_csv
