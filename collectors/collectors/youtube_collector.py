# collectors/youtube_collector.py
import os, time, csv
from datetime import datetime
from googleapiclient.discovery import build
from collectors.templates import categorize, expected_behavior_for_category, build_prompt_from_comment

def _yt_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY env var not set")
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)

def search_video_ids(query: str, max_results=5):
    yt = _yt_client()
    res = yt.search().list(
        part="id", q=query, type="video", maxResults=max_results, relevanceLanguage="en"
    ).execute()
    ids = [item["id"]["videoId"] for item in res.get("items", [])]
    return ids

def trending_video_ids(region_code="IN", max_results=5):
    yt = _yt_client()
    res = yt.videos().list(
        part="id",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=max_results
    ).execute()
    return [item["id"] for item in res.get("items", [])]

def fetch_comments(video_id: str, max_total=50):
    yt = _yt_client()
    comments = []
    page_token = None
    while len(comments) < max_total:
        res = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=min(100, max_total - len(comments)),
            pageToken=page_token
        ).execute()
        for item in res.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(top.get("textDisplay",""))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.3)  # be polite
    return comments

def collect_youtube(query=None, region="IN", per_video_comments=30, max_videos=5, out_csv=None):
    if not out_csv:
        out_csv = f"data/live_prompts_{datetime.utcnow().strftime('%Y%m%d')}.csv"

    if query:
        vids = search_video_ids(query, max_results=max_videos)
    else:
        vids = trending_video_ids(region_code=region, max_results=max_videos)

    rows = []
    for vid in vids:
        for c in fetch_comments(vid, max_total=per_video_comments):
            cat = categorize(c)
            expected = expected_behavior_for_category(cat)
            prompt = build_prompt_from_comment(c)
            rows.append({
                "prompt": prompt,
                "category": cat,
                "expected_behavior": expected,
                "source": "youtube",
                "url": f"https://www.youtube.com/watch?v={vid}",
            })

    # dedupe short/empty
    clean = []
    seen = set()
    for r in rows:
        p = r["prompt"].strip()
        if len(p) < 30 or p in seen:
            continue
        seen.add(p)
        clean.append(r)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt","category","expected_behavior","source","url"])
        w.writeheader()
        w.writerows(clean)
    return out_csv
