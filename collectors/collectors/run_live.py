# collectors/run_live.py
import os, sys, argparse
from collectors.youtube_collector import collect_youtube
from collectors.google_trends_collector import collect_trends

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["youtube", "trends"], default="youtube")
    p.add_argument("--query", default=None, help="YouTube search query (if omitted, use trending)")
    p.add_argument("--region", default=os.getenv("REGION","IN"))
    p.add_argument("--per_video_comments", type=int, default=30)
    p.add_argument("--max_videos", type=int, default=5)
    p.add_argument("--top_n", type=int, default=20, help="for trends")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.source == "youtube":
        path = collect_youtube(
            query=args.query,
            region=args.region,
            per_video_comments=args.per_video_comments,
            max_videos=args.max_videos,
            out_csv=args.out
        )
    else:
        path = collect_trends(region=args.region, top_n=args.top_n, out_csv=args.out)

    print(f"✅ Saved live dataset: {path}")

if __name__ == "__main__":
    main()
