"""Operation 500K — Standing Measurement Loop (plan section 4.1).

Pulls per-video retention + traffic-source data from the YouTube Analytics
API (yt-analytics.readonly scope — already present in token.pickle) and
public stats from the Data API, writes an append-only JSONL log, and prints
a ranked table.

Usage:
    python tools/retention_report.py                 # last 30 days of videos
    python tools/retention_report.py --days 90       # wider window
    python tools/retention_report.py --all           # every upload (baseline)

DECISION RULES (from the Operation 500K plan):
  • AVD% >= 75 AND views > 500 @48h  -> queue 2 same-character sequels
    (boost that arc's weight immediately).
  • AVD% < 40  -> frame-forensic that video; tag the failure mode
    (hook? mid-drop? visual?) before the next render ships.
  • Any experiment (loop echo, end band, SFX) that drops channel-median
    AVD% by >10 points across 3 videos -> revert the experiment.
  • Weekly: subs delta + returning-viewer proxy (traffic source =
    channel-page %) -> report.
"""
import argparse
import json
import os
import pickle
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from googleapiclient.discovery import build  # noqa: E402

TOKEN = os.path.join(os.path.dirname(__file__), "..", "token.pickle")
LOG   = os.path.join(os.path.dirname(__file__), "..", "analytics", "retention_log.jsonl")


def _services():
    with open(TOKEN, "rb") as f:
        creds = pickle.load(f)
    data = build("youtube", "v3", credentials=creds)
    analytics = build("youtubeAnalytics", "v2", credentials=creds)
    return data, analytics


def _all_uploads(data):
    ch = data.channels().list(part="contentDetails,statistics", mine=True).execute()
    item = ch["items"][0]
    pl = item["contentDetails"]["relatedPlaylists"]["uploads"]
    subs = item["statistics"].get("subscriberCount", "?")
    vids, tok = [], None
    while True:
        r = data.playlistItems().list(part="contentDetails,snippet",
                                      playlistId=pl, maxResults=50,
                                      pageToken=tok).execute()
        for it in r["items"]:
            vids.append({
                "id": it["contentDetails"]["videoId"],
                "publishedAt": it["contentDetails"].get(
                    "videoPublishedAt", it["snippet"]["publishedAt"]),
                "title": it["snippet"]["title"],
            })
        tok = r.get("nextPageToken")
        if not tok:
            break
    ids = [v["id"] for v in vids]
    stats = {}
    for i in range(0, len(ids), 50):
        for it in data.videos().list(part="statistics,status",
                                     id=",".join(ids[i:i + 50])).execute()["items"]:
            stats[it["id"]] = it
    for v in vids:
        s = stats.get(v["id"], {})
        v["views"]    = int(s.get("statistics", {}).get("viewCount", 0))
        v["likes"]    = int(s.get("statistics", {}).get("likeCount", 0))
        v["comments"] = int(s.get("statistics", {}).get("commentCount", 0))
        v["privacy"]  = s.get("status", {}).get("privacyStatus", "?")
    return subs, sorted(vids, key=lambda x: x["publishedAt"], reverse=True)


def _video_analytics(analytics, video_id: str, start: str, end: str) -> dict:
    """Per-video retention metrics. Returns {} on API failure (non-fatal)."""
    try:
        r = analytics.reports().query(
            ids="channel==MINE",
            startDate=start, endDate=end,
            metrics=("views,estimatedMinutesWatched,averageViewDuration,"
                     "averageViewPercentage,likes,subscribersGained"),
            filters=f"video=={video_id}",
        ).execute()
        rows = r.get("rows") or []
        if not rows:
            return {}
        cols = [h["name"] for h in r["columnHeaders"]]
        return dict(zip(cols, rows[0]))
    except Exception as e:
        print(f"  [analytics] {video_id}: {str(e)[:90]}", file=sys.stderr)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--all", action="store_true",
                    help="every upload ever (baseline mode)")
    args = ap.parse_args()

    data, analytics = _services()
    subs, vids = _all_uploads(data)

    today = date.today().isoformat()
    if args.all:
        start = "2026-01-01"
        targets = vids
    else:
        cutoff = (date.today() - timedelta(days=args.days)).isoformat()
        start = cutoff
        targets = [v for v in vids if v["publishedAt"][:10] >= cutoff]

    print(f"CHANNEL subs={subs}  |  analyzing {len(targets)} videos "
          f"(window {start} → {today})\n")

    rows = []
    for v in targets:
        a = _video_analytics(analytics, v["id"], start, today)
        rows.append({**v, **{
            "avd_s":   round(float(a.get("averageViewDuration", 0)), 1),
            "avd_pct": round(float(a.get("averageViewPercentage", 0)), 1),
            "watch_min": round(float(a.get("estimatedMinutesWatched", 0)), 1),
            "subs_gained": int(a.get("subscribersGained", 0)),
            "report_date": today,
        }})

    # Append to JSONL log
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Ranked table (by AVD% — the tier-escalation metric)
    rows.sort(key=lambda x: x["avd_pct"], reverse=True)
    print(f"{'AVD%':>5} {'AVDs':>5} {'VIEWS':>6} {'+SUB':>4} {'PRIV':<9} {'DATE':<11} TITLE")
    print("-" * 100)
    for r in rows:
        flag = ""
        if r["avd_pct"] >= 75 and r["views"] > 500:
            flag = "  << SCALE: queue 2 same-character sequels"
        elif 0 < r["avd_pct"] < 40:
            flag = "  << FORENSIC: tag failure mode"
        print(f"{r['avd_pct']:>5} {r['avd_s']:>5} {r['views']:>6} "
              f"{r['subs_gained']:>4} {r['privacy']:<9} "
              f"{r['publishedAt'][:10]:<11} {r['title'][:48]}{flag}")

    public = [r for r in rows if r["privacy"] == "public" and r["avd_pct"] > 0]
    if public:
        med = sorted(r["avd_pct"] for r in public)[len(public) // 2]
        print(f"\nChannel median AVD% (public, analytics-visible): {med}")
        print(f"Logged {len(rows)} rows → {os.path.relpath(LOG)}")


if __name__ == "__main__":
    main()
