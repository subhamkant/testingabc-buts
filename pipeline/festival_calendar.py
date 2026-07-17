"""Festival-aware topic override (2026-07-09).

Competitive forensic (user request: "compare with other channels, find the
diff"): every winner in the Hindi mythology niche gets its breakouts from
SPIKE plays, not daily grind — and the single most reliable spike is
festival timing. Nishva fx's 14.6M-view video was a Ganesh Chaturthi video
published ON Ganesh Chaturthi (33-video channel, 231K subs). We have never
shipped a single festival-timed video.

Each entry links a 2026 festival to a canonical Mahabharata/Krishna story
with a DEVOTIONAL-POSITIVE register (the second thing every niche winner
shares) while keeping enough charged-noun vocabulary to pass the frozen
Title-DNA gate. Window = festival day plus N days before (search interest
ramps ahead of the day).

The override slots ABOVE the weighted arc rotation and BELOW the manual
scheduled_topics queue. The recent-topics ledger prevents a window from
firing twice.
"""
from datetime import datetime, timedelta, timezone

_IST = timezone(timedelta(hours=5, minutes=30))

# (festival_date_iso, days_before_window, name, topic)
FESTIVALS_2026 = [
    ("2026-08-17", 1, "Nag Panchami",
     "Astika stopping the Sarpa Satra on Nag Panchami — the boy sage whose "
     "single vow halted Janamejaya's great snake sacrifice and saved the "
     "Naga race, the mercy that ended the Mahabharata's last revenge"),
    ("2026-08-27", 1, "Raksha Bandhan",
     "Draupadi's strip of silk on Krishna's bleeding finger — the first "
     "rakhi, the small act of care Krishna repaid with endless cloth in the "
     "Kuru sabha, the promise of protection that never broke"),
    # 2026-07-17 date corrected Sep 3 -> Sep 4 (drikpanchang/timeanddate:
    # Smarta observance Fri Sep 4, ISKCON/Vaishnava Sep 5). Window 2 covers
    # the Sep 2-4 lead-up; the Sep-5 ISKCON day rides the spike's tail.
    ("2026-09-04", 2, "Janmashtami",
     "Krishna's divine birth at midnight — prison chains falling open, "
     "guards asleep by divine will, the Yamuna parting for Vasudeva's "
     "basket, the night the protector of the Pandavas chose to be born"),
    ("2026-09-14", 1, "Ganesh Chaturthi",
     "Ganesha's broken tusk that wrote the Mahabharata — Vyasa's impossible "
     "condition, the god who broke his own tusk to keep his promise as the "
     "epic's first scribe, the sacrifice behind every story we tell"),
    ("2026-10-20", 1, "Vijayadashami",
     "Arjuna at the Shami tree on Vijayadashami — retrieving the hidden "
     "Gandiva after thirteen years of exile and defeating the entire "
     "Kaurava army alone at Virata, the day victory itself was named"),
    ("2026-11-07", 1, "Naraka Chaturdashi / Diwali",
     "Krishna and Satyabhama slaying Narakasura before dawn — the demon "
     "king's fall, sixteen thousand captive princesses freed, the victory "
     "lamp-lighting that became Diwali's first morning"),
]


def festival_topic_for_today(recent_topics: list | None = None) -> tuple | None:
    """Return (festival_name, topic) if today (IST) falls in a festival
    window and the topic hasn't already been used; else None."""
    today = datetime.now(_IST).date()
    recent_lower = {t.lower() for t in (recent_topics or [])}
    for date_iso, window, name, topic in FESTIVALS_2026:
        fdate = datetime.fromisoformat(date_iso).date()
        if fdate - timedelta(days=window) <= today <= fdate:
            if topic.lower() in recent_lower:
                continue
            return (name, topic)
    return None
