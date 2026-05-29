"""
Topic-signature matching for dedup.

Two consumers:
  1. `backfill_recent_topics.py` — one-off backfill from YouTube channel titles
  2. `script_generator._pick_next_arc_topic()` — runtime pre-check before
      script generation. Pulls last N video titles from the channel, extracts
      (character, incident) signatures, and rejects any arc-topic candidate
      whose signature overlaps a published title.

Why both:
  - `recent_topics.json` is the source-of-truth for exact-string dedup
    (gets auto-committed by GHA after every successful upload)
  - The runtime check is a SAFETY NET for: (a) topics emitted under
    slightly different exact strings, (b) cases where recent_topics.json
    drifts out of sync with the channel, (c) manual-trigger paths that
    bypass the arc-walk.

The runtime check fails SAFE — any API error or missing token returns an
empty signature set, which is equivalent to "no overlap" and falls back to
the existing recent_topics.json dedup. Renders are never blocked by this
module.
"""
import os
import pickle
import re


# ── Character + incident keyword banks (English ↔ Devanagari) ───────────────
# Lowercased English variants for case-insensitive matching. Devanagari is
# left as-is (no case-folding required for Indic scripts).

CHARACTER_VARIANTS = {
    "bhishma":     ["bhishma", "bhisma", "भीष्म", "देवव्रत", "devavrata"],
    "karna":       ["karna", "कर्ण", "vasusena"],
    "arjuna":      ["arjuna", "arjun", "अर्जुन", "partha"],
    "draupadi":    ["draupadi", "द्रौपदी", "panchali", "पाँचाली"],
    "yudhishthira":["yudhishthira", "yudhishtir", "yudhisthira", "युधिष्ठिर", "dharmaraja"],
    "eklavya":     ["eklavya", "ekalavya", "एकलव्य"],
    "ashwatthama": ["ashwatthama", "ashvatthama", "अश्वत्थामा"],
    "krishna":     ["krishna", "कृष्ण", "vasudeva", "वासुदेव"],
    "kunti":       ["kunti", "कुंती"],
    "drona":       ["drona", "द्रोण"],
    "shikhandi":   ["shikhandi", "शिखंडी", "amba", "अंबा"],
    "duryodhana":  ["duryodhana", "दुर्योधन"],
    "dushasana":   ["dushasana", "duhshasana", "दुःशासन", "दुशासन"],
    "vidura":      ["vidura", "विदुर"],
    "shakuni":     ["shakuni", "शकुनि"],
    "abhimanyu":   ["abhimanyu", "अभिमन्यु"],
    "jayadratha":  ["jayadratha", "जयद्रथ"],
    "gandhari":    ["gandhari", "गांधारी"],
    "bhima":       ["bhima", "भीम"],
    "nakula":      ["nakula", "नकुल"],
    "sahadeva":    ["sahadeva", "सहदेव"],
    "parashurama": ["parashurama", "parshurama", "परशुराम"],
    "barbarika":   ["barbarika", "बर्बरीक"],
    "balarama":    ["balarama", "बलराम"],
    "aravan":      ["aravan", "अरावन"],
}

INCIDENT_KEYWORDS = {
    "vow":         ["vow", "pledge", "oath", "promise", "प्रतिज्ञा", "प्रण", "शपथ"],
    "celibacy":    ["celibacy", "celibate", "marriage", "wedding", "विवाह", "ब्रह्मचर्य"],
    "vastraharan": ["vastraharan", "disrobing", "robe", "hair", "वस्त्रहरण", "अपमान", "humiliation", "humiliated"],
    "swayamvara":  ["swayamvara", "swayamwar", "स्वयंवर", "rejection", "refused", "ठुकराया"],
    "kurukshetra": ["kurukshetra", "battlefield", "battle", "war", "कुरुक्षेत्र", "युद्ध"],
    "death":       ["death", "die", "died", "killed", "killer", "वध", "मृत्यु", "मरने", "मौत"],
    "curse":       ["curse", "cursed", "श्राप", "शाप"],
    "sacrifice":   ["sacrifice", "sacrificed", "बलिदान", "त्याग"],
    "birth":       ["birth", "born", "abandonment", "abandoned", "जन्म", "गंगा"],
    "exile":       ["exile", "forest", "वनवास", "अज्ञातवास"],
    "chakravyuha": ["chakravyuha", "chakravyuh", "चक्रव्यूह"],
    "bed_arrows":  ["bed of arrows", "arrow bed", "shaiya", "बाणशय्या", "अंतिम शिक्षा", "final teaching"],
    "gita":        ["gita", "bhagavad", "गीता", "भगवद"],
    "silence":     ["silence", "silent", "मौन", "चुप"],
    # `secret` category REMOVED 2026-05-30 (iter-4 bug fix). The keywords
    # "secret/hidden/untold/real/truth/रहस्य/सच/अनकहा/असली" are Phase 1-Plus
    # title BRAND vocabulary, not story-incident signals. Every video uses
    # one. Treating them as incident-overlap markers blocked the entire
    # Karna + Bhishma arcs by false-positive (every Karna topic with "secret"
    # in its description was flagged as duplicate of "Karna's Untold Truth").
    # The category is gone; story-specific incidents below remain.
    "thumb":       ["thumb", "gurudakshina", "अंगूठा", "गुरुदक्षिणा"],
    "wound":       ["wound", "scar", "injured", "घाव", "जख्म"],
    "betrayal":    ["betrayal", "betrayed", "धोखा", "विश्वासघात"],
    "parent_loss": ["father", "पिता", "abandoned", "abandonment", "त्यागा", "अनाथ"],
    # `parent_loss` (2026-05-30 iter-4 bug fix — was `father` with overlap
    # into character names). "kunti", "माँ", "mother" REMOVED because Kunti
    # is already in CHARACTER_VARIANTS; mixing parent-NAME with parent-ROLE
    # caused every Kunti-related topic to false-positive overlap on
    # `father` incident. The category now strictly tracks parent-child
    # SEPARATION events (abandonment, orphan-status), not the parent's name.
    "lie":         ["lie", "lied", "deception", "झूठ"],
    "kavach":      ["kavach", "kundal", "armor", "armour", "कवच", "कुंडल"],
    "massacre":    ["massacre", "slaughter", "नरसंहार", "हत्या"],
    "advice":      ["advice", "warning", "counsel", "सलाह", "चेतावनी"],
    "yaksha":      ["yaksha", "lake", "यक्ष"],
    "dice":        ["dice", "game", "gambling", "द्यूत", "जुआ"],
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def characters_in(text: str) -> set:
    """Set of canonical character names mentioned in `text`."""
    t = _norm(text)
    hits = set()
    for canonical, variants in CHARACTER_VARIANTS.items():
        for v in variants:
            if v.lower() in t:
                hits.add(canonical)
                break
    return hits


def incidents_in(text: str) -> set:
    """Set of incident keyword categories mentioned in `text`."""
    t = _norm(text)
    hits = set()
    for category, keywords in INCIDENT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in t:
                hits.add(category)
                break
    return hits


def signature_of(text: str) -> tuple:
    """Return (frozenset chars, frozenset incidents) for `text`. Hashable
    so signatures can be stored in a set."""
    return (frozenset(characters_in(text)), frozenset(incidents_in(text)))


# ── Runtime API pre-check ───────────────────────────────────────────────────

_RUNTIME_SIGNATURES_CACHE = None   # process-lifetime cache


def fetch_recent_title_signatures(
    token_path: str = "token.pickle",
    limit: int = 50,
) -> list:
    """Return a list of (chars, incidents) tuples extracted from the most
    recent `limit` published videos on the channel. Cached for the process
    lifetime.

    Fail-safe: returns an empty list on ANY error (missing token, API quota,
    network failure, JSON parse error). The caller is expected to fall back
    to the recent_topics.json dedup only — renders must never be blocked
    by this module.
    """
    global _RUNTIME_SIGNATURES_CACHE
    if _RUNTIME_SIGNATURES_CACHE is not None:
        return _RUNTIME_SIGNATURES_CACHE

    try:
        if not os.path.exists(token_path):
            print(f"    [runtime-dedup] {token_path} missing — runtime title check disabled")
            _RUNTIME_SIGNATURES_CACHE = []
            return _RUNTIME_SIGNATURES_CACHE

        # Lazy import to keep script_generator import-light
        from googleapiclient.discovery import build

        with open(token_path, "rb") as f:
            creds = pickle.load(f)
        yt = build("youtube", "v3", credentials=creds)

        ch = yt.channels().list(part="contentDetails", mine=True).execute()
        uploads_pl = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

        ids = []
        page = None
        while len(ids) < limit:
            r = yt.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_pl,
                maxResults=min(50, limit - len(ids)),
                pageToken=page,
            ).execute()
            ids += [it["contentDetails"]["videoId"] for it in r["items"]]
            page = r.get("nextPageToken")
            if not page:
                break

        ids = ids[:limit]
        if not ids:
            _RUNTIME_SIGNATURES_CACHE = []
            return _RUNTIME_SIGNATURES_CACHE

        sigs = []
        for i in range(0, len(ids), 50):
            chunk = ids[i:i + 50]
            r = yt.videos().list(part="snippet", id=",".join(chunk)).execute()
            for it in r["items"]:
                title = it["snippet"]["title"]
                sigs.append(signature_of(title))

        _RUNTIME_SIGNATURES_CACHE = sigs
        print(f"    [runtime-dedup] loaded {len(sigs)} published-title signatures from channel")
        return _RUNTIME_SIGNATURES_CACHE
    except Exception as e:
        print(f"    [runtime-dedup] disabled (error reading channel): {str(e)[:120]}")
        _RUNTIME_SIGNATURES_CACHE = []
        return _RUNTIME_SIGNATURES_CACHE


def topic_overlaps_published(topic_text: str, published_signatures: list) -> bool:
    """Return True if `topic_text`'s signature overlaps any published title's
    signature strongly enough to be a duplicate risk.

    Overlap rule: at least ONE shared character AND at least ONE shared
    incident keyword. A topic with no character (e.g., "Kurukshetra War")
    can never trigger overlap — caller should still dedup those via
    exact-string check.
    """
    tc, ti = signature_of(topic_text)
    if not tc:
        return False
    for pc, pi in published_signatures:
        if tc & pc and ti & pi:
            return True
    return False


def reset_runtime_cache() -> None:
    """Clear the process-lifetime cache. Used by tests."""
    global _RUNTIME_SIGNATURES_CACHE
    _RUNTIME_SIGNATURES_CACHE = None
