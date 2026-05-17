"""
One-off: print your Instagram Business Account ID by calling Graph API
with the IG_ACCESS_TOKEN already saved in your .env. Never prints the
token itself.

Usage:
    python find_ig_account_id.py
"""

import os
import sys
import io
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

token = os.environ.get("IG_ACCESS_TOKEN", "").strip()
if not token:
    sys.exit("[ERROR] IG_ACCESS_TOKEN not set in .env")

print("Calling /me/accounts to find Facebook Pages linked to the system user...")

r = requests.get(
    "https://graph.facebook.com/v19.0/me/accounts",
    params={
        "access_token": token,
        "fields": "id,name,instagram_business_account",
    },
    timeout=20,
)

if r.status_code != 200:
    print(f"[ERROR] HTTP {r.status_code}: {r.text[:400]}")
    sys.exit(1)

pages = r.json().get("data", [])
if not pages:
    sys.exit(
        "[ERROR] No pages returned. Possible reasons:\n"
        "  • System user wasn't granted access to a Page in Business settings\n"
        "  • The token's scopes don't include pages_show_list\n"
        "  • The IG account isn't linked to any Page yet"
    )

print(f"\nFound {len(pages)} Page(s) accessible to this token:\n")
candidates = []
for p in pages:
    name = p.get("name", "?")
    pid = p.get("id", "?")
    igba = p.get("instagram_business_account")
    igba_id = igba.get("id") if igba else None
    marker = f"IG: {igba_id}" if igba_id else "no IG linked"
    print(f"  • {name}  (page_id={pid}, {marker})")
    if igba_id:
        candidates.append((name, igba_id))

if not candidates:
    sys.exit(
        "\n[ERROR] None of the Pages have a linked Instagram Business account.\n"
        "Make sure your IG account (@vyasa.ai.stories) is a Business/Creator\n"
        "account AND is linked to one of the Pages above (IG app → Settings\n"
        "→ Linked accounts → Facebook)."
    )

name, ig_id = candidates[0]
print(f"\n{'=' * 60}")
print(f"  IG_BUSINESS_ACCOUNT_ID = {ig_id}")
print(f"{'=' * 60}")
print(f"\n  (linked via Page '{name}')")
print(f"\nAdd to your .env:")
print(f"  IG_BUSINESS_ACCOUNT_ID={ig_id}")
print(f"\nAdd to GitHub Secrets:")
print(f"  Name:  IG_BUSINESS_ACCOUNT_ID")
print(f"  Value: {ig_id}")
print()

# Sanity ping: confirm we can read the IG account's metadata with this token
r2 = requests.get(
    f"https://graph.facebook.com/v19.0/{ig_id}",
    params={"access_token": token, "fields": "username,media_count"},
    timeout=15,
)
if r2.status_code == 200:
    body = r2.json()
    print(f"[OK] Connected to @{body.get('username')} "
          f"({body.get('media_count', '?')} existing media)")
else:
    print(f"[!] WARN: couldn't read account metadata "
          f"(HTTP {r2.status_code}): {r2.text[:200]}")
    print("    Token may still work for publishing — try a test upload.")
