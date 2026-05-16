"""
🔑  One-Time Instagram Reels OAuth Setup
=========================================
Run this ONCE on your local machine before deploying to GitHub Actions.

Prerequisites (do these on Meta side first):
  1. Convert your Instagram account to BUSINESS or CREATOR type
     (IG app → Settings → Account → "Switch to professional")
  2. Link the IG account to a Facebook Page
     (Settings → Linked accounts → Facebook)
  3. Create a Meta for Developers app at https://developers.facebook.com
     • App type: Business
     • Add product: "Instagram" (Instagram API with Instagram Login)
     • Permissions: instagram_business_basic, instagram_business_content_publish
  4. Generate a short-lived USER access token via Graph API Explorer
     (https://developers.facebook.com/tools/explorer/) selecting your app
     and the 2 permissions above.

What this script does:
  • Exchanges your short-lived token for a long-lived (~60 day) token
  • Finds your IG Business Account ID via your Facebook Page
  • Prints both values for you to save as GitHub Secrets:
      IG_ACCESS_TOKEN          ← long-lived token
      IG_BUSINESS_ACCOUNT_ID   ← numeric ID

Run:
  python setup_instagram_auth.py

It will prompt you for:
  • Your Facebook App ID
  • Your Facebook App Secret
  • The short-lived USER access token from Graph API Explorer
"""

import sys
import io
import json
from getpass import getpass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

_GRAPH = "https://graph.facebook.com/v19.0"


def _exchange_for_long_lived(app_id: str, app_secret: str, short_token: str) -> str:
    """POST /oauth/access_token grant_type=fb_exchange_token."""
    print("\n[1/3] Exchanging short-lived → long-lived (~60 day) token...")
    r = requests.get(
        f"{_GRAPH}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "fb_exchange_token": short_token,
        },
        timeout=20,
    )
    if r.status_code != 200:
        sys.exit(f"[ERROR] exchange failed: {r.status_code} {r.text[:300]}")
    long_token = r.json().get("access_token", "")
    if not long_token:
        sys.exit(f"[ERROR] no access_token in response: {r.text[:200]}")
    print("    [OK] long-lived token obtained (~60 days valid)")
    return long_token


def _find_ig_business_account_id(long_token: str) -> str:
    """GET /me/accounts → find first page with an instagram_business_account."""
    print("\n[2/3] Finding your Facebook Page + linked Instagram Business Account...")
    r = requests.get(
        f"{_GRAPH}/me/accounts",
        params={
            "access_token": long_token,
            "fields": "id,name,instagram_business_account",
        },
        timeout=20,
    )
    if r.status_code != 200:
        sys.exit(f"[ERROR] /me/accounts failed: {r.status_code} {r.text[:300]}")
    pages = r.json().get("data", [])
    if not pages:
        sys.exit("[ERROR] no Facebook Pages found on this account. "
                 "Create or link a Page first.")

    print(f"    Found {len(pages)} Page(s):")
    candidates = []
    for p in pages:
        igba = p.get("instagram_business_account")
        igba_id = igba.get("id") if igba else None
        marker = f"IG: {igba_id}" if igba_id else "no linked IG"
        print(f"      • {p.get('name')} (page_id={p['id']}, {marker})")
        if igba_id:
            candidates.append((p, igba_id))

    if not candidates:
        sys.exit("[ERROR] none of your Pages have a linked Instagram Business "
                 "account. Link your IG (Business/Creator) to a Page first.")

    if len(candidates) > 1:
        print(f"\n    Multiple linked IG accounts found. Using the first: "
              f"{candidates[0][0]['name']}")
    page, ig_business_id = candidates[0]
    print(f"    [OK] IG Business Account ID = {ig_business_id} "
          f"(linked via Page '{page['name']}')")
    return ig_business_id


def _verify_publish_permission(long_token: str, ig_business_id: str) -> None:
    """Sanity-check the token can hit the IG account endpoints we'll need."""
    print("\n[3/3] Verifying publish permission...")
    r = requests.get(
        f"{_GRAPH}/{ig_business_id}",
        params={"access_token": long_token, "fields": "username,media_count"},
        timeout=20,
    )
    if r.status_code == 200:
        body = r.json()
        print(f"    [OK] Connected to @{body.get('username')} "
              f"({body.get('media_count', '?')} existing media)")
    else:
        print(f"    [!] WARN: GET /{ig_business_id} returned {r.status_code}. "
              f"Token may still work for publishing — but verify with a test "
              f"upload. Body: {r.text[:200]}")


def main():
    print("="*60)
    print(" Instagram Reels OAuth Setup")
    print("="*60)
    print("\nThis script asks for 3 values from your Meta for Developers app:\n"
          "  • App ID\n"
          "  • App Secret\n"
          "  • A short-lived USER access token (from Graph API Explorer)\n"
          "\nLeave anything blank to abort.\n")

    app_id = input("Facebook App ID: ").strip()
    if not app_id:
        sys.exit("aborted")
    app_secret = getpass("Facebook App Secret (hidden): ").strip()
    if not app_secret:
        sys.exit("aborted")
    short_token = getpass("Short-lived USER access token (hidden): ").strip()
    if not short_token:
        sys.exit("aborted")

    long_token = _exchange_for_long_lived(app_id, app_secret, short_token)
    ig_business_id = _find_ig_business_account_id(long_token)
    _verify_publish_permission(long_token, ig_business_id)

    print()
    print("=" * 60)
    print(" SAVE THESE AS GITHUB REPOSITORY SECRETS")
    print("=" * 60)
    print(f"\n  IG_ACCESS_TOKEN          = {long_token[:24]}...{long_token[-8:]}")
    print(f"  IG_BUSINESS_ACCOUNT_ID   = {ig_business_id}")
    print(f"\n  Full token saved to:  instagram_token.txt")
    print(f"  Account ID saved to:  instagram_account_id.txt\n")

    with open("instagram_token.txt", "w", encoding="utf-8") as f:
        f.write(long_token)
    with open("instagram_account_id.txt", "w", encoding="utf-8") as f:
        f.write(ig_business_id)

    print("To use locally, add to your .env:")
    print(f'  IG_ACCESS_TOKEN={long_token[:12]}...')
    print(f'  IG_BUSINESS_ACCOUNT_ID={ig_business_id}\n')
    print("To use in GitHub Actions, add the same 2 values as Repository Secrets:")
    print("  Repository → Settings → Secrets and variables → Actions → New secret")
    print()
    print("⏰ Long-lived tokens last ~60 days. Re-run this script to refresh.")


if __name__ == "__main__":
    main()
