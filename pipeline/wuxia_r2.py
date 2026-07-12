"""
Wuxia R2 storage — pipeline/wuxia_r2.py
=======================================

Upload finished episodes to Cloudflare R2 (S3-compatible). Net-new — no R2 code
existed in the repo (the CLOUDFLARE_* secrets are Workers-AI, NOT R2). R2 needs
its own bucket + S3 access-key/secret.

GATED: callers should only invoke when R2_BUCKET is set. If boto3 or creds are
missing, this raises — the driver catches it and falls back to the GHA artifact
(non-fatal), matching explainer.yml's "leave mp4 as artifact" behaviour.

Required env:
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
Optional:
  R2_PUBLIC_BASE_URL  (public/custom-domain base; if unset returns an r2:// ref)
"""
from __future__ import annotations

import os


def _client():
    import boto3
    from botocore.config import Config

    account = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def upload_episode(local_path: str, key: str, content_type: str = "video/mp4") -> str:
    """Upload local_path to R2 under `key`. Returns a public URL if
    R2_PUBLIC_BASE_URL is configured, else an r2://bucket/key reference."""
    bucket = os.environ["R2_BUCKET"]
    client = _client()
    client.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": content_type})
    base = os.environ.get("R2_PUBLIC_BASE_URL", "").rstrip("/")
    return f"{base}/{key}" if base else f"r2://{bucket}/{key}"
