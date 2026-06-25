#!/usr/bin/env python3

import os
import sys
import concurrent.futures
import threading
import fnmatch

import boto3
from botocore.config import Config
from huggingface_hub import HfApi, hf_hub_url
import requests

# ── configuration ──────────────────────────────────────────────────────────────

ENDPOINT_URL      = os.environ["S3_ENDPOINT_URL"]
BUCKET            = os.environ["S3_BUCKET"]
AWS_KEY           = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET        = os.environ["AWS_SECRET_ACCESS_KEY"]
HF_TOKEN = os.environ.get("HF_TOKEN")
SUBSET            = os.environ.get("C4_SUBSET", "en/*")
S3_PREFIX         = os.environ.get("S3_PREFIX", "c4").rstrip("/")
WORKERS           = int(os.environ.get("WORKERS", "4"))
CHUNK_SIZE        = int(os.environ.get("CHUNK_SIZE_MB", "64")) * 1024 * 1024

DATASET_REPO      = "allenai/c4"
PRINT_LOCK        = threading.Lock()


# ── helpers ────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)

def make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )

def key_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False

def stream_to_s3(s3, hf_path: str, s3_key: str) -> None:
    """Download one HF file and multipart-upload it to S3."""
    url = hf_hub_url(
        repo_id=DATASET_REPO,
        filename=hf_path,
        repo_type="dataset",
    )
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()

        # Content-Length may not always be present
        total = int(resp.headers.get("Content-Length", 0))

        # Use multipart upload so we can stream without buffering the whole file
        mpu = s3.create_multipart_upload(Bucket=BUCKET, Key=s3_key)
        upload_id = mpu["UploadId"]
        parts = []
        part_number = 1
        buf = bytearray()

        try:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                buf.extend(chunk)
                if len(buf) >= CHUNK_SIZE:
                    part = s3.upload_part(
                        Bucket=BUCKET,
                        Key=s3_key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=bytes(buf),
                    )
                    parts.append({"PartNumber": part_number, "ETag": part["ETag"]})
                    part_number += 1
                    buf = bytearray()

            # Upload the final (possibly partial) chunk
            if buf:
                part = s3.upload_part(
                    Bucket=BUCKET,
                    Key=s3_key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=bytes(buf),
                )
                parts.append({"PartNumber": part_number, "ETag": part["ETag"]})

            s3.complete_multipart_upload(
                Bucket=BUCKET,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            s3.abort_multipart_upload(Bucket=BUCKET, Key=s3_key, UploadId=upload_id)
            raise


def mirror_file(hf_path: str) -> tuple[str, str]:
    """Mirror one file; return (hf_path, status)."""
    s3_key = f"{S3_PREFIX}/{hf_path}"
    s3 = make_s3_client()

    if key_exists(s3, s3_key):
        log(f"  SKIP  {hf_path}  (already in S3)")
        return hf_path, "skipped"

    log(f"  START {hf_path}")
    stream_to_s3(s3, hf_path, s3_key)
    log(f"  DONE  {hf_path}")
    return hf_path, "uploaded"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    api = HfApi(token=HF_TOKEN)

    log(f"Listing files in {DATASET_REPO} matching '{SUBSET}' …")
    all_files = [
        f
        for f in api.list_repo_files(DATASET_REPO, repo_type="dataset")
        if fnmatch.fnmatch(f, SUBSET)
    ]

    if not all_files:
        log(f"No files matched pattern '{SUBSET}'. Exiting.")
        sys.exit(1)

    log(f"Found {len(all_files)} files to mirror into s3://{BUCKET}/{S3_PREFIX}/\n")

    # Ensure bucket exists (create it if not)
    s3 = make_s3_client()
    try:
        s3.head_bucket(Bucket=BUCKET)
    except Exception:
        log(f"Bucket '{BUCKET}' not found — creating it.")
        s3.create_bucket(Bucket=BUCKET)

    counts = {"uploaded": 0, "skipped": 0, "failed": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(mirror_file, p): p for p in all_files}
        for fut in concurrent.futures.as_completed(futures):
            hf_path = futures[fut]
            try:
                _, status = fut.result()
                counts[status] += 1
            except Exception as exc:
                log(f"  FAIL  {hf_path}: {exc}")
                counts["failed"] += 1

    log(
        f"\nDone.  uploaded={counts['uploaded']}  "
        f"skipped={counts['skipped']}  failed={counts['failed']}"
    )
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
