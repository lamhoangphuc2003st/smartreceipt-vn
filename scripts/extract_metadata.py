"""
Extract metadata from receipt images using OpenAI Vision API.

Usage:
    python scripts/extract_metadata.py [--dry-run] [--group GROUP] [--overwrite]

Requirements:
    pip install openai pillow

Environment:
    OPENAI_API_KEY — required
"""

import argparse
import base64
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
METADATA_CSV = ROOT / "data" / "metadata.csv"

GROUPS = [
    "01_pos_clean",
    "02_vat_invoice",
    "03_long_invoice",
    "04_handwritten",
    "05_low_quality",
]

CSV_FIELDS = ["filename", "group", "merchant_name", "date", "has_vat", "total_amount", "notes"]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a Vietnamese accounting expert specialised in reading Vietnamese receipts, "
    "including POS receipts, VAT invoices (hóa đơn VAT/GTGT), long invoices, handwritten bills, "
    "and low-quality photos. Extract the requested fields accurately. "
    "Preserve Vietnamese diacritics exactly as they appear. "
    "Return integer VND amounts with no commas or currency symbols. "
    "If a field is not visible or unreadable, return null — never guess."
)

USER_PROMPT = """Look at this Vietnamese receipt image and extract the following fields.

Steps to follow:
1. Identify the receipt type (POS / VAT invoice / handwritten / other).
2. Find the merchant name (tên cửa hàng / tên đơn vị bán).
3. Find the transaction date (ngày) in YYYY-MM-DD format.
4. Determine if VAT is shown separately on the receipt (has_vat: true/false).
5. Find the total amount paid in VND (tổng cộng / total), as an integer.
6. Note any unusual characteristics (blurry, rotated, partial, handwritten, multiple receipts, etc.).

Constraints:
- Do NOT invent data not visible in the image.
- If the image is not a receipt, set merchant_name to null and add a note "not_a_receipt".
- Dates must be between 2015-01-01 and today.
- total_amount must be a positive integer (VND), null if unreadable.

Return ONLY a JSON object with these exact keys:
{
  "merchant_name": string | null,
  "date": "YYYY-MM-DD" | null,
  "has_vat": true | false | null,
  "total_amount": integer | null,
  "notes": string | null
}"""

# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def encode_image(path: Path) -> str:
    """Return base64-encoded image."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")


def extract_fields(client: OpenAI, image_path: Path, retries: int = 3) -> dict:
    b64 = encode_image(image_path)
    media_type = get_media_type(image_path)

    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                temperature=0,
                max_output_tokens=512,
                input=[
                   {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": SYSTEM_PROMPT}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:{media_type};base64,{b64}",
                                "detail": "high",
                            },
                            {
                                "type": "input_text",
                                "text": USER_PROMPT,
                            },
                        ],
                    },
                ],
            )

            raw = response.output_text.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw)

        except Exception as e:
            log.warning("API error: %s", e)
            return {
                "merchant_name": None,
                "date": None,
                "has_vat": None,
                "total_amount": None,
                "notes": f"api_error: {e}"
            }
# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_existing(csv_path: Path) -> set[str]:
    """Return set of filenames already in the CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["filename"] for row in reader}


def append_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_images(only_group: str | None) -> list[tuple[Path, str]]:
    """Return list of (image_path, group_name) for all images to process."""
    images = []
    for group in GROUPS:
        if only_group and group != only_group:
            continue
        group_dir = RAW_DIR / group
        if not group_dir.exists():
            log.warning("Group directory not found: %s", group_dir)
            continue
        for img in sorted(group_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                images.append((img, group))
    return images


def main():
    parser = argparse.ArgumentParser(description="Extract receipt metadata using OpenAI Vision")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be processed without calling the API")
    parser.add_argument("--group", default=None,
                        help="Process only a specific group (e.g. 01_pos_clean)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process images already present in metadata.csv")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        log.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key) if not args.dry_run else None
    existing = load_existing(METADATA_CSV) if not args.overwrite else set()

    images = collect_images(args.group)
    log.info("Found %d images across %d groups.", len(images),
             len({g for _, g in images}))

    skipped = 0
    processed = 0
    errors = 0

    for img_path, group in images:
        filename = img_path.name

        if filename in existing:
            log.debug("Skipping already-processed: %s", filename)
            skipped += 1
            continue

        if args.dry_run:
            print(f"[DRY RUN] Would process: {group}/{filename}")
            continue

        log.info("Processing [%s] %s", group, filename)
        fields = extract_fields(client, img_path)

        row = {
            "filename": filename,
            "group": group,
            "merchant_name": fields.get("merchant_name"),
            "date": fields.get("date"),
            "has_vat": fields.get("has_vat"),
            "total_amount": fields.get("total_amount"),
            "notes": fields.get("notes"),
        }
        append_row(METADATA_CSV, row)
        processed += 1

        if fields.get("notes") and ("error" in str(fields.get("notes", "")).lower()):
            errors += 1

        # Brief pause to stay within rate limits
        time.sleep(0.5)

    log.info(
        "Done. processed=%d  skipped=%d  errors=%d  output=%s",
        processed, skipped, errors, METADATA_CSV,
    )


if __name__ == "__main__":
    main()
