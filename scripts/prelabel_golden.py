"""
Pre-label receipt images using OpenAI Vision API for golden test set creation.

Outputs data/pre_golden.json — a list of entries you review and correct
before promoting to data/golden_test_set.json.

Usage:
    python scripts/prelabel_golden.py [--group GROUP] [--limit N] [--overwrite]

    --group    Process only one group (e.g. 01_pos_clean)
    --limit    Max images to process in total (default: all)
    --overwrite  Re-process images already in pre_golden.json

Requirements:
    pip install openai

Environment:
    OPENAI_API_KEY — required
"""

import argparse
import base64
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
OUTPUT_FILE = ROOT / "data" / "pre_golden.json"

GROUPS = [
    "01_pos_clean",
    "02_vat_invoice",
    "03_long_invoice",
    "04_handwritten",
    "05_low_quality",
]

CATEGORIES = [
    "Ăn uống", "Đi lại", "Mua sắm", "Giải trí",
    "Hoá đơn tiện ích", "Sức khoẻ", "Giáo dục",
    "Du lịch", "Nhà cửa", "Khác",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a Vietnamese accounting expert specialised in reading Vietnamese receipts, "
    "including POS receipts (biên lai POS), VAT invoices (hóa đơn GTGT), long supermarket "
    "invoices, handwritten bills, and low-quality photos. "
    "Preserve all Vietnamese diacritics exactly as printed. "
    "Return integer VND amounts with no commas or currency symbols. "
    "If a field is not visible or unreadable, return null — never guess or invent data."
)

USER_PROMPT = f"""Analyse this Vietnamese receipt image and extract all fields below.

Reasoning steps (do these mentally before outputting):
1. Identify receipt type: POS / VAT invoice / supermarket long receipt / handwritten / other.
2. Locate merchant name (tên cửa hàng / tên đơn vị bán / công ty).
3. Locate transaction date (ngày) → DD-MM-YYYY. Locate time (giờ) → HH:MM (24h).
4. Extract every line item: name, quantity, unit_price, line total.
   IMPORTANT: If there are service fees, room charges, or hourly fees (tiền giờ, tiền phòng,
   phí dịch vụ) printed OUTSIDE the main items table — include them in items as separate
   entries. Every amount that contributes to the total must appear as an item.
5. Verify: sum(items.total) must equal total_amount (±1 VND rounding). If not, re-read the
   receipt for missed charges.
6. Find subtotal (trước thuế / chưa VAT), VAT amount (thuế GTGT), total (tổng cộng).
7. Identify payment method: cash / card / transfer / unknown.
8. Assign expense category from: {', '.join(CATEGORIES)}.
   Rules (apply in this order):
   a) If the merchant is clearly an entertainment venue (karaoke, bi-a/billiards, bowling,
      rạp chiếu phim, game center, khu vui chơi) → always "Giải trí", regardless of whether
      food/drinks make up the majority of the bill. The customer is there for entertainment;
      food and drinks are incidental.
   b) Classify by item content, not merchant name. Key rules per category:
      - "Ăn uống": food, drinks, ingredients, restaurant, cafe, grocery where items are food.
        Examples: rau củ, thịt cá, sữa, bia rượu, cà phê, trà sữa, gia vị, bánh kẹo.
      - "Mua sắm": clothing, electronics, cosmetics, personal care (non-medical), toys.
        Examples: quần áo, giày dép, điện thoại, laptop, mỹ phẩm, đồ chơi, quà tặng.
      - "Nhà cửa": home appliances, furniture, construction materials, home repair services,
        cleaning services. Examples: tủ lạnh, máy giặt, điều hoà, nồi chảo, bàn ghế, sơn
        nhà, vật tư xây dựng, sửa chữa điện máy gia dụng, nhân công sửa chữa, dọn vệ sinh.
        IMPORTANT: "sửa chữa [any home appliance/equipment]" and "chi phí nhân công sửa chữa"
        are always "Nhà cửa", even when merchant name is missing.
      - "Sức khoẻ": pharmacy, clinic, hospital, gym, vitamins, medical devices.
      - "Giáo dục": books, tuition, courses, school supplies used for study.
      - "Hoá đơn tiện ích": electricity (EVN/CTĐL), water, gas, internet, cable TV, phone bills.
      - "Đi lại": fuel, parking, ride-hailing (GrabCar/Be), bus/train tickets (short distance).
      - "Du lịch": hotel, flights, long-distance intercity transport, tour packages.
      - For neutral merchants (supermarket, minimart) → use the category of items that account
        for ≥70% of the total. If mixed and unclear → "Mua sắm".
   c) If spend is roughly equal (50/50) → prefer the more specific category
      (Sức khoẻ > Mua sắm, Giáo dục > Mua sắm).
   d) Each receipt is classified independently — food bought during a trip is still "Ăn uống".
9. Self-assess confidence: high (all fields clear), medium (some fields uncertain), low (image unclear or many nulls).

Hard constraints:
- Do NOT invent items not visible in the image.
- Digital booking confirmations and e-tickets (xác nhận đặt vé, vé điện tử) that show a
  merchant, date, and total amount ARE valid receipts — extract them normally, do NOT mark
  as not_a_receipt. Treat each ticket type as a line item; if unit_price is not shown,
  calculate it as total_amount / quantity.
- Bank transfer confirmations (biên lai chuyển tiền — Techcombank, VCB, MB, v.v.) ARE valid
  receipts. Set merchant_name to null if the recipient is an individual (all-caps personal
  name, no business identifier). Use the "Lời nhắn" (transfer note) as the item name if
  present. Set payment_method to "transfer". For category: infer from the note only if
  unambiguous (e.g. "tiền phòng" → Nhà cửa, "học phí" → Giáo dục); otherwise use "Khác".
- E-wallet payment receipts (MoMo, ZaloPay, VNPay, ViettelPay — "Chi Tiết Giao Dịch" screen)
  ARE valid receipts. Set payment_method to "e_wallet". Use "Nhà cung cấp" as merchant_name
  and "Kỳ thanh toán" as the item name when paying utility bills.
- Only mark not_a_receipt if the image has NO transaction information whatsoever
  (e.g. a random photo, a menu without prices, a blank page).
- Dates must be between 2015-01-01 and today. Times in 24h format HH:MM.
- All monetary values are positive integers in VND.
- unit_price and quantity are per line item — not cumulative.

Return ONLY a JSON object with this exact structure (no extra keys, no markdown):
{{
  "merchant_name": string | null,
  "date": "DD-MM-YYYY" | null,
  "time": "HH:MM" | null,
  "items": [
    {{
      "name": string,
      "quantity": number,
      "unit_price": integer | null,
      "total": integer | null
    }}
  ],
  "subtotal": integer | null,
  "vat": integer | null,
  "total_amount": integer | null,
  "payment_method": "cash" | "card" | "transfer" | "e_wallet" | "unknown" | null,
  "category": string,
  "confidence": "high" | "medium" | "low"
}}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")


def call_api(client: OpenAI, image_path: Path, retries: int = 3) -> dict:
    b64 = encode_image(image_path)
    media_type = get_media_type(image_path)

    # Long invoices (group 03) can have 30-50 items — 1024 tokens is not enough
    # and causes truncated JSON. Use 4096 as a safe ceiling across all groups.
    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                temperature=0,
                max_output_tokens=4096,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:{media_type};base64,{b64}",
                                "detail": "high",
                            },
                            {"type": "input_text", "text": USER_PROMPT},
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

        except json.JSONDecodeError as e:
            log.warning("JSON parse error attempt %d/%d for %s: %s", attempt, retries, image_path.name, e)
            # Truncated JSON will fail identically on every retry — no point retrying
            if not raw.rstrip().endswith("}"):
                log.warning("Response appears truncated (does not end with '}') — skipping retries")
                return _error_entry(f"truncated_response: {e}")
            if attempt == retries:
                return _error_entry(f"json_parse_error: {e}")

        except Exception as e:
            log.warning("API error attempt %d/%d for %s: %s", attempt, retries, image_path.name, e)
            if attempt == retries:
                return _error_entry(f"api_error: {e}")
            time.sleep(2 ** attempt)

    return _error_entry("unknown_error")


def _error_entry(reason: str) -> dict:
    return {
        "merchant_name": None,
        "date": None,
        "time": None,
        "items": [],
        "subtotal": None,
        "vat": None,
        "total_amount": None,
        "payment_method": None,
        "category": "Khác",
        "confidence": "low",
        "_error": reason,
    }


# ---------------------------------------------------------------------------
# JSON persistence  (append-safe: load → update → save)
# ---------------------------------------------------------------------------

def load_output() -> list[dict]:
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_output(entries: list[dict]) -> None:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def existing_images(entries: list[dict]) -> set[str]:
    return {e["image"] for e in entries if "image" in e}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_images(only_group: str | None) -> list[tuple[Path, str]]:
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
    parser = argparse.ArgumentParser(description="Pre-label receipt images for golden test set")
    parser.add_argument("--group", default=None, help="Process only one group (e.g. 01_pos_clean)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of images to process")
    parser.add_argument("--overwrite", action="store_true", help="Re-process already-labeled images")
    parser.add_argument("--dry-run", action="store_true", help="List images without calling API")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        log.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key) if not args.dry_run else None

    entries = load_output()
    done = existing_images(entries) if not args.overwrite else set()

    images = collect_images(args.group)
    to_process = [(p, g) for p, g in images if p.name not in done]

    if args.limit:
        to_process = to_process[: args.limit]

    log.info(
        "Total images: %d | Already labeled: %d | To process: %d",
        len(images), len(done), len(to_process),
    )

    if args.dry_run:
        for img_path, group in to_process:
            print(f"  {group}/{img_path.name}")
        return

    for i, (img_path, group) in enumerate(to_process, 1):
        log.info("[%d/%d] %s / %s", i, len(to_process), group, img_path.name)

        fields = call_api(client, img_path)
        entry = {
            "image": img_path.name,
            "group": group,
            **fields,
        }

        if args.overwrite:
            entries = [e for e in entries if e.get("image") != img_path.name]
        entries.append(entry)

        # Save after every image — safe to interrupt
        save_output(entries)

        time.sleep(0.3)

    log.info("Done. %d entries saved to %s", len(entries), OUTPUT_FILE)
    log.info(
        "Next step: review %s, correct mistakes, then copy to data/golden_test_set.json",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
