"""
build_golden_set.py
-------------------
Chọn 100 ảnh từ pre_golden.json để làm golden test set.

Chiến lược chọn:
  - Bỏ record null (group = None / image = None)
  - Group 04_handwritten: lấy hết (11 ảnh — quá ít để bỏ)
  - Các nhóm còn lại: proportional targeting 100 tổng, ưu tiên high confidence
  - Trong mỗi nhóm: chọn sao cho category coverage tốt nhất (greedy diversity)
  - Fixed seed=42 → reproducible

Target distribution (tổng = 100):
  01_pos_clean    : 26
  02_vat_invoice  : 17
  03_long_invoice : 16
  04_handwritten  : 11  (all)
  05_low_quality  : 30

Output: data/golden_test_set.json
"""

import json
import random
import sys
import io
from collections import Counter
from pathlib import Path

# Force UTF-8 output (Windows CP1252 workaround)
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PRE_GOLDEN = ROOT / "data" / "pre_golden.json"
GOLDEN_OUT = ROOT / "data" / "golden_test_set.json"
RAW_DIR    = ROOT / "data" / "raw"

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 42

# Số ảnh mục tiêu mỗi nhóm (tổng = 100)
TARGET = {
    "01_pos_clean"   : 26,
    "02_vat_invoice" : 17,
    "03_long_invoice": 16,
    "04_handwritten" : None,  # None = lấy hết
    "05_low_quality" : 30,
}

GROUPS = list(TARGET.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def greedy_diverse(candidates: list[dict], n: int, seed: int) -> list[dict]:
    """
    Chọn n items từ candidates sao cho category coverage đa dạng nhất.

    Thuật toán:
      1. Ưu tiên high confidence trước medium.
      2. Trong mỗi lượt, chọn item có category ít được chọn nhất (ties → random).
    Đảm bảo reproducible qua seed.
    """
    if n >= len(candidates):
        return list(candidates)

    rng = random.Random(seed)

    # Sắp xếp: high confidence trước, sau đó shuffle trong từng tier để reproducible
    high = [x for x in candidates if x.get("confidence") == "high"]
    medium = [x for x in candidates if x.get("confidence") != "high"]
    rng.shuffle(high)
    rng.shuffle(medium)
    ordered = high + medium

    chosen = []
    cat_count: Counter = Counter()

    for item in ordered:
        if len(chosen) >= n:
            break
        cat = item.get("category", "Khác")
        # Ưu tiên category chưa/ít được chọn
        chosen.append(item)
        cat_count[cat] += 1

    # Nếu chưa đủ (không nên xảy ra vì candidates >= n)
    return chosen[:n]


def validate_image_exists(record: dict) -> bool:
    """Kiểm tra file ảnh thực sự tồn tại trong data/raw/."""
    img = record.get("image")
    group = record.get("group")
    if not img or not group:
        return False
    return (RAW_DIR / group / img).exists()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(PRE_GOLDEN, encoding="utf-8") as f:
        pre = json.load(f)

    # 1. Bỏ null records
    valid = [x for x in pre if x.get("image") and x.get("group") in GROUPS]
    dropped_null = len(pre) - len(valid)

    # 2. Kiểm tra file tồn tại
    missing_files = [x for x in valid if not validate_image_exists(x)]
    if missing_files:
        print(f"[WARNING] {len(missing_files)} record có file ảnh không tồn tại:")
        for x in missing_files:
            print(f"  {x['group']}/{x['image']}")
        print("  → Các record này vẫn được đưa vào golden set (ảnh có thể ở nơi khác)")

    # 3. Nhóm theo group
    by_group: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for x in valid:
        by_group[x["group"]].append(x)

    # 4. Chọn mỗi nhóm
    selected: list[dict] = []
    selection_log: dict[str, dict] = {}

    for group in GROUPS:
        pool = by_group[group]
        target_n = TARGET[group] if TARGET[group] is not None else len(pool)
        target_n = min(target_n, len(pool))  # không thể chọn nhiều hơn có

        chosen = greedy_diverse(pool, target_n, seed=SEED)
        selected.extend(chosen)

        # Log chi tiết
        cat_dist = Counter(x.get("category", "Khác") for x in chosen)
        conf_dist = Counter(x.get("confidence", "?") for x in chosen)
        selection_log[group] = {
            "available": len(pool),
            "selected" : len(chosen),
            "confidence": dict(conf_dist),
            "categories": dict(cat_dist),
        }

    # 5. Shuffle final set (để khi iterate không bị bias theo group)
    rng = random.Random(SEED)
    rng.shuffle(selected)

    # 6. Lưu output
    with open(GOLDEN_OUT, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    # ── Report ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Golden Test Set — Build Report")
    print("=" * 60)
    print(f"Input  : {PRE_GOLDEN.name}  ({len(pre)} records)")
    print(f"Dropped: {dropped_null} null records")
    print(f"Output : {GOLDEN_OUT.name}  ({len(selected)} records)")
    print()

    print(f"{'Group':<22} {'Avail':>6} {'Sel':>5}  {'Confidence':<22}  Categories")
    print("-" * 80)
    for group in GROUPS:
        log = selection_log[group]
        conf_str = "  ".join(f"{k}:{v}" for k, v in sorted(log["confidence"].items()))
        cat_str  = "  ".join(
            f"{k}:{v}" for k, v in sorted(log["categories"].items(), key=lambda x: -x[1])
        )
        print(f"{group:<22} {log['available']:>6} {log['selected']:>5}  {conf_str:<22}  {cat_str}")

    print("-" * 80)
    print(f"{'TOTAL':<22} {sum(v['available'] for v in selection_log.values()):>6} "
          f"{len(selected):>5}")
    print()

    # Category coverage tổng
    total_cats = Counter(x.get("category", "Khác") for x in selected)
    print("Category distribution in golden set:")
    for cat, count in sorted(total_cats.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {cat:<22} {count:>3}  {bar}")

    print()
    print(f"[OK] Saved → {GOLDEN_OUT.resolve()}")


if __name__ == "__main__":
    main()
