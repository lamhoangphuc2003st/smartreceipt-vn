"""
scripts/generate_training_data.py
-----------------------------------
Build training CSV for PhoBERT expense classifier.

Sources:
  1. MANUAL — pre_golden.json: 187 fully labeled receipts (all 187 raw images)
     This includes the 100 golden test set + 87 extra labeled records.
  2. SYNTHETIC — augmented variants (random item drop) for class balancing

Output: data/training/phobert_training.csv
Columns: text, label, source, confidence

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --augment 8   # 8x per sample (default 5)
    python scripts/generate_training_data.py --no-augment  # manual only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.phobert_classifier import CATEGORIES, ExpenseClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PRE_GOLDEN_PATH = PROJECT_ROOT / "data" / "pre_golden.json"
GOLDEN_PATH     = PROJECT_ROOT / "data" / "golden_test_set.json"
OUTPUT_PATH     = PROJECT_ROOT / "data" / "training" / "phobert_training.csv"


# ── Text builder ──────────────────────────────────────────────────────────────

def build_text(merchant_name: str | None, items: list[dict]) -> str:
    item_names = [it["name"] for it in (items or []) if it.get("name")]
    return ExpenseClassifier._build_input(merchant_name, item_names)


# ── Source 1: Manual labels ───────────────────────────────────────────────────

def load_manual_samples(pre_golden: list[dict], golden_images: set[str]) -> tuple[list[dict], list[dict]]:
    """
    Split pre_golden into:
      - train_samples: records NOT in golden test set (safe to train on)
      - golden_samples: records IN golden test set (held out — do NOT train)

    Returns (train_samples, golden_samples) for reporting.
    """
    train, held_out = [], []
    skipped = 0

    for r in pre_golden:
        img = r.get("image")
        category = r.get("category")

        if not category or category not in CATEGORIES:
            skipped += 1
            continue

        text = build_text(r.get("merchant_name"), r.get("items", []))
        if not text.strip():
            skipped += 1
            continue

        entry = {"text": text, "label": category, "source": "manual", "confidence": 1.0}

        if img and img in golden_images:
            held_out.append(entry)
        else:
            train.append(entry)

    log.info(
        "Manual samples: %d train, %d golden (held out), %d skipped",
        len(train), len(held_out), skipped,
    )
    return train, held_out


# ── Source 2: Augmentation ────────────────────────────────────────────────────

def augment_samples(samples: list[dict], n_per_sample: int) -> list[dict]:
    """
    Augment training data by randomly dropping 1-2 items.
    Helps balance minority classes and improves robustness.
    """
    augmented = []
    for s in samples:
        text = s["text"]
        parts = text.split(" | ", 1)
        merchant_part = parts[0] if len(parts) > 1 else ""
        items_part    = parts[1] if len(parts) > 1 else parts[0]
        items_list    = [x.strip() for x in items_part.split(",") if x.strip()]

        for _ in range(n_per_sample):
            if len(items_list) > 2:
                n_drop = random.randint(1, min(2, len(items_list) - 1))
                kept = random.sample(items_list, len(items_list) - n_drop)
                new_items = ", ".join(kept)
            else:
                new_items = items_part

            new_text = f"{merchant_part} | {new_items}" if merchant_part else new_items
            augmented.append({
                "text": new_text,
                "label": s["label"],
                "source": "synthetic",
                "confidence": 0.95,
            })

    log.info("Generated %d augmented samples", len(augmented))
    return augmented


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(train: list[dict], held_out: list[dict]) -> None:
    all_train = train
    cats = Counter(s["label"] for s in all_train)
    sources = Counter(s["source"] for s in all_train)

    log.info("=" * 50)
    log.info("TRAINING SET: %d samples", len(all_train))
    log.info("HELD OUT (golden test set — NOT trained): %d samples", len(held_out))
    log.info("By source: %s", dict(sources))
    log.info("By category (training only):")
    max_n = max(cats.values()) if cats else 1
    for cat in CATEGORIES:
        n = cats.get(cat, 0)
        bar = "█" * int(n / max_n * 20)
        log.info("  %-25s %3d  %s", cat, n, bar)
    log.info("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--augment",     type=int, default=5,  help="Augmentation copies per manual sample")
    p.add_argument("--no-augment",  action="store_true",  help="Skip augmentation")
    p.add_argument("--output",      default=str(OUTPUT_PATH))
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(PRE_GOLDEN_PATH, encoding="utf-8") as f:
        pre_golden = json.load(f)
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    golden_images = {r["image"] for r in golden}

    # Source 1: manual labels (non-golden only for training)
    train_manual, held_out = load_manual_samples(pre_golden, golden_images)

    all_train: list[dict] = list(train_manual)

    # Source 2: augmentation on training split only (never augment golden)
    if not args.no_augment and args.augment > 0:
        aug = augment_samples(train_manual, n_per_sample=args.augment)
        all_train.extend(aug)

    random.shuffle(all_train)

    # Write training CSV
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source", "confidence"])
        writer.writeheader()
        writer.writerows(all_train)

    # Also write held-out eval CSV (for reporting, not training)
    eval_path = output_path.parent / "phobert_eval.csv"
    with open(eval_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source", "confidence"])
        writer.writeheader()
        writer.writerows(held_out)

    print_summary(all_train, held_out)
    log.info("Training CSV -> %s", output_path)
    log.info("Eval CSV     -> %s", eval_path)


if __name__ == "__main__":
    main()
