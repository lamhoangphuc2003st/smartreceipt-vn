"""
src/evaluation/run_eval.py
---------------------------
Entry point để đánh giá SmartReceipt VN trên golden test set.

Hai mode:

  --mode classification (mặc định, nhanh, miễn phí)
      Dùng ground-truth merchant_name + items để đánh giá PhoBERT classifier.
      Không cần Gemini API. Chạy ~5 giây.

  --mode full
      Chạy toàn bộ pipeline (preprocessing → Gemini → validation → classifier)
      trên từng ảnh trong golden set. Cần GEMINI_API_KEY. Mất ~5-8 phút, ~$0.10.

Kết quả được lưu vào docs/experiments/<output_file>.md

Ví dụ:
    python -m src.evaluation.run_eval
    python -m src.evaluation.run_eval --mode classification --output exp_001_phobert_v1.md
    python -m src.evaluation.run_eval --mode full --output exp_002_full_pipeline_v1.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows terminal often defaults to cp1252 — force UTF-8 so Vietnamese prints correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
GOLDEN_PATH  = ROOT / "data" / "golden_test_set.json"
RAW_DIR      = ROOT / "data" / "raw"
EXPERIMENTS  = ROOT / "docs" / "experiments"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_golden() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def _image_path(record: dict) -> Path:
    """Resolve image filename → absolute path under data/raw/<group>/."""
    return RAW_DIR / record["group"] / record["image"]


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v:.1%}"


# ── Mode 1: Classification-only ────────────────────────────────────────────────

def run_classification_eval(golden: list[dict]) -> dict:
    """
    Đánh giá PhoBERT classifier bằng cách dùng ground-truth merchant + items.
    Không cần ảnh, không cần Gemini.
    """
    from src.classification import ExpenseClassifier

    print("Loading PhoBERT classifier...")
    clf = ExpenseClassifier.from_pretrained("models/phobert-expense-v1")
    if not clf.using_phobert:
        print("WARNING: PhoBERT không load được, đang dùng keyword fallback.")
    else:
        print("PhoBERT loaded OK.\n")

    preds: list[dict] = []
    latencies: list[float] = []

    for i, gt in enumerate(golden):
        merchant = gt.get("merchant_name")
        items    = [it["name"] for it in (gt.get("items") or [])]

        t0 = time.perf_counter()
        result = clf.classify(merchant, items)
        latency = time.perf_counter() - t0

        preds.append({
            "image"         : gt["image"],
            "group"         : gt["group"],
            "merchant_name" : merchant,
            "date"          : gt.get("date"),
            "total_amount"  : gt.get("total_amount"),
            "items"         : gt.get("items", []),
            "category"      : result.category,
        })
        latencies.append(latency)

        # Progress
        if (i + 1) % 20 == 0 or (i + 1) == len(golden):
            print(f"  [{i+1}/{len(golden)}] {gt['image']} -> {result.category} ({result.confidence:.2f})")

    from src.evaluation.metrics import evaluate_dataset
    report = evaluate_dataset(preds, golden, latencies)
    return report, preds


# ── Mode 2: Full pipeline ──────────────────────────────────────────────────────

def run_full_eval(golden: list[dict]) -> dict:
    """
    Chạy toàn bộ pipeline trên mỗi ảnh trong golden set.
    Cần GEMINI_API_KEY và ảnh trong data/raw/.
    """
    from src.pipeline.receipt_pipeline import ReceiptPipeline

    print("Initialising full pipeline...")
    pipeline = ReceiptPipeline()
    print("Pipeline ready.\n")

    preds: list[dict] = []
    latencies: list[float] = []

    for i, gt in enumerate(golden):
        img_path = _image_path(gt)
        if not img_path.exists():
            print(f"  [{i+1}/{len(golden)}] SKIP — image not found: {img_path.name}")
            # Pad with empty pred so index alignment stays correct
            preds.append({"image": gt["image"], "group": gt["group"]})
            latencies.append(0.0)
            continue

        t0 = time.perf_counter()
        result = pipeline.process(img_path)
        latency = time.perf_counter() - t0

        preds.append({
            "image"         : gt["image"],
            "group"         : gt["group"],
            "merchant_name" : result.merchant_name,
            "date"          : result.date,
            "total_amount"  : result.total_amount,
            "items"         : result.items,
            "category"      : result.category,
        })
        latencies.append(latency)

        status = "OK" if result.success else f"FAIL({result.error_type})"
        print(f"  [{i+1}/{len(golden)}] {gt['image']} -> {status} | "
              f"cat={result.category} | {latency:.1f}s")

    from src.evaluation.metrics import evaluate_dataset
    report = evaluate_dataset(preds, golden, latencies)
    return report, preds


# ── Report formatting ──────────────────────────────────────────────────────────

def _render_markdown(report, preds: list[dict], golden: list[dict], mode: str, version: str) -> str:
    from src.evaluation.metrics import get_failures

    fa = report.field_accuracy
    lines = [
        f"# Evaluation: {version}",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Mode:** {mode}  ",
        f"**Samples:** {report.n_samples}  ",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| End-to-end success | {_fmt(report.end_to_end_success)} |",
        f"| Merchant accuracy | {_fmt(fa.get('merchant', float('nan')))} |",
        f"| Date accuracy | {_fmt(fa.get('date', float('nan')))} |",
        f"| Total accuracy | {_fmt(fa.get('total', float('nan')))} |",
        f"| Items F1 | {_fmt(report.items_f1)} |",
        f"| Category accuracy | {_fmt(report.category_accuracy)} |",
        f"| Latency p50 | {report.latency_p50*1000:.0f} ms |" if not math.isnan(report.latency_p50) else "| Latency p50 | N/A |",
        f"| Latency p95 | {report.latency_p95*1000:.0f} ms |" if not math.isnan(report.latency_p95) else "| Latency p95 | N/A |",
        f"",
        f"## Per-difficulty breakdown",
        f"",
        f"| Group | N | Merchant | Date | Total | Items F1 | Category | E2E |",
        f"|-------|---|----------|------|-------|----------|----------|-----|",
    ]

    for g_name, gs in report.per_group.items():
        lines.append(
            f"| {g_name} | {gs.n} "
            f"| {_fmt(gs.merchant_accuracy)} "
            f"| {_fmt(gs.date_accuracy)} "
            f"| {_fmt(gs.total_accuracy)} "
            f"| {_fmt(gs.items_f1)} "
            f"| {_fmt(gs.category_accuracy)} "
            f"| {_fmt(gs.end_to_end_success)} |"
        )

    lines += [
        f"",
        f"## Category accuracy",
        f"",
        f"| Category | Accuracy |",
        f"|----------|----------|",
    ]
    for cat, acc in sorted(report.per_category_accuracy.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {_fmt(acc)} |")

    # Failure analysis (top 10)
    if mode == "full":
        failures = get_failures(preds, golden)
        lines += [
            f"",
            f"## Failure analysis (end-to-end failures, first 10)",
            f"",
            f"| Image | Group | Merchant? | Date? | Total? | Items F1 |",
            f"|-------|-------|-----------|-------|--------|----------|",
        ]
        for fail in failures[:10]:
            lines.append(
                f"| {fail['image']} | {fail['group']} "
                f"| {'✓' if fail['merchant_ok'] else '✗'} "
                f"| {'✓' if fail['date_ok'] else '✗'} "
                f"| {'✓' if fail['total_ok'] else '✗'} "
                f"| {fail['items_f1']:.2f} |"
            )

    lines += [
        f"",
        f"## Notes",
        f"",
        f"<!-- Add observations here -->",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SmartReceipt VN on golden test set")
    parser.add_argument(
        "--mode",
        choices=["classification", "full"],
        default="classification",
        help="classification: eval PhoBERT only (fast). full: run entire pipeline (needs API key).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename under docs/experiments/ (e.g. exp_001_phobert_v1.md). "
             "Defaults to exp_<timestamp>_<mode>.md",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version label for the report title (e.g. phobert-v1). Defaults to mode+timestamp.",
    )
    args = parser.parse_args()

    # Resolve output path
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_name = args.output or f"exp_{ts}_{args.mode}.md"
    output_path = EXPERIMENTS / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    version = args.version or f"{args.mode}-{ts}"

    print(f"=== SmartReceipt VN Evaluation ===")
    print(f"Mode:    {args.mode}")
    print(f"Golden:  {GOLDEN_PATH}")
    print(f"Output:  {output_path}")
    print()

    golden = _load_golden()
    print(f"Loaded {len(golden)} golden samples.\n")

    t_start = time.perf_counter()
    if args.mode == "classification":
        report, preds = run_classification_eval(golden)
    else:
        report, preds = run_full_eval(golden)
    elapsed = time.perf_counter() - t_start

    # Print summary to console
    fa = report.field_accuracy
    print(f"\n{'='*50}")
    print(f"RESULTS ({elapsed:.1f}s total)")
    print(f"{'='*50}")
    print(f"  Samples          : {report.n_samples}")
    print(f"  End-to-end       : {_fmt(report.end_to_end_success)}")
    print(f"  Merchant         : {_fmt(fa.get('merchant', float('nan')))}")
    print(f"  Date             : {_fmt(fa.get('date', float('nan')))}")
    print(f"  Total amount     : {_fmt(fa.get('total', float('nan')))}")
    print(f"  Items F1         : {_fmt(report.items_f1)}")
    print(f"  Category         : {_fmt(report.category_accuracy)}")
    if not math.isnan(report.latency_p50):
        print(f"  Latency p50/p95  : {report.latency_p50*1000:.0f}ms / {report.latency_p95*1000:.0f}ms")
    print()

    print("Per-category accuracy:")
    for cat, acc in sorted(report.per_category_accuracy.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 20)
        print(f"  {cat:<20} {_fmt(acc):>7}  {bar}")

    # Save markdown report
    md = _render_markdown(report, preds, golden, args.mode, version)
    output_path.write_text(md, encoding="utf-8")
    print(f"\nReport saved → {output_path}")


if __name__ == "__main__":
    main()
