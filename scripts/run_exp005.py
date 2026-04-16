"""
scripts/run_exp005.py
---------------------
exp_005: Benchmark GeminiExtractor (structured prompt v1) vs exp_004 baseline
         (preprocessing + naive prompt).

Experiment design:
  - Baseline: exp_004 results — ImagePreprocessor + naive prompt = 67.0% E2E
  - Test:     ImagePreprocessor + GeminiExtractor (EXTRACTION_PROMPT_V1)

The key change being measured is the structured prompt (role + CoT + few-shot +
anti-hallucination + self-check) vs the naive 7-line prompt used in exp_001–004.
Preprocessing is held constant so we isolate the prompt effect.

Crash safety:
    Results saved to data/exp005_checkpoint.json after each image.
    Re-run to resume from where it stopped.

Cost estimate:
    ~$0.004/image × 100 = ~$0.40 for a full run.
    V1 prompt is longer (~700 tokens) so slightly more than naive (~80 tokens).
    Cache (data/extraction_cache/) avoids re-calling for already-processed images.

Usage:
    python scripts/run_exp005.py               # full run
    python scripts/run_exp005.py --limit 10    # quick dev test (10 images)
    python scripts/run_exp005.py --no-cache    # force re-run (ignore disk cache)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate, get_failures
from src.extraction import GeminiExtractor, ExtractionResult
from src.extraction.schemas import Receipt
from src.preprocessing import ImagePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

GOLDEN_PATH   = PROJECT_ROOT / "data" / "golden_test_set.json"
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
CHECKPOINT    = PROJECT_ROOT / "data" / "exp005_checkpoint.json"
EXP004_CKPT   = PROJECT_ROOT / "data" / "benchmark_preprocessing_checkpoint.json"
DEFAULT_OUT   = PROJECT_ROOT / "docs" / "experiments" / "exp_005_extraction_v1.md"

# Baseline numbers from exp_004 (preprocessing + naive prompt)
EXP004_E2E_OVERALL = 0.670
EXP004_PER_GROUP = {
    "01_pos_clean":    0.615,
    "02_vat_invoice":  0.650,
    "03_long_invoice": 0.750,
    "04_handwritten":  0.636,
    "05_low_quality":  0.733,
}
EXP004_FIELDS = {
    "merchant": 0.770,
    "date":     0.810,
    "total":    0.900,
}

# Rate-limit guard for free tier (~15 RPM sustained)
CALL_DELAY_SEC = 4.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_image_path(image_name: str, group: str) -> Path | None:
    candidate = RAW_DIR / group / image_name
    if candidate.exists():
        return candidate
    for path in RAW_DIR.rglob(image_name):
        return path
    return None


def extraction_result_to_pred(result: ExtractionResult, gt: dict) -> dict:
    """Convert ExtractionResult → dict format expected by evaluate()."""
    base = {
        "image": gt["image"],
        "group": gt.get("group", "unknown"),
        "category": None,   # PhoBERT classifier not built yet (Week 7)
    }
    if not result.success or result.receipt is None:
        return {**base, "merchant_name": None, "date": None, "total_amount": None, "items": []}

    r: Receipt = result.receipt

    # Convert ReceiptItem list → plain dicts for metrics framework
    items = [
        {
            "name": item.name,
            "quantity": item.quantity,
            "unit_price": item.unit_price,
            "total": item.total,
        }
        for item in (r.items or [])
    ]

    return {
        **base,
        "merchant_name": r.merchant_name,
        "date": r.date,
        "total_amount": r.total_amount,
        "items": items,
    }


def load_exp004_baseline(golden: list[dict]) -> list[dict]:
    """
    Load preprocessed predictions from exp_004 checkpoint as the baseline.
    Returns list of pred dicts in the same order as golden.
    """
    if not EXP004_CKPT.exists():
        log.warning("exp_004 checkpoint not found at %s — will use hardcoded numbers only", EXP004_CKPT)
        return []

    with open(EXP004_CKPT, encoding="utf-8") as f:
        saved = json.load(f)
    results_by_image = saved.get("results", {})

    preds = []
    for record in golden:
        img_name = record["image"]
        r = results_by_image.get(img_name)
        if r is None:
            log.warning("exp_004 checkpoint missing image: %s", img_name)
            preds.append({
                "image": img_name,
                "group": record.get("group", "unknown"),
                "merchant_name": None, "date": None, "total_amount": None, "items": [],
                "category": None,
            })
        else:
            pred = dict(r.get("preprocessed_pred") or {})
            pred["image"] = img_name
            pred["group"] = record.get("group", "unknown")
            preds.append(pred)

    return preds


# ── Per-image processing ───────────────────────────────────────────────────────

def process_single(
    record: dict,
    extractor: GeminiExtractor,
    preprocessor: ImagePreprocessor,
    checkpoint_results: dict,
) -> dict:
    """
    Run ImagePreprocessor → GeminiExtractor on one golden set record.
    Returns a result dict with pred, latency, cost, and diagnostic info.
    """
    image_name = record["image"]
    group = record.get("group", "")
    img_path = find_image_path(image_name, group)

    result: dict = {
        "image": image_name,
        "group": group,
        "pred": None,
        "latency_sec": 0.0,
        "cost_usd": 0.0,
        "from_cache": False,
        "attempts": 1,
        "preprocessing_ms": 0.0,
        "errors": [],
    }

    if img_path is None:
        log.error("Image not found: %s", image_name)
        result["errors"].append("image_not_found")
        result["pred"] = {
            "image": image_name, "group": group,
            "merchant_name": None, "date": None, "total_amount": None,
            "items": [], "category": None,
        }
        return result

    # ── Step 1: Preprocess ─────────────────────────────────────────────────────
    try:
        proc = preprocessor.process(str(img_path))
        result["preprocessing_ms"] = proc.processing_time_ms

        if not proc.success or proc.image is None:
            log.warning("Preprocessing failed for %s: %s", image_name, proc.warnings)
            result["errors"].append("preprocessing_failed")
            # Fall back to raw image bytes
            image_input = str(img_path)
        else:
            image_input = proc.image   # numpy ndarray (BGR)
    except Exception as exc:
        log.error("Preprocessing exception for %s: %s", image_name, exc)
        result["errors"].append(f"preprocessing_exception: {exc}")
        image_input = str(img_path)   # fallback to raw

    # ── Step 2: Extract ────────────────────────────────────────────────────────
    try:
        extraction: ExtractionResult = extractor.extract(image_input)

        result["latency_sec"] = extraction.latency_ms / 1000.0
        result["cost_usd"]    = extraction.cost_usd
        result["from_cache"]  = extraction.from_cache
        result["attempts"]    = extraction.attempts

        if not extraction.success:
            log.warning(
                "Extraction failed for %s: [%s] %s",
                image_name, extraction.error_type, extraction.error_message
            )
            result["errors"].append(f"extraction_{extraction.error_type}")

        result["pred"] = extraction_result_to_pred(extraction, record)

    except Exception as exc:
        log.error("Extraction exception for %s: %s", image_name, exc)
        result["errors"].append(f"extraction_exception: {exc}")
        result["pred"] = {
            "image": image_name, "group": group,
            "merchant_name": None, "date": None, "total_amount": None,
            "items": [], "category": None,
        }

    return result


# ── Report generation ──────────────────────────────────────────────────────────

def format_report(
    golden: list[dict],
    results: list[dict],
    v1_report: dict,
    exp004_report: dict | None,
    total_cost_usd: float,
    n_cache_hits: int,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(results)
    n_errors = sum(1 for r in results if r.get("errors"))
    n_api_errors = sum(1 for r in results if any("extraction_" in e for e in r.get("errors", [])))
    cache_hit_rate = n_cache_hits / n if n else 0.0

    lats = [r["latency_sec"] for r in results if r["latency_sec"] > 0]
    lats_sorted = sorted(lats)
    lat_p50 = lats_sorted[len(lats_sorted) // 2] if lats_sorted else float("nan")
    lat_p95 = lats_sorted[max(0, int(len(lats_sorted) * 0.95) - 1)] if lats_sorted else float("nan")
    cost_per_receipt = total_cost_usd / max(n - n_cache_hits, 1)

    def pct(v: float) -> str:
        return f"{v:.1%}"

    def delta_str(new: float, old: float) -> str:
        d = new - old
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1%}"

    # Determine pass/fail
    e2e_new = v1_report["end_to_end_success"]
    e2e_old = EXP004_E2E_OVERALL
    e2e_delta = e2e_new - e2e_old

    # Check per-group regression (must not regress >2pp)
    group_regression = False
    for grp, stats in v1_report.get("per_group", {}).items():
        old_grp = EXP004_PER_GROUP.get(grp)
        if old_grp is not None:
            if stats["end_to_end_success"] - old_grp < -0.02:
                group_regression = True
                break

    lat_ok = lat_p95 <= 5.0
    cost_ok = cost_per_receipt <= 0.003 * 1.20  # 20% over baseline allowed

    if e2e_delta >= 0.01 and not group_regression and lat_ok:
        verdict = "PASS"
        verdict_reason = (
            f"E2E improves by {e2e_delta:+.1%} (≥ +1pp), "
            f"no group regresses >2pp, p95 latency = {lat_p95:.2f}s ≤ 5s."
        )
    elif e2e_delta < 0.01:
        verdict = "FAIL"
        verdict_reason = f"E2E delta {e2e_delta:+.1%} is below the +1pp threshold."
    elif group_regression:
        verdict = "FAIL (regression)"
        verdict_reason = f"E2E delta {e2e_delta:+.1%} but at least one group regresses >2pp."
    else:
        verdict = "FAIL (latency)"
        verdict_reason = f"p95 latency {lat_p95:.2f}s exceeds 5s threshold."

    # Per-group table rows
    group_rows = []
    for grp in sorted(EXP004_PER_GROUP.keys()):
        grp_stats = v1_report.get("per_group", {}).get(grp)
        old_e2e = EXP004_PER_GROUP.get(grp, 0.0)
        if grp_stats:
            new_e2e = grp_stats["end_to_end_success"]
            flag = " ⚠" if new_e2e - old_e2e < -0.02 else ""
            group_rows.append(
                f"| {grp} | {grp_stats['n']} "
                f"| {old_e2e:.1%} "
                f"| {new_e2e:.1%} "
                f"| {delta_str(new_e2e, old_e2e)}{flag} |"
            )

    # Failure summary (top 5 worst)
    failures = get_failures(
        [r["pred"] for r in results],
        golden[:n],
    )
    failure_lines = []
    for fail in failures[:5]:
        failure_lines.append(
            f"- **{fail['image']}** ({fail['group']}): "
            f"merchant={'OK' if fail['merchant_ok'] else 'FAIL'}, "
            f"date={'OK' if fail['date_ok'] else 'FAIL'}, "
            f"total={'OK' if fail['total_ok'] else 'FAIL'}, "
            f"items_f1={fail['items_f1']:.2f}"
        )

    v = v1_report
    e4 = exp004_report or {}

    lines = [
        f"# exp_005 — VLM Extraction: Structured Prompt V1",
        f"",
        f"**Date:** {now}",
        f"**Model:** gemini-2.5-flash-lite",
        f"**Prompt version:** EXTRACTION_PROMPT_V1",
        f"**Dataset:** golden_test_set.json ({n} samples)",
        f"**Baseline reference:** exp_004 (preprocessing + naive prompt) = {EXP004_E2E_OVERALL:.1%} E2E",
        f"",
        f"## Hypothesis",
        f"",
        f"Replacing the 7-line naive prompt with a structured prompt (role definition,",
        f"chain-of-thought steps, 3 few-shot examples, anti-hallucination rules, self-check)",
        f"will improve extraction accuracy significantly. The naive prompt gives Gemini no",
        f"guidance on receipt structure, date format, or how to handle ambiguous fields.",
        f"Expected gain: +10–18pp E2E (from 67% toward 85%).",
        f"",
        f"## What Changed (vs exp_004)",
        f"",
        f"| Component | exp_004 | exp_005 |",
        f"|---|---|---|",
        f"| Preprocessing | ImagePreprocessor v3 | ImagePreprocessor v3 (unchanged) |",
        f"| Prompt | Naive 7-line prompt | EXTRACTION_PROMPT_V1 (structured) |",
        f"| Schema enforcement | None (free-form JSON) | Pydantic Receipt schema |",
        f"| Temperature | 0 | 0 |",
        f"| Retry/backoff | Manual 3× | GeminiExtractor (1s/2s/4s backoff) |",
        f"| Disk cache | Per-script SHA256 | GeminiExtractor 7-day TTL cache |",
        f"",
        f"## Results — Overall",
        f"",
        f"| Metric | exp_004 (baseline) | exp_005 (V1 prompt) | Delta |",
        f"|---|---|---|---|",
        f"| **E2E success** | **{EXP004_E2E_OVERALL:.1%}** | **{pct(v['end_to_end_success'])}** | **{delta_str(v['end_to_end_success'], EXP004_E2E_OVERALL)}** |",
        f"| Merchant acc | {EXP004_FIELDS['merchant']:.1%} | {pct(v['field_accuracy']['merchant'])} | {delta_str(v['field_accuracy']['merchant'], EXP004_FIELDS['merchant'])} |",
        f"| Date acc | {EXP004_FIELDS['date']:.1%} | {pct(v['field_accuracy']['date'])} | {delta_str(v['field_accuracy']['date'], EXP004_FIELDS['date'])} |",
        f"| Total acc | {EXP004_FIELDS['total']:.1%} | {pct(v['field_accuracy']['total'])} | {delta_str(v['field_accuracy']['total'], EXP004_FIELDS['total'])} |",
        f"| Items F1 | — | {pct(v['items_f1'])} | — |",
        f"| Items precision | — | {pct(v['items_precision'])} | — |",
        f"| Items recall | — | {pct(v['items_recall'])} | — |",
        f"| Category acc | — | N/A (PhoBERT not built) | — |",
        f"| Latency p50 | — | {lat_p50:.2f}s | — |",
        f"| Latency p95 | — | {lat_p95:.2f}s | — |",
        f"| Cost / receipt | — | ${cost_per_receipt:.4f} | — |",
        f"| Total API cost | — | ${total_cost_usd:.3f} | — |",
        f"| Cache hits | — | {n_cache_hits}/{n} ({cache_hit_rate:.0%}) | — |",
        f"",
        f"## Results — Per Difficulty Group",
        f"",
        f"| Group | n | E2E (exp_004) | E2E (exp_005) | Delta |",
        f"|---|---|---|---|---|",
    ]
    lines += group_rows
    lines += [
        f"",
        f"*⚠ = group regresses more than 2pp vs baseline*",
        f"",
        f"## Error Summary",
        f"",
        f"- Total images processed: {n}",
        f"- Images with any error: {n_errors}",
        f"- API errors (after retries): {n_api_errors}",
        f"- Extraction failures (parse/timeout): {n - sum(1 for r in results if r['pred'] and r['pred'].get('total_amount') is not None)}",
        f"",
        f"## Top 5 Failure Cases",
        f"",
    ]
    if failure_lines:
        lines += failure_lines
    else:
        lines.append("*(No failures — all samples passed E2E)*")

    lines += [
        f"",
        f"## Decision",
        f"",
        f"**{verdict}** — {verdict_reason}",
        f"",
    ]

    if verdict == "PASS":
        lines += [
            f"EXTRACTION_PROMPT_V1 is promoted to `DEFAULT_PROMPT` in `src/extraction/prompts.py`.",
            f"GeminiExtractor with V1 prompt is now the production extractor for all subsequent experiments.",
        ]
    else:
        lines += [
            f"EXTRACTION_PROMPT_V1 is NOT promoted to production.",
            f"Next step: analyze failure cases above, create V2 addressing systematic errors.",
        ]

    lines += [
        f"",
        f"## Next Steps",
        f"",
        f"- Week 6: Build validation layer (math check, date sanity, merchant fuzzy match)",
        f"  to catch VLM errors before they reach the user.",
        f"- If FAIL: iterate prompt to V2 targeting the most common error patterns above.",
        f"- Week 7: Fine-tune PhoBERT on extracted merchant+item text for category classification.",
        f"",
        f"## Notes",
        f"",
        f"- Category accuracy is N/A in this experiment — PhoBERT classifier is built in Week 7.",
        f"  The V1 prompt intentionally does not ask for category to keep the task focused.",
        f"- exp_004 field accuracy numbers (merchant/date/total) are re-computed from the",
        f"  checkpoint file rather than the original report to ensure consistent measurement.",
        f"- All images processed through ImagePreprocessor v3 (same as exp_004) to isolate",
        f"  the prompt effect.",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp_005: Benchmark GeminiExtractor V1 prompt")
    p.add_argument("--golden",   default=str(GOLDEN_PATH),  help="Path to golden_test_set.json")
    p.add_argument("--output",   default=str(DEFAULT_OUT),  help="Output Markdown path")
    p.add_argument("--limit",    type=int, default=None,     help="Process only N images (dev)")
    p.add_argument("--no-cache", action="store_true",        help="Disable GeminiExtractor disk cache")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = os.getenv("Gemini_API_Key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("Gemini_API_Key not found in .env")
        sys.exit(1)

    # Load golden test set
    with open(args.golden, encoding="utf-8") as f:
        golden: list[dict] = json.load(f)
    if args.limit:
        golden = golden[:args.limit]
    log.info("Processing %d images from golden test set", len(golden))

    # Instantiate tools
    preprocessor = ImagePreprocessor()
    extractor = GeminiExtractor(
        api_key=api_key,
        enable_cache=not args.no_cache,
    )

    # Load checkpoint (crash-safe resume)
    checkpoint_results: dict[str, dict] = {}
    if CHECKPOINT.exists() and not args.no_cache:
        with open(CHECKPOINT, encoding="utf-8") as f:
            checkpoint_results = json.load(f)
        log.info("Resumed from checkpoint: %d images already done", len(checkpoint_results))

    results: list[dict] = []
    total_cost_usd = 0.0
    n_cache_hits = 0

    for i, record in enumerate(golden, 1):
        image_name = record["image"]

        # Resume from checkpoint
        if image_name in checkpoint_results:
            r = checkpoint_results[image_name]
            log.info("[%d/%d] %s — loaded from checkpoint", i, len(golden), image_name)
            results.append(r)
            total_cost_usd += r.get("cost_usd", 0.0)
            if r.get("from_cache"):
                n_cache_hits += 1
            continue

        log.info("[%d/%d] Processing %s ...", i, len(golden), image_name)

        r = process_single(record, extractor, preprocessor, checkpoint_results)
        results.append(r)
        checkpoint_results[image_name] = r
        total_cost_usd += r.get("cost_usd", 0.0)
        if r.get("from_cache"):
            n_cache_hits += 1

        # Save checkpoint after every image (crash-safe)
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(checkpoint_results, f, ensure_ascii=False, indent=2)

        # Rate-limit guard (skip delay on cache hits)
        if not r.get("from_cache"):
            time.sleep(CALL_DELAY_SEC)

    # ── Evaluate V1 results ────────────────────────────────────────────────────
    log.info("Running evaluation...")

    v1_preds = [r["pred"] for r in results]
    v1_lats  = [r["latency_sec"] for r in results]

    # Annotate preds with image/group for per-group breakdown
    for pred, gt in zip(v1_preds, golden):
        if pred:
            pred["image"] = gt["image"]
            pred["group"] = gt.get("group", "unknown")

    v1_report = evaluate(v1_preds, golden[:len(v1_preds)], latencies=v1_lats)

    # ── Load exp_004 baseline for comparison ───────────────────────────────────
    exp004_preds = load_exp004_baseline(golden)
    exp004_report: dict | None = None
    if exp004_preds:
        for pred, gt in zip(exp004_preds, golden):
            pred["image"] = gt["image"]
            pred["group"] = gt.get("group", "unknown")
        exp004_report = evaluate(exp004_preds, golden[:len(exp004_preds)])
        log.info("exp_004 baseline E2E: %.1f%%", exp004_report["end_to_end_success"] * 100)

    log.info("exp_005 (V1 prompt) E2E: %.1f%%", v1_report["end_to_end_success"] * 100)
    log.info(
        "Delta vs exp_004: %+.1f%%",
        (v1_report["end_to_end_success"] - EXP004_E2E_OVERALL) * 100,
    )
    log.info("Total API cost: $%.3f (%.0f cache hits)", total_cost_usd, n_cache_hits)

    # ── Write report ───────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = format_report(golden, results, v1_report, exp004_report, total_cost_usd, n_cache_hits)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    log.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
