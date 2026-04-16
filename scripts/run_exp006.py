"""
scripts/run_exp006.py
---------------------
exp_006: Benchmark EXTRACTION_PROMPT_V2 vs exp_005 baseline (V1 prompt).

Experiment design:
  - Baseline: exp_005 results — ImagePreprocessor + EXTRACTION_PROMPT_V1 = 68.0% E2E
  - Test:     ImagePreprocessor + GeminiExtractor (EXTRACTION_PROMPT_V2)

Key changes being measured (V2 vs V1):
  1. Step 1: 5-way receipt type classification with explicit "handwritten" detection
  2. Step 3: SUPERMARKET MULTI-DATE RULE — use transaction date, not invoice print date
  3. Step 4: HANDWRITTEN RULE — best-effort reading instead of strict null
  4. Step 5: SUPERMARKET MULTI-TOTAL RULE — use grand total, ignore category subtotals
  5. Hard rules: two-tier null policy (printed strict, handwritten lenient)

Hypotheses:
  H1 (primary): 04_handwritten recovers from -36.3pp regression (27.3% → 55%+)
  H2 (secondary): 03_long_invoice improves (68.8% → 75%+) via date/total fixes

Crash safety:
    Results saved to data/exp006_checkpoint.json after each image.
    Re-run to resume from where it stopped.

Usage:
    python scripts/run_exp006.py               # full run
    python scripts/run_exp006.py --limit 10    # quick dev test (10 images)
    python scripts/run_exp006.py --no-cache    # force re-run (ignore disk cache)
    python scripts/run_exp006.py --clear-groups 04_handwritten 03_long_invoice
                                               # re-run specific groups (clears their checkpoint)
"""

from __future__ import annotations

import argparse
import json
import logging
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

GOLDEN_PATH  = PROJECT_ROOT / "data" / "golden_test_set.json"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
CHECKPOINT   = PROJECT_ROOT / "data" / "exp006_checkpoint.json"
DEFAULT_OUT  = PROJECT_ROOT / "docs" / "experiments" / "exp_006_prompt_v2.md"

# Baseline numbers from exp_005 (V1 prompt)
EXP005_E2E_OVERALL = 0.680
EXP005_PER_GROUP = {
    "01_pos_clean":    0.731,   # +11.6pp vs exp_004
    "02_vat_invoice":  0.700,   # +5.6pp vs exp_004
    "03_long_invoice": 0.688,   # -6.2pp vs exp_004 ⚠ (target: fix to ≥0.750)
    "04_handwritten":  0.273,   # -36.3pp vs exp_004 ⚠ (target: fix to ≥0.550)
    "05_low_quality":  0.767,   # +3.4pp vs exp_004
}
EXP005_FIELDS = {
    "merchant": 0.850,
    "date":     0.840,
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


# ── Per-image processing ───────────────────────────────────────────────────────

def process_single(
    record: dict,
    extractor: GeminiExtractor,
    preprocessor: ImagePreprocessor,
) -> dict:
    """
    Run ImagePreprocessor → GeminiExtractor (V2 prompt) on one golden set record.
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
            image_input = str(img_path)   # fallback to raw
        else:
            image_input = proc.image   # numpy ndarray (BGR)
    except Exception as exc:
        log.error("Preprocessing exception for %s: %s", image_name, exc)
        result["errors"].append(f"preprocessing_exception: {exc}")
        image_input = str(img_path)

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
    v2_report: dict,
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

    # ── Pass/fail determination ────────────────────────────────────────────────
    e2e_new = v2_report["end_to_end_success"]
    e2e_old = EXP005_E2E_OVERALL
    e2e_delta = e2e_new - e2e_old

    group_regression = False
    regressed_groups = []
    for grp, stats in v2_report.get("per_group", {}).items():
        old_grp = EXP005_PER_GROUP.get(grp)
        if old_grp is not None:
            if stats["end_to_end_success"] - old_grp < -0.02:
                group_regression = True
                regressed_groups.append(grp)

    lat_ok = lat_p95 <= 5.0

    if e2e_delta >= 0.01 and not group_regression and lat_ok:
        verdict = "PASS"
        verdict_reason = (
            f"E2E improves by {e2e_delta:+.1%} (≥ +1pp), "
            f"no group regresses >2pp, p95 latency = {lat_p95:.2f}s ≤ 5s."
        )
    elif e2e_delta < 0.01:
        verdict = "FAIL"
        verdict_reason = (
            f"E2E delta {e2e_delta:+.1%} is below the +1pp threshold. "
            f"V2 prompt changes did not produce sufficient overall improvement."
        )
    elif group_regression:
        verdict = "FAIL (regression)"
        verdict_reason = (
            f"E2E delta {e2e_delta:+.1%} but groups regress >2pp: "
            f"{', '.join(regressed_groups)}."
        )
    else:
        verdict = "FAIL (latency)"
        verdict_reason = f"p95 latency {lat_p95:.2f}s exceeds 5s threshold."

    # ── Per-group table ────────────────────────────────────────────────────────
    group_rows = []
    for grp in sorted(EXP005_PER_GROUP.keys()):
        grp_stats = v2_report.get("per_group", {}).get(grp)
        old_e2e = EXP005_PER_GROUP.get(grp, 0.0)
        if grp_stats:
            new_e2e = grp_stats["end_to_end_success"]
            flag = " ⚠" if new_e2e - old_e2e < -0.02 else (" ✓" if new_e2e - old_e2e >= 0.05 else "")
            group_rows.append(
                f"| {grp} | {grp_stats['n']} "
                f"| {old_e2e:.1%} "
                f"| {new_e2e:.1%} "
                f"| {delta_str(new_e2e, old_e2e)}{flag} |"
            )

    # ── Hypothesis verification ────────────────────────────────────────────────
    hw_grp = v2_report.get("per_group", {}).get("04_handwritten", {})
    li_grp = v2_report.get("per_group", {}).get("03_long_invoice", {})
    hw_new = hw_grp.get("end_to_end_success", 0.0)
    li_new = li_grp.get("end_to_end_success", 0.0)

    h1_result = (
        f"H1 (handwritten recovery): {EXP005_PER_GROUP['04_handwritten']:.1%} → {hw_new:.1%} "
        f"({delta_str(hw_new, EXP005_PER_GROUP['04_handwritten'])}) — "
        f"{'CONFIRMED ✓' if hw_new >= 0.55 else ('PARTIAL' if hw_new > EXP005_PER_GROUP['04_handwritten'] else 'FAILED ✗')}"
    )
    h2_result = (
        f"H2 (long_invoice date/total fix): {EXP005_PER_GROUP['03_long_invoice']:.1%} → {li_new:.1%} "
        f"({delta_str(li_new, EXP005_PER_GROUP['03_long_invoice'])}) — "
        f"{'CONFIRMED ✓' if li_new >= 0.75 else ('PARTIAL' if li_new > EXP005_PER_GROUP['03_long_invoice'] else 'FAILED ✗')}"
    )

    # ── Failure summary ────────────────────────────────────────────────────────
    failures = get_failures(
        [r["pred"] for r in results],
        golden[:n],
    )
    failure_lines = []
    for fail in failures[:8]:
        failure_lines.append(
            f"- **{fail['image']}** ({fail['group']}): "
            f"merchant={'OK' if fail['merchant_ok'] else 'FAIL'}, "
            f"date={'OK' if fail['date_ok'] else 'FAIL'}, "
            f"total={'OK' if fail['total_ok'] else 'FAIL'}, "
            f"items_f1={fail['items_f1']:.2f}"
        )

    v = v2_report

    lines = [
        f"# exp_006 — Extraction Prompt V2: Handwritten + Long Invoice Fixes",
        f"",
        f"**Date:** {now}",
        f"**Model:** gemini-2.5-flash-lite",
        f"**Prompt version:** EXTRACTION_PROMPT_V2",
        f"**Dataset:** golden_test_set.json ({n} samples)",
        f"**Baseline reference:** exp_005 (EXTRACTION_PROMPT_V1) = {EXP005_E2E_OVERALL:.1%} E2E",
        f"",
        f"## Hypotheses",
        f"",
        f"Two systematic failures were identified in exp_005 and targeted in V2:",
        f"",
        f"**H1 — Handwritten regression** (04_handwritten: 63.6% → 27.3%, -36.3pp)",
        f"Root cause: V1 anti-hallucination rules ('return null if unclear') caused Gemini",
        f"to return null for ALL uncertain handwritten fields — correct for printed receipts,",
        f"but catastrophic for handwritten where uncertainty is expected.",
        f"V2 fix: Two-tier null policy. Handwritten receipts get 'best-effort' instructions:",
        f"'read every item even if partially legible', 'provide best reading rather than null'.",
        f"Expected: 04_handwritten recovers to ≥55%.",
        f"",
        f"**H2 — Long invoice date/total failures** (03_long_invoice: 75.0% → 68.8%, -6.2pp)",
        f"Root cause: Supermarket receipts (BigC, Lotte, Co.opmart) print multiple dates",
        f"(invoice date vs transaction date) and multiple subtotals (per category).",
        f"V1 prompt gave no guidance, so Gemini picked the wrong date/total.",
        f"V2 fix: Explicit SUPERMARKET MULTI-DATE RULE and SUPERMARKET MULTI-TOTAL RULE",
        f"with named labels to look for ('Ngày mua', 'Tổng thanh toán') and chain store list.",
        f"Expected: 03_long_invoice recovers to ≥75%.",
        f"",
        f"## What Changed (V1 → V2)",
        f"",
        f"| Component | V1 | V2 |",
        f"|---|---|---|",
        f"| Preprocessing | ImagePreprocessor v3 | ImagePreprocessor v3 (unchanged) |",
        f"| Receipt classification (Step 1) | Binary (receipt / not) | 5-way type classification |",
        f"| Date extraction (Step 3) | Single date | + SUPERMARKET MULTI-DATE RULE |",
        f"| Item extraction (Step 4) | Strict null | + HANDWRITTEN RULE (best-effort) |",
        f"| Total extraction (Step 5) | Single total | + SUPERMARKET MULTI-TOTAL RULE |",
        f"| Few-shot Example 2 | VAT invoice | Supermarket multi-total (Lotte Mart) |",
        f"| Few-shot Example 3 | Handwritten with nulls | Handwritten with best-effort items |",
        f"| Hard rules | Single null policy | Two-tier: printed strict / handwritten lenient |",
        f"",
        f"## Results — Overall",
        f"",
        f"| Metric | exp_005 (V1 baseline) | exp_006 (V2 prompt) | Delta |",
        f"|---|---|---|---|",
        f"| **E2E success** | **{EXP005_E2E_OVERALL:.1%}** | **{pct(v['end_to_end_success'])}** | **{delta_str(v['end_to_end_success'], EXP005_E2E_OVERALL)}** |",
        f"| Merchant acc | {EXP005_FIELDS['merchant']:.1%} | {pct(v['field_accuracy']['merchant'])} | {delta_str(v['field_accuracy']['merchant'], EXP005_FIELDS['merchant'])} |",
        f"| Date acc | {EXP005_FIELDS['date']:.1%} | {pct(v['field_accuracy']['date'])} | {delta_str(v['field_accuracy']['date'], EXP005_FIELDS['date'])} |",
        f"| Total acc | {EXP005_FIELDS['total']:.1%} | {pct(v['field_accuracy']['total'])} | {delta_str(v['field_accuracy']['total'], EXP005_FIELDS['total'])} |",
        f"| Items F1 | — | {pct(v['items_f1'])} | — |",
        f"| Items precision | — | {pct(v['items_precision'])} | — |",
        f"| Items recall | — | {pct(v['items_recall'])} | — |",
        f"| Latency p50 | — | {lat_p50:.2f}s | — |",
        f"| Latency p95 | — | {lat_p95:.2f}s | — |",
        f"| Cost / receipt | — | ${cost_per_receipt:.4f} | — |",
        f"| Total API cost | — | ${total_cost_usd:.3f} | — |",
        f"| Cache hits | — | {n_cache_hits}/{n} ({cache_hit_rate:.0%}) | — |",
        f"",
        f"## Results — Per Difficulty Group",
        f"",
        f"| Group | n | E2E (V1, exp_005) | E2E (V2, exp_006) | Delta |",
        f"|---|---|---|---|---|",
    ]
    lines += group_rows
    lines += [
        f"",
        f"*⚠ = group regresses more than 2pp  |  ✓ = group improves by ≥5pp*",
        f"",
        f"## Hypothesis Verification",
        f"",
        f"- {h1_result}",
        f"- {h2_result}",
        f"",
        f"## Error Summary",
        f"",
        f"- Total images processed: {n}",
        f"- Images with any error: {n_errors}",
        f"- API errors (after retries): {n_api_errors}",
        f"- Extraction failures: {n - sum(1 for r in results if r['pred'] and r['pred'].get('total_amount') is not None)}",
        f"",
        f"## Top Failure Cases",
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
            f"EXTRACTION_PROMPT_V2 is promoted to `DEFAULT_PROMPT` in `src/extraction/prompts.py`.",
            f"V2 is now the production extractor for all subsequent experiments.",
            f"",
            f"## Next Steps",
            f"",
            f"- Week 7: Fine-tune PhoBERT classifier (`src/classification/`)",
            f"  - Base model: `vinai/phobert-base`",
            f"  - Input: `merchant_name | item_names[:10]`",
            f"  - 10 expense categories (Ăn uống, Đi lại, Mua sắm...)",
            f"- Week 8: Pipeline orchestrator + FastAPI backend",
        ]
    else:
        lines += [
            f"EXTRACTION_PROMPT_V2 is NOT promoted. Analysis of remaining failures required.",
            f"",
            f"## Root Cause Analysis",
            f"",
        ]
        if hw_new < 0.55:
            lines += [
                f"### 04_handwritten still underperforming ({hw_new:.1%})",
                f"",
                f"Possible causes:",
                f"- Gemini still defaults to null for very low-confidence handwritten items",
                f"- OCR noise too high for model to produce meaningful text",
                f"- Few-shot example not representative enough of actual handwritten samples",
                f"",
                f"Suggested V3 fix: Add explicit instruction 'Write the characters you see,",
                f"even if you are only 30% confident — use [?] for unreadable single characters'.",
                f"",
            ]
        if li_new < 0.75:
            lines += [
                f"### 03_long_invoice still underperforming ({li_new:.1%})",
                f"",
                f"Possible causes:",
                f"- Multi-date / multi-total rule not specific enough for the failing images",
                f"- Some long invoices fail due to item extraction (sliding window needed?)",
                f"- Date format variant not covered (e.g., non-standard date position)",
                f"",
                f"Suggested V3 fix: Review the actual failing images, check if date/total",
                f"fields are the remaining blockers or if item extraction is now the limit.",
                f"",
            ]
        lines += [
            f"## Next Steps",
            f"",
            f"1. Manually inspect the top failure cases above",
            f"2. Identify whether failures are: date format, total selection, or item hallucination",
            f"3. Build EXTRACTION_PROMPT_V3 targeting remaining systematic errors",
            f"4. Re-run exp_007 (prompt V3 benchmark)",
        ]

    lines += [
        f"",
        f"## Notes",
        f"",
        f"- Category accuracy is N/A in this experiment — PhoBERT classifier is built in Week 7.",
        f"- V2 prompt is ~20% longer than V1 (~840 tokens vs ~700), reflected in cost.",
        f"- exp_005 handwritten subgroup numbers (27.3%) include the ReceiptItem.total=null",
        f"  bug fix from mid-exp_005 run. All exp_006 numbers use the fixed schema.",
        f"- All images processed through ImagePreprocessor v3 (unchanged since exp_004).",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp_006: Benchmark EXTRACTION_PROMPT_V2")
    p.add_argument("--golden",         default=str(GOLDEN_PATH), help="Path to golden_test_set.json")
    p.add_argument("--output",         default=str(DEFAULT_OUT), help="Output Markdown path")
    p.add_argument("--limit",          type=int, default=None,   help="Process only N images (dev)")
    p.add_argument("--no-cache",       action="store_true",      help="Disable GeminiExtractor disk cache")
    p.add_argument(
        "--clear-groups",
        nargs="+",
        metavar="GROUP",
        default=[],
        help="Clear checkpoint entries for these groups before running (forces fresh API calls)",
    )
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

    # Clear specific groups (forces fresh API calls for those groups)
    if args.clear_groups:
        cleared = 0
        for img_name in list(checkpoint_results.keys()):
            entry = checkpoint_results[img_name]
            if entry.get("group") in args.clear_groups:
                del checkpoint_results[img_name]
                cleared += 1
        if cleared:
            log.info("Cleared %d checkpoint entries for groups: %s", cleared, args.clear_groups)

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

        r = process_single(record, extractor, preprocessor)
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

    # ── Evaluate V2 results ────────────────────────────────────────────────────
    log.info("Running evaluation...")

    v2_preds = [r["pred"] for r in results]
    v2_lats  = [r["latency_sec"] for r in results]

    for pred, gt in zip(v2_preds, golden):
        if pred:
            pred["image"] = gt["image"]
            pred["group"] = gt.get("group", "unknown")

    v2_report = evaluate(v2_preds, golden[:len(v2_preds)], latencies=v2_lats)

    log.info("exp_006 (V2 prompt) E2E: %.1f%%", v2_report["end_to_end_success"] * 100)
    log.info(
        "Delta vs exp_005 (V1): %+.1f%%",
        (v2_report["end_to_end_success"] - EXP005_E2E_OVERALL) * 100,
    )

    # Per-group summary
    for grp in sorted(EXP005_PER_GROUP.keys()):
        grp_stats = v2_report.get("per_group", {}).get(grp, {})
        new_e2e = grp_stats.get("end_to_end_success", 0.0)
        old_e2e = EXP005_PER_GROUP.get(grp, 0.0)
        log.info(
            "  %s: %.1f%% → %.1f%% (%+.1f%%)",
            grp, old_e2e * 100, new_e2e * 100, (new_e2e - old_e2e) * 100,
        )

    log.info("Total API cost: $%.3f (%.0f cache hits)", total_cost_usd, n_cache_hits)

    # ── Write report ───────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = format_report(golden, results, v2_report, total_cost_usd, n_cache_hits)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    log.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
