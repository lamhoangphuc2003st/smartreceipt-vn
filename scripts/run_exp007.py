"""
scripts/run_exp007.py
---------------------
exp_007: Benchmark EXTRACTION_PROMPT_V3 + structured output vs exp_006 (V2 prompt).

Experiment design:
  - Baseline: exp_006 — ImagePreprocessor + EXTRACTION_PROMPT_V2 = 68.0% E2E
  - Test:     ImagePreprocessor + GeminiExtractor (EXTRACTION_PROMPT_V3)
               + response_schema=Receipt (Gemini native structured output)

Two simultaneous changes vs exp_006:
  A) response_schema=Receipt + response_mime_type="application/json" in API call
     → Gemini enforces schema server-side → no more code-fence stripping heuristics
     → parse errors go to 0 (was 4 in exp_006)
  B) EXTRACTION_PROMPT_V3 prompt changes:
     1. Removed "Return JSON only" (handled by API)
     2. Restored VAT invoice as Example 2 (fixes -11.2pp vat_invoice regression)
     3. MULTI-DATE/MULTI-TOTAL rules now explicitly scoped to named supermarket chains
     4. Handwritten: [?] placeholder, explicit "never empty items list" instruction

Hypotheses:
  H1: 02_vat_invoice recovers from 58.8% → ≥70% (fix example ordering regression)
  H2: 04_handwritten ≥ 27.3% (maintain or improve via [?] placeholder)
  H3: 03_long_invoice ≥ 68.8% (multi-date/total scoping prevents false triggers)
  H4: parse_errors = 0 (structured output guarantee)

Usage:
    python scripts/run_exp007.py               # full run
    python scripts/run_exp007.py --limit 10    # quick dev test
    python scripts/run_exp007.py --no-cache    # force re-run
    python scripts/run_exp007.py --clear-groups 02_vat_invoice 04_handwritten
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
CHECKPOINT   = PROJECT_ROOT / "data" / "exp007_checkpoint.json"
DEFAULT_OUT  = PROJECT_ROOT / "docs" / "experiments" / "exp_007_prompt_v3.md"

# Baseline numbers from exp_006 (V2 prompt, no structured output)
EXP006_E2E_OVERALL = 0.680
EXP006_PER_GROUP = {
    "01_pos_clean":    0.769,
    "02_vat_invoice":  0.588,   # regressed in exp_006 ⚠ — target: recover to ≥0.700
    "03_long_invoice": 0.688,   # stuck at this level since exp_005
    "04_handwritten":  0.273,   # catastrophic since exp_005
    "05_low_quality":  0.800,
}
EXP006_FIELDS = {
    "merchant": 0.860,
    "date":     0.830,
    "total":    0.900,
}
EXP006_PARSE_ERRORS = 4  # extraction failures due to JSON parse errors

# Rate-limit guard (~15 RPM free tier)
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
    base = {
        "image": gt["image"],
        "group": gt.get("group", "unknown"),
        "category": None,
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

    try:
        proc = preprocessor.process(str(img_path))
        result["preprocessing_ms"] = proc.processing_time_ms
        image_input = proc.image if (proc.success and proc.image is not None) else str(img_path)
        if not proc.success:
            result["errors"].append("preprocessing_failed")
    except Exception as exc:
        log.error("Preprocessing exception for %s: %s", image_name, exc)
        result["errors"].append(f"preprocessing_exception: {exc}")
        image_input = str(img_path)

    try:
        extraction: ExtractionResult = extractor.extract(image_input)
        result["latency_sec"] = extraction.latency_ms / 1000.0
        result["cost_usd"]    = extraction.cost_usd
        result["from_cache"]  = extraction.from_cache
        result["attempts"]    = extraction.attempts

        if not extraction.success:
            log.warning(
                "Extraction failed for %s: [%s] %s",
                image_name, extraction.error_type, extraction.error_message,
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


# ── Report ─────────────────────────────────────────────────────────────────────

def format_report(
    golden: list[dict],
    results: list[dict],
    v3_report: dict,
    total_cost_usd: float,
    n_cache_hits: int,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(results)
    n_errors = sum(1 for r in results if r.get("errors"))
    n_api_errors = sum(1 for r in results if any("extraction_" in e for e in r.get("errors", [])))
    n_parse_errors = sum(1 for r in results if any("parse" in e for e in r.get("errors", [])))
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

    e2e_new = v3_report["end_to_end_success"]
    e2e_delta = e2e_new - EXP006_E2E_OVERALL

    group_regression = False
    regressed_groups: list[str] = []
    for grp, stats in v3_report.get("per_group", {}).items():
        old_grp = EXP006_PER_GROUP.get(grp)
        if old_grp is not None and stats["end_to_end_success"] - old_grp < -0.02:
            group_regression = True
            regressed_groups.append(grp)

    lat_ok = lat_p95 <= 5.0

    if e2e_delta >= 0.01 and not group_regression and lat_ok:
        verdict = "PASS"
        verdict_reason = (
            f"E2E improves by {e2e_delta:+.1%} (≥ +1pp), "
            f"no group regresses >2pp, p95 = {lat_p95:.2f}s ≤ 5s."
        )
    elif e2e_delta < 0.01:
        verdict = "FAIL"
        verdict_reason = f"E2E delta {e2e_delta:+.1%} < +1pp threshold."
    elif group_regression:
        verdict = "FAIL (regression)"
        verdict_reason = f"E2E {e2e_delta:+.1%} but groups regress: {', '.join(regressed_groups)}."
    else:
        verdict = "FAIL (latency)"
        verdict_reason = f"p95 latency {lat_p95:.2f}s > 5s."

    # Per-group table
    group_rows = []
    for grp in sorted(EXP006_PER_GROUP.keys()):
        grp_stats = v3_report.get("per_group", {}).get(grp)
        old_e2e = EXP006_PER_GROUP.get(grp, 0.0)
        if grp_stats:
            new_e2e = grp_stats["end_to_end_success"]
            flag = " ⚠" if new_e2e - old_e2e < -0.02 else (" ✓" if new_e2e - old_e2e >= 0.05 else "")
            group_rows.append(
                f"| {grp} | {grp_stats['n']} "
                f"| {old_e2e:.1%} "
                f"| {new_e2e:.1%} "
                f"| {delta_str(new_e2e, old_e2e)}{flag} |"
            )

    # Hypothesis results
    def h_result(grp: str, target: float, label: str) -> str:
        stats = v3_report.get("per_group", {}).get(grp, {})
        new_e2e = stats.get("end_to_end_success", 0.0)
        old_e2e = EXP006_PER_GROUP.get(grp, 0.0)
        status = "CONFIRMED ✓" if new_e2e >= target else ("PARTIAL" if new_e2e > old_e2e else "FAILED ✗")
        return (
            f"{label}: {old_e2e:.1%} → {new_e2e:.1%} "
            f"({delta_str(new_e2e, old_e2e)}) — {status}"
        )

    # Failure summary
    failures = get_failures([r["pred"] for r in results], golden[:n])
    failure_lines = [
        f"- **{f['image']}** ({f['group']}): "
        f"merchant={'OK' if f['merchant_ok'] else 'FAIL'}, "
        f"date={'OK' if f['date_ok'] else 'FAIL'}, "
        f"total={'OK' if f['total_ok'] else 'FAIL'}, "
        f"items_f1={f['items_f1']:.2f}"
        for f in failures[:8]
    ]

    v = v3_report
    lines = [
        f"# exp_007 — Prompt V3 + Structured Output (response_schema)",
        f"",
        f"**Date:** {now}",
        f"**Model:** gemini-2.5-flash-lite",
        f"**Prompt version:** EXTRACTION_PROMPT_V3",
        f"**Structured output:** response_schema=Receipt, response_mime_type=application/json",
        f"**Dataset:** golden_test_set.json ({n} samples)",
        f"**Baseline:** exp_006 (EXTRACTION_PROMPT_V2, free-form JSON) = {EXP006_E2E_OVERALL:.1%} E2E",
        f"",
        f"## What Changed (V2 → V3)",
        f"",
        f"| Component | exp_006 (V2) | exp_007 (V3) |",
        f"|---|---|---|",
        f"| Output enforcement | Prompt: 'Return JSON only' | API: response_schema=Receipt |",
        f"| JSON parsing | Code-fence strip + brace-find heuristics | json.loads() direct |",
        f"| Few-shot Example 2 | Supermarket (Lotte Mart) | VAT invoice (CÔNG TY ABC) ← restored |",
        f"| Few-shot Example 3 | Handwritten (partial) | Supermarket multi-total |",
        f"| MULTI-DATE RULE scope | All receipts | Named supermarket chains only |",
        f"| MULTI-TOTAL RULE scope | All receipts | Named supermarket chains only |",
        f"| Handwritten instruction | Best-effort reading | [?] placeholder + 'never empty items' |",
        f"",
        f"## Hypotheses",
        f"",
        f"- H1: 02_vat_invoice recovers from 58.8% → ≥70% (example 2 restored as VAT invoice)",
        f"- H2: 04_handwritten ≥ 27.3% (maintain or improve via explicit [?] placeholder)",
        f"- H3: 03_long_invoice ≥ 68.8% (chain-scoped rules prevent false triggers on other types)",
        f"- H4: parse_errors = 0 (structured output guarantee vs {EXP006_PARSE_ERRORS} in exp_006)",
        f"",
        f"## Results — Overall",
        f"",
        f"| Metric | exp_006 (V2) | exp_007 (V3) | Delta |",
        f"|---|---|---|---|",
        f"| **E2E success** | **{EXP006_E2E_OVERALL:.1%}** | **{pct(v['end_to_end_success'])}** | **{delta_str(v['end_to_end_success'], EXP006_E2E_OVERALL)}** |",
        f"| Merchant acc | {EXP006_FIELDS['merchant']:.1%} | {pct(v['field_accuracy']['merchant'])} | {delta_str(v['field_accuracy']['merchant'], EXP006_FIELDS['merchant'])} |",
        f"| Date acc | {EXP006_FIELDS['date']:.1%} | {pct(v['field_accuracy']['date'])} | {delta_str(v['field_accuracy']['date'], EXP006_FIELDS['date'])} |",
        f"| Total acc | {EXP006_FIELDS['total']:.1%} | {pct(v['field_accuracy']['total'])} | {delta_str(v['field_accuracy']['total'], EXP006_FIELDS['total'])} |",
        f"| Items F1 | 84.3% | {pct(v['items_f1'])} | {delta_str(v['items_f1'], 0.843)} |",
        f"| Items precision | 88.6% | {pct(v['items_precision'])} | {delta_str(v['items_precision'], 0.886)} |",
        f"| Items recall | 84.5% | {pct(v['items_recall'])} | {delta_str(v['items_recall'], 0.845)} |",
        f"| Parse errors | {EXP006_PARSE_ERRORS} | {n_parse_errors} | {n_parse_errors - EXP006_PARSE_ERRORS:+d} |",
        f"| Latency p50 | 4.56s | {lat_p50:.2f}s | — |",
        f"| Latency p95 | 7.32s | {lat_p95:.2f}s | — |",
        f"| Cost / receipt | $0.0003 | ${cost_per_receipt:.4f} | — |",
        f"| Total API cost | — | ${total_cost_usd:.3f} | — |",
        f"| Cache hits | — | {n_cache_hits}/{n} ({cache_hit_rate:.0%}) | — |",
        f"",
        f"## Results — Per Difficulty Group",
        f"",
        f"| Group | n | E2E (V2, exp_006) | E2E (V3, exp_007) | Delta |",
        f"|---|---|---|---|---|",
    ]
    lines += group_rows
    lines += [
        f"",
        f"*⚠ = regresses >2pp  |  ✓ = improves ≥5pp*",
        f"",
        f"## Hypothesis Verification",
        f"",
        f"- H1 {h_result('02_vat_invoice', 0.700, '02_vat_invoice recovery')}",
        f"- H2 {h_result('04_handwritten', 0.273, '04_handwritten baseline hold')}",
        f"- H3 {h_result('03_long_invoice', 0.688, '03_long_invoice baseline hold')}",
        f"- H4: parse_errors = {n_parse_errors} (was {EXP006_PARSE_ERRORS}) — "
        f"{'CONFIRMED ✓' if n_parse_errors == 0 else f'PARTIAL ({n_parse_errors} remaining)'}",
        f"",
        f"## Error Summary",
        f"",
        f"- Total images: {n}",
        f"- Images with any error: {n_errors}",
        f"- API errors (after retries): {n_api_errors}",
        f"- Parse errors: {n_parse_errors} (structured output target: 0)",
        f"",
        f"## Top Failure Cases",
        f"",
    ]
    if failure_lines:
        lines += failure_lines
    else:
        lines.append("*(No failures)*")

    lines += [
        f"",
        f"## Decision",
        f"",
        f"**{verdict}** — {verdict_reason}",
        f"",
    ]

    if verdict == "PASS":
        lines += [
            f"EXTRACTION_PROMPT_V3 + structured output promoted to production.",
            f"`DEFAULT_PROMPT = EXTRACTION_PROMPT_V3` in `src/extraction/prompts.py`.",
            f"",
            f"## Next Steps",
            f"",
            f"- Week 7: Fine-tune PhoBERT expense classifier (`src/classification/`)",
            f"- Week 8: Pipeline orchestrator + FastAPI backend",
        ]
    else:
        lines += [
            f"EXTRACTION_PROMPT_V3 is NOT promoted. See root cause analysis below.",
            f"",
            f"## Root Cause Analysis",
            f"",
        ]
        vat_stats = v3_report.get("per_group", {}).get("02_vat_invoice", {})
        hw_stats  = v3_report.get("per_group", {}).get("04_handwritten", {})
        vat_e2e = vat_stats.get("end_to_end_success", 0.0)
        hw_e2e  = hw_stats.get("end_to_end_success", 0.0)

        if vat_e2e < 0.700:
            lines += [
                f"### 02_vat_invoice still underperforming ({vat_e2e:.1%})",
                f"",
                f"Expected VAT example restored in position 2 to fix -11.2pp regression.",
                f"If still failing: check if failures are merchant (OCR noise on company names)",
                f"or date (invoice date format differs from supermarket — YYYY/MM/DD etc.).",
                f"",
            ]
        if hw_e2e < 0.400:
            lines += [
                f"### 04_handwritten still underperforming ({hw_e2e:.1%})",
                f"",
                f"The [?] placeholder and 'never empty list' instruction may still be insufficient.",
                f"Consider: preprocessing enhancement specifically for handwritten (higher contrast,",
                f"binarization), or accept handwritten as a known limitation and document it.",
                f"",
            ]
        lines += [
            f"## Next Steps",
            f"",
            f"1. Inspect failing images for the most-regressed groups",
            f"2. Determine if failures are OCR-limited (image quality) or prompt-limited",
            f"3. If OCR-limited: apply targeted preprocessing (binarization for handwritten)",
            f"4. If prompt-limited: iterate to V4",
        ]

    lines += [
        f"",
        f"## Notes",
        f"",
        f"- Structured output (response_schema) was added alongside prompt V3.",
        f"  If parse errors drop to 0, the structured output change is confirmed beneficial",
        f"  regardless of overall E2E outcome.",
        f"- Category accuracy N/A — PhoBERT classifier is Week 7.",
        f"- All images processed through ImagePreprocessor v3 (unchanged).",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp_007: Benchmark EXTRACTION_PROMPT_V3 + structured output")
    p.add_argument("--golden",         default=str(GOLDEN_PATH))
    p.add_argument("--output",         default=str(DEFAULT_OUT))
    p.add_argument("--limit",          type=int, default=None)
    p.add_argument("--no-cache",       action="store_true")
    p.add_argument("--clear-groups",   nargs="+", metavar="GROUP", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = os.getenv("Gemini_API_Key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("Gemini_API_Key not found in .env")
        sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        golden: list[dict] = json.load(f)
    if args.limit:
        golden = golden[:args.limit]
    log.info("Processing %d images", len(golden))

    preprocessor = ImagePreprocessor()
    extractor = GeminiExtractor(api_key=api_key, enable_cache=not args.no_cache)

    checkpoint_results: dict[str, dict] = {}
    if CHECKPOINT.exists() and not args.no_cache:
        with open(CHECKPOINT, encoding="utf-8") as f:
            checkpoint_results = json.load(f)
        log.info("Resumed checkpoint: %d done", len(checkpoint_results))

    if args.clear_groups:
        cleared = sum(
            1 for k in list(checkpoint_results)
            if checkpoint_results.pop(k, {}).get("group") in args.clear_groups
        )
        log.info("Cleared %d entries for groups: %s", cleared, args.clear_groups)

    results: list[dict] = []
    total_cost_usd = 0.0
    n_cache_hits = 0

    for i, record in enumerate(golden, 1):
        image_name = record["image"]

        if image_name in checkpoint_results:
            r = checkpoint_results[image_name]
            log.info("[%d/%d] %s — checkpoint", i, len(golden), image_name)
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

        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(checkpoint_results, f, ensure_ascii=False, indent=2)

        if not r.get("from_cache"):
            time.sleep(CALL_DELAY_SEC)

    log.info("Evaluating...")
    v3_preds = [r["pred"] for r in results]
    for pred, gt in zip(v3_preds, golden):
        if pred:
            pred["image"] = gt["image"]
            pred["group"] = gt.get("group", "unknown")

    v3_report = evaluate(v3_preds, golden[:len(v3_preds)], latencies=[r["latency_sec"] for r in results])

    log.info("exp_007 (V3 + structured) E2E: %.1f%%", v3_report["end_to_end_success"] * 100)
    log.info("Delta vs exp_006: %+.1f%%", (v3_report["end_to_end_success"] - EXP006_E2E_OVERALL) * 100)
    for grp in sorted(EXP006_PER_GROUP.keys()):
        stats = v3_report.get("per_group", {}).get(grp, {})
        new_e2e = stats.get("end_to_end_success", 0.0)
        old_e2e = EXP006_PER_GROUP[grp]
        log.info("  %s: %.1f%% → %.1f%% (%+.1f%%)", grp, old_e2e*100, new_e2e*100, (new_e2e-old_e2e)*100)
    log.info("Cost: $%.3f (%d cache hits)", total_cost_usd, n_cache_hits)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = format_report(golden, results, v3_report, total_cost_usd, n_cache_hits)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    log.info("Report → %s", output_path)


if __name__ == "__main__":
    main()
