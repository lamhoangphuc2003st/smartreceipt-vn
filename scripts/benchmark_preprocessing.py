"""
scripts/benchmark_preprocessing.py
------------------------------------
Compare Gemini extraction accuracy: raw image vs preprocessed image.

Runs the naive baseline prompt on all 100 golden set images:
  - Once with the raw image (as in exp_001)
  - Once with the preprocessed image (through ImagePreprocessor)

Evaluates both sets with the same metrics framework, then writes
docs/experiments/exp_002_preprocessing.md with the comparison.

Usage:
    python scripts/benchmark_preprocessing.py
    python scripts/benchmark_preprocessing.py --limit 10    # quick dev run
    python scripts/benchmark_preprocessing.py --no-cache    # force re-evaluation
    python scripts/benchmark_preprocessing.py --no-raw      # skip raw (use exp_001 results)

Requirements:
    Gemini_API_Key in .env
    100 images in data/raw/ matching golden_test_set.json

Cost estimate:
    ~$0.003/image × 100 images × 2 (raw + preprocessed) ≈ $0.60 per full run
    --limit 10 costs ~$0.06 for development testing

Crash safety:
    Results saved to data/benchmark_preprocessing_checkpoint.json after each image.
    Re-run the script to resume from where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import PIL.Image
from dotenv import load_dotenv

# ── Project setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import ImagePreprocessor
from src.evaluation.metrics import evaluate, get_failures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
GOLDEN_PATH  = PROJECT_ROOT / "data" / "golden_test_set.json"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
CHECKPOINT   = PROJECT_ROOT / "data" / "benchmark_preprocessing_checkpoint.json"
DEFAULT_OUT  = PROJECT_ROOT / "docs" / "experiments" / "exp_002_preprocessing.md"

# ── Gemini config ─────────────────────────────────────────────────────────────
MODEL          = "gemini-2.5-flash-lite"
MAX_RETRIES    = 3
CALL_DELAY_SEC = 4.5       # free tier rate limit
MAX_IMG_WIDTH  = 900       # px — match baseline notebook for fair comparison

# ── Naive prompt (identical to exp_001 baseline) ──────────────────────────────
NAIVE_PROMPT = """\
Extract information from this receipt image and return a JSON object.

Return only a JSON object with these fields:
- merchant_name: string or null
- date: string in DD-MM-YYYY format, or null
- total_amount: integer (VND, no commas), or null
- items: list of {"name": string, "quantity": number, "unit_price": integer, "total": integer}
- category: one of ["Ăn uống", "Đi lại", "Mua sắm", "Giải trí", "Hoá đơn tiện ích", "Sức khoẻ", "Giáo dục", "Du lịch", "Nhà cửa", "Khác"]

Return only the JSON, no explanation."""


# ── Image helpers ─────────────────────────────────────────────────────────────

def find_image_path(image_name: str, group: str) -> Path | None:
    """Search data/raw/<group>/<image_name>, fallback to recursive search."""
    candidate = RAW_DIR / group / image_name
    if candidate.exists():
        return candidate
    # Fallback: search all groups
    for path in RAW_DIR.rglob(image_name):
        return path
    return None


def image_to_bytes(pil_img: PIL.Image.Image, max_width: int = MAX_IMG_WIDTH) -> bytes:
    """Resize if too wide, convert to JPEG bytes."""
    w, h = pil_img.size
    if w > max_width:
        pil_img = pil_img.resize((max_width, int(h * max_width / w)))
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def ndarray_to_bytes(image: "np.ndarray", max_width: int = MAX_IMG_WIDTH) -> bytes:
    """Convert BGR numpy array (from cv2) to JPEG bytes."""
    import cv2, numpy as np
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(img_rgb)
    return image_to_bytes(pil_img, max_width)


def sha256_key(image_bytes: bytes, prompt: str) -> str:
    """Cache key: SHA256 of image bytes + prompt text."""
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(prompt.encode())
    return h.hexdigest()[:16]


# ── Gemini API call ───────────────────────────────────────────────────────────

def call_gemini(
    client,
    image_bytes: bytes,
    prompt: str,
) -> tuple[dict | None, float, str | None]:
    """
    Call Gemini Vision with retry + exponential backoff.

    Returns (parsed_result_dict, latency_sec, error_message_or_None).
    parsed_result_dict is None on failure.
    """
    from google.genai import types

    t0 = time.perf_counter()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            latency = time.perf_counter() - t0
            raw = response.text.strip() if response.text else ""
            parsed = _parse_response(raw)
            if parsed is None:
                return None, latency, "json_parse_error"
            return parsed, latency, None

        except Exception as exc:
            last_error = str(exc)
            retryable = any(
                x in last_error for x in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]
            )
            if retryable and attempt < MAX_RETRIES:
                wait = 5 * (2 ** attempt)
                log.warning("  [retry %d/%d] rate-limited, waiting %ds...", attempt, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                break

    latency = time.perf_counter() - t0
    return None, latency, last_error


def _parse_response(raw: str) -> dict | None:
    """Extract JSON from Gemini response (handles ```json ... ``` wrapping)."""
    if not raw:
        return None
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    candidate = match.group(1).strip() if match else raw.strip()
    if not candidate.startswith("{"):
        brace = re.search(r"(\{[\s\S]+\})", raw)
        candidate = brace.group(1) if brace else candidate
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _error_pred(reason: str) -> dict:
    """Null prediction for failed API calls."""
    return {
        "merchant_name": None,
        "date": None,
        "total_amount": None,
        "items": [],
        "category": "Khác",
        "_error": reason,
    }


# ── Process a single record ───────────────────────────────────────────────────

def process_single(
    record: dict,
    client,
    preprocessor: ImagePreprocessor,
    cache: dict,
    run_raw: bool,
    run_preprocessed: bool,
) -> dict:
    """
    Process one golden set record: call Gemini with raw and/or preprocessed image.

    Returns a result dict with raw_pred, preprocessed_pred, latencies, etc.
    Saves to cache by SHA256 key to avoid redundant API calls.
    """
    image_name = record["image"]
    group = record.get("group", "")
    img_path = find_image_path(image_name, group)

    result: dict = {
        "image": image_name,
        "group": group,
        "raw_pred": None,
        "raw_latency": 0.0,
        "preprocessed_pred": None,
        "preprocessed_latency": 0.0,
        "preprocessing_ms": 0.0,
        "preprocessing_steps": [],
        "preprocessing_warnings": [],
        "errors": [],
    }

    if img_path is None:
        result["errors"].append(f"image not found: {image_name}")
        result["raw_pred"] = _error_pred("image_not_found")
        result["preprocessed_pred"] = _error_pred("image_not_found")
        return result

    # ── Raw image ─────────────────────────────────────────────────────────────
    if run_raw:
        try:
            raw_bytes = image_to_bytes(PIL.Image.open(img_path))
            cache_key = "raw_" + sha256_key(raw_bytes, NAIVE_PROMPT)

            if cache_key in cache:
                result["raw_pred"] = cache[cache_key]["pred"]
                result["raw_latency"] = cache[cache_key]["latency"]
            else:
                pred, latency, error = call_gemini(client, raw_bytes, NAIVE_PROMPT)
                time.sleep(CALL_DELAY_SEC)
                if pred is None:
                    pred = _error_pred(error or "unknown")
                    result["errors"].append(f"raw Gemini error: {error}")
                result["raw_pred"] = pred
                result["raw_latency"] = latency
                cache[cache_key] = {"pred": pred, "latency": latency}

        except Exception as exc:
            log.error("raw processing error for %s: %s", image_name, exc)
            result["raw_pred"] = _error_pred(str(exc))
            result["errors"].append(f"raw error: {exc}")
    else:
        result["raw_pred"] = _error_pred("skipped")

    # ── Preprocessed image ────────────────────────────────────────────────────
    if run_preprocessed:
        try:
            import cv2
            proc_result = preprocessor.process(str(img_path))
            result["preprocessing_ms"] = proc_result.processing_time_ms
            result["preprocessing_steps"] = proc_result.steps_applied
            result["preprocessing_warnings"] = proc_result.warnings

            if not proc_result.success or proc_result.image is None:
                result["preprocessed_pred"] = _error_pred("preprocessing_failed")
                result["errors"].append("preprocessing failed")
            else:
                pre_bytes = ndarray_to_bytes(proc_result.image)
                cache_key = "pre_" + sha256_key(pre_bytes, NAIVE_PROMPT)

                if cache_key in cache:
                    result["preprocessed_pred"] = cache[cache_key]["pred"]
                    result["preprocessed_latency"] = cache[cache_key]["latency"]
                else:
                    pred, latency, error = call_gemini(client, pre_bytes, NAIVE_PROMPT)
                    time.sleep(CALL_DELAY_SEC)
                    if pred is None:
                        pred = _error_pred(error or "unknown")
                        result["errors"].append(f"preprocessed Gemini error: {error}")
                    result["preprocessed_pred"] = pred
                    result["preprocessed_latency"] = latency
                    cache[cache_key] = {"pred": pred, "latency": latency}

        except Exception as exc:
            log.error("preprocessed processing error for %s: %s", image_name, exc)
            result["preprocessed_pred"] = _error_pred(str(exc))
            result["errors"].append(f"preprocessed error: {exc}")
    else:
        result["preprocessed_pred"] = _error_pred("skipped")

    return result


# ── Report generation ─────────────────────────────────────────────────────────

def format_report(
    golden: list[dict],
    results: list[dict],
    raw_report: dict,
    pre_report: dict,
    baseline_e2e: float = 0.620,
) -> str:
    """Render the Markdown experiment document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(results)
    n_preprocessed = sum(1 for r in results if r.get("preprocessed_steps_applied"))

    # Preprocessing stats
    n_orientation = sum(
        1 for r in results
        if any("orientation" in s for s in r.get("preprocessing_steps", []))
    )
    n_deskew = sum(
        1 for r in results
        if any("deskew" in s for s in r.get("preprocessing_steps", []))
    )
    n_doc_crop = sum(
        1 for r in results
        if any("document_crop" in s for s in r.get("preprocessing_steps", []))
    )
    n_dark = sum(
        1 for r in results
        if any("gamma" in s for s in r.get("preprocessing_steps", []))
    )
    preproc_times = [r["preprocessing_ms"] for r in results if r["preprocessing_ms"] > 0]
    preproc_p50 = sorted(preproc_times)[len(preproc_times) // 2] if preproc_times else 0
    preproc_p95_idx = max(0, int(len(preproc_times) * 0.95) - 1)
    preproc_p95 = sorted(preproc_times)[preproc_p95_idx] if preproc_times else 0
    n_with_warnings = sum(1 for r in results if r.get("preprocessing_warnings"))

    def pct(v):
        return f"{v:.1%}"

    def delta(new, old):
        d = new - old
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1%}"

    r = raw_report
    p = pre_report

    lines = [
        f"# exp_002 — Image Preprocessing Module",
        f"",
        f"**Date:** {now}",
        f"**Model:** {MODEL}",
        f"**Preprocessing version:** preprocessing_v1",
        f"**Dataset:** golden_test_set.json ({n} samples evaluated)",
        f"**Baseline reference:** exp_001_naive_baseline.md ({baseline_e2e:.1%} E2E)",
        f"",
        f"## Hypothesis",
        f"",
        f"Preprocessing (orientation fix, CLAHE contrast, deskew, denoise) will improve",
        f"Gemini extraction accuracy by correcting image quality issues before sending to the VLM.",
        f"Expected gain: +5–10% E2E, especially for groups 04_handwritten and 05_low_quality.",
        f"",
        f"## Preprocessing Pipeline Config",
        f"",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Target image size | 1600px (longest side) |",
        f"| CLAHE clip limit | 2.0 |",
        f"| Deskew max angle | 15° |",
        f"| NLM denoise strength | h=6.0 |",
        f"| Idempotency gate | skip heavy steps if brightness > 150 |",
        f"| Prompt | naive_v0 (identical to exp_001) |",
        f"",
        f"## Results — Overall",
        f"",
        f"| Metric | Baseline (exp_001) | Raw re-run | With Preprocessing | Delta (pre vs raw) |",
        f"|---|---|---|---|---|",
        f"| **E2E success** | **{baseline_e2e:.1%}** | **{pct(r['end_to_end_success'])}** | **{pct(p['end_to_end_success'])}** | **{delta(p['end_to_end_success'], r['end_to_end_success'])}** |",
        f"| Merchant acc | 78.0% | {pct(r['field_accuracy']['merchant'])} | {pct(p['field_accuracy']['merchant'])} | {delta(p['field_accuracy']['merchant'], r['field_accuracy']['merchant'])} |",
        f"| Date acc | 81.0% | {pct(r['field_accuracy']['date'])} | {pct(p['field_accuracy']['date'])} | {delta(p['field_accuracy']['date'], r['field_accuracy']['date'])} |",
        f"| Total acc | 92.0% | {pct(r['field_accuracy']['total'])} | {pct(p['field_accuracy']['total'])} | {delta(p['field_accuracy']['total'], r['field_accuracy']['total'])} |",
        f"| Items F1 | 85.3% | {pct(r['items_f1'])} | {pct(p['items_f1'])} | {delta(p['items_f1'], r['items_f1'])} |",
        f"| Category acc | 75.0% | {pct(r['category_accuracy'])} | {pct(p['category_accuracy'])} | {delta(p['category_accuracy'], r['category_accuracy'])} |",
        f"| Latency p50 | 3.22s | {r['latency_p50']:.2f}s | {p['latency_p50']:.2f}s | — |",
        f"| Latency p95 | 6.44s | {r['latency_p95']:.2f}s | {p['latency_p95']:.2f}s | — |",
        f"",
        f"## Results — Per Group",
        f"",
        f"| Group | n | E2E baseline | E2E raw | E2E preprocessed | Delta |",
        f"|---|---|---|---|---|---|",
    ]

    # Baseline per-group from exp_001
    baseline_per_group = {
        "01_pos_clean":   0.731,
        "02_vat_invoice": 0.588,
        "03_long_invoice": 0.750,
        "04_handwritten": 0.273,
        "05_low_quality": 0.600,
    }

    for grp, gs_raw in sorted(r.get("per_group", {}).items()):
        gs_pre = p.get("per_group", {}).get(grp)
        if gs_pre is None:
            continue
        baseline_grp = baseline_per_group.get(grp, 0.0)
        lines.append(
            f"| {grp} | {gs_raw['n']} | {baseline_grp:.1%} | "
            f"{gs_raw['end_to_end_success']:.1%} | "
            f"{gs_pre['end_to_end_success']:.1%} | "
            f"{delta(gs_pre['end_to_end_success'], gs_raw['end_to_end_success'])} |"
        )

    lines += [
        f"",
        f"## Preprocessing Stats",
        f"",
        f"| Stat | Value |",
        f"|---|---|",
        f"| Images with orientation correction | {n_orientation}/{n} |",
        f"| Images with deskew applied | {n_deskew}/{n} |",
        f"| Images with document crop | {n_doc_crop}/{n} |",
        f"| Images with darkness correction | {n_dark}/{n} |",
        f"| Mean preprocessing time | {sum(preproc_times)/len(preproc_times):.0f}ms |" if preproc_times else "| Mean preprocessing time | N/A |",
        f"| Preprocessing p50 | {preproc_p50:.0f}ms |",
        f"| Preprocessing p95 | {preproc_p95:.0f}ms |",
        f"| Images with preprocessing warnings | {n_with_warnings}/{n} |",
        f"",
        f"## Failure Analysis",
        f"",
    ]

    # Failure comparison
    raw_preds = [r_item["raw_pred"] for r_item in results if r_item["raw_pred"]]
    pre_preds = [r_item["preprocessed_pred"] for r_item in results if r_item["preprocessed_pred"]]
    gt_list = golden[:len(raw_preds)]

    raw_failures = get_failures(raw_preds, gt_list)
    pre_failures = get_failures(pre_preds, gt_list[:len(pre_preds)])

    lines += [
        f"- Raw failures: {len(raw_failures)}/{n} ({len(raw_failures)/n:.1%})",
        f"- Preprocessed failures: {len(pre_failures)}/{n} ({len(pre_failures)/n:.1%})",
        f"",
        f"## Decision",
        f"",
    ]

    e2e_delta = p["end_to_end_success"] - r["end_to_end_success"]
    all_groups_ok = all(
        abs(p.get("per_group", {}).get(g, {}).get("end_to_end_success", 0)
            - r.get("per_group", {}).get(g, {}).get("end_to_end_success", 0)) >= -0.02
        for g in r.get("per_group", {})
    )

    if e2e_delta >= 0.01 and all_groups_ok:
        verdict = (
            f"**PASS** — E2E improves by {e2e_delta:+.1%} (≥ +1pp threshold) "
            f"and no group regresses more than 2pp."
        )
    elif e2e_delta < 0.01:
        verdict = (
            f"**FAIL** — E2E delta {e2e_delta:+.1%} is below the +1pp threshold. "
            f"Preprocessing module needs tuning before promotion to production."
        )
    else:
        verdict = (
            f"**FAIL (regression)** — E2E delta is {e2e_delta:+.1%} but at least one "
            f"group regresses by more than 2pp. Investigate per-group results."
        )

    lines.append(verdict)
    lines += [
        f"",
        f"*(Fill in qualitative observations after reviewing failure cases)*",
        f"",
        f"## Notes",
        f"",
        f"- Both raw and preprocessed evaluations use the identical naive_v0 prompt",
        f"  (same as exp_001) to isolate the effect of preprocessing from prompt changes.",
        f"- Preprocessing p95 latency must stay under 500ms to leave budget for",
        f"  the <5s total latency target (5000ms - 500ms = 4500ms for Gemini).",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark preprocessing vs raw image for Gemini extraction")
    p.add_argument("--golden",  default=str(GOLDEN_PATH), help="Path to golden_test_set.json")
    p.add_argument("--output",  default=str(DEFAULT_OUT),  help="Output Markdown report path")
    p.add_argument("--limit",   type=int, default=None,    help="Process only N images (for dev)")
    p.add_argument("--no-cache",  action="store_true",     help="Ignore existing cache")
    p.add_argument("--no-raw",    action="store_true",     help="Skip raw image evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = os.getenv("Gemini_API_Key")
    if not api_key:
        log.error("Gemini_API_Key not found in .env")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    # Load golden test set
    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)
    if args.limit:
        golden = golden[:args.limit]
    log.info("Processing %d images from golden test set", len(golden))

    # Load checkpoint (crash-safe resume)
    cache: dict = {}
    checkpoint_results: dict[str, dict] = {}
    if not args.no_cache and CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            saved = json.load(f)
            cache = saved.get("cache", {})
            checkpoint_results = saved.get("results", {})
        log.info("Resumed from checkpoint: %d images already done", len(checkpoint_results))

    preprocessor = ImagePreprocessor()
    results: list[dict] = []

    for i, record in enumerate(golden, 1):
        image_name = record["image"]
        if image_name in checkpoint_results:
            log.info("[%d/%d] %s — loaded from checkpoint", i, len(golden), image_name)
            results.append(checkpoint_results[image_name])
            continue

        log.info("[%d/%d] Processing %s ...", i, len(golden), image_name)
        result = process_single(
            record, client, preprocessor, cache,
            run_raw=(not args.no_raw),
            run_preprocessed=True,
        )
        results.append(result)
        checkpoint_results[image_name] = result

        # Save checkpoint after every image (crash-safe)
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump({"cache": cache, "results": checkpoint_results}, f, ensure_ascii=False, indent=2)

    # Evaluate
    log.info("Running evaluation...")
    raw_preds   = [r["raw_pred"]          for r in results if r["raw_pred"]]
    pre_preds   = [r["preprocessed_pred"] for r in results if r["preprocessed_pred"]]
    raw_lats    = [r["raw_latency"]        for r in results]
    pre_lats    = [r["preprocessed_latency"] for r in results]
    gt_list     = golden[:len(raw_preds)]

    # Add group/image fields to predictions for per-group breakdown
    for pred, gt in zip(raw_preds, gt_list):
        pred["image"] = gt["image"]
        pred["group"] = gt.get("group", "unknown")
    for pred, gt in zip(pre_preds, gt_list[:len(pre_preds)]):
        pred["image"] = gt["image"]
        pred["group"] = gt.get("group", "unknown")

    raw_report = evaluate(raw_preds, gt_list, latencies=raw_lats[:len(raw_preds)])
    pre_report = evaluate(pre_preds, gt_list[:len(pre_preds)], latencies=pre_lats[:len(pre_preds)])

    log.info("Raw      E2E: %.1f%%", raw_report["end_to_end_success"] * 100)
    log.info("Preprocessed E2E: %.1f%%", pre_report["end_to_end_success"] * 100)
    log.info("Delta: %+.1f%%", (pre_report["end_to_end_success"] - raw_report["end_to_end_success"]) * 100)

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = format_report(gt_list, results, raw_report, pre_report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    log.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
