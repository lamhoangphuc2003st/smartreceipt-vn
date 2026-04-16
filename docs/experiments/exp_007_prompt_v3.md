# exp_007 — Prompt V3 + Structured Output (response_schema)

**Date:** 2026-04-14 17:32
**Model:** gemini-2.5-flash-lite
**Prompt version:** EXTRACTION_PROMPT_V3
**Structured output:** response_schema=Receipt, response_mime_type=application/json
**Dataset:** golden_test_set.json (100 samples)
**Baseline:** exp_006 (EXTRACTION_PROMPT_V2, free-form JSON) = 68.0% E2E

## What Changed (V2 → V3)

| Component | exp_006 (V2) | exp_007 (V3) |
|---|---|---|
| Output enforcement | Prompt: 'Return JSON only' | API: response_schema=Receipt |
| JSON parsing | Code-fence strip + brace-find heuristics | json.loads() direct |
| Few-shot Example 2 | Supermarket (Lotte Mart) | VAT invoice (CÔNG TY ABC) ← restored |
| Few-shot Example 3 | Handwritten (partial) | Supermarket multi-total |
| MULTI-DATE RULE scope | All receipts | Named supermarket chains only |
| MULTI-TOTAL RULE scope | All receipts | Named supermarket chains only |
| Handwritten instruction | Best-effort reading | [?] placeholder + 'never empty items' |

## Hypotheses

- H1: 02_vat_invoice recovers from 58.8% → ≥70% (example 2 restored as VAT invoice)
- H2: 04_handwritten ≥ 27.3% (maintain or improve via explicit [?] placeholder)
- H3: 03_long_invoice ≥ 68.8% (chain-scoped rules prevent false triggers on other types)
- H4: parse_errors = 0 (structured output guarantee vs 4 in exp_006)

## Results — Overall

| Metric | exp_006 (V2) | exp_007 (V3) | Delta |
|---|---|---|---|
| **E2E success** | **68.0%** | **67.0%** | **-1.0%** |
| Merchant acc | 86.0% | 85.0% | -1.0% |
| Date acc | 83.0% | 83.0% | +0.0% |
| Total acc | 90.0% | 90.0% | +0.0% |
| Items F1 | 84.3% | 81.0% | -3.3% |
| Items precision | 88.6% | 88.1% | -0.5% |
| Items recall | 84.5% | 81.5% | -3.0% |
| Parse errors | 4 | 1 | -3 |
| Latency p50 | 4.56s | 4.34s | — |
| Latency p95 | 7.32s | 6.48s | — |
| Cost / receipt | $0.0003 | $0.0003 | — |
| Total API cost | — | $0.031 | — |
| Cache hits | — | 0/100 (0%) | — |

## Results — Per Difficulty Group

| Group | n | E2E (V2, exp_006) | E2E (V3, exp_007) | Delta |
|---|---|---|---|---|
| 01_pos_clean | 26 | 76.9% | 76.9% | +0.0% |
| 02_vat_invoice | 17 | 58.8% | 70.6% | +11.8% ✓ |
| 03_long_invoice | 16 | 68.8% | 68.8% | -0.0% |
| 04_handwritten | 11 | 27.3% | 27.3% | -0.0% |
| 05_low_quality | 30 | 80.0% | 70.0% | -10.0% ⚠ |

*⚠ = regresses >2pp  |  ✓ = improves ≥5pp*

## Hypothesis Verification

- H1 02_vat_invoice recovery: 58.8% → 70.6% (+11.8%) — CONFIRMED ✓
- H2 04_handwritten baseline hold: 27.3% → 27.3% (-0.0%) — FAILED ✗
- H3 03_long_invoice baseline hold: 68.8% → 68.8% (-0.0%) — FAILED ✗
- H4: parse_errors = 1 (was 4) — PARTIAL (1 remaining)

## Error Summary

- Total images: 100
- Images with any error: 2
- API errors (after retries): 1
- Parse errors: 1 (structured output target: 0)

## Top Failure Cases

- **handwritten_0007.jpg** (04_handwritten): merchant=OK, date=FAIL, total=OK, items_f1=0.00
- **pos_clean_0024.jpg** (01_pos_clean): merchant=FAIL, date=OK, total=FAIL, items_f1=0.00
- **long_invoice_0032.jpg** (03_long_invoice): merchant=OK, date=FAIL, total=OK, items_f1=1.00
- **low_quality_0042.jpg** (05_low_quality): merchant=FAIL, date=OK, total=OK, items_f1=1.00
- **vat_invoice_0023.jpg** (02_vat_invoice): merchant=FAIL, date=FAIL, total=FAIL, items_f1=0.00
- **vat_invoice_0005.jpg** (02_vat_invoice): merchant=OK, date=FAIL, total=OK, items_f1=1.00
- **low_quality_0000.jpg** (05_low_quality): merchant=FAIL, date=OK, total=FAIL, items_f1=0.00
- **low_quality_0038.jpg** (05_low_quality): merchant=OK, date=FAIL, total=OK, items_f1=0.50

## Decision

**FAIL** — E2E delta -1.0% < +1pp threshold.

EXTRACTION_PROMPT_V3 is NOT promoted. See root cause analysis below.

## Root Cause Analysis

### 04_handwritten still underperforming (27.3%)

The [?] placeholder and 'never empty list' instruction may still be insufficient.
Consider: preprocessing enhancement specifically for handwritten (higher contrast,
binarization), or accept handwritten as a known limitation and document it.

## Next Steps

1. Inspect failing images for the most-regressed groups
2. Determine if failures are OCR-limited (image quality) or prompt-limited
3. If OCR-limited: apply targeted preprocessing (binarization for handwritten)
4. If prompt-limited: iterate to V4

## Notes

- Structured output (response_schema) was added alongside prompt V3.
  If parse errors drop to 0, the structured output change is confirmed beneficial
  regardless of overall E2E outcome.
- Category accuracy N/A — PhoBERT classifier is Week 7.
- All images processed through ImagePreprocessor v3 (unchanged).