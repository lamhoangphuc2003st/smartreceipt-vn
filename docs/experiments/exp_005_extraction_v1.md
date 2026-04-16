# exp_005 — VLM Extraction: Structured Prompt V1

**Date:** 2026-04-14 14:42
**Model:** gemini-2.5-flash-lite
**Prompt version:** EXTRACTION_PROMPT_V1
**Dataset:** golden_test_set.json (100 samples)
**Baseline reference:** exp_004 (preprocessing + naive prompt) = 67.0% E2E

## Hypothesis

Replacing the 7-line naive prompt with a structured prompt (role definition,
chain-of-thought steps, 3 few-shot examples, anti-hallucination rules, self-check)
will improve extraction accuracy significantly. The naive prompt gives Gemini no
guidance on receipt structure, date format, or how to handle ambiguous fields.
Expected gain: +10–18pp E2E (from 67% toward 85%).

## What Changed (vs exp_004)

| Component | exp_004 | exp_005 |
|---|---|---|
| Preprocessing | ImagePreprocessor v3 | ImagePreprocessor v3 (unchanged) |
| Prompt | Naive 7-line prompt | EXTRACTION_PROMPT_V1 (structured) |
| Schema enforcement | None (free-form JSON) | Pydantic Receipt schema |
| Temperature | 0 | 0 |
| Retry/backoff | Manual 3× | GeminiExtractor (1s/2s/4s backoff) |
| Disk cache | Per-script SHA256 | GeminiExtractor 7-day TTL cache |

## Results — Overall

| Metric | exp_004 (baseline) | exp_005 (V1 prompt) | Delta |
|---|---|---|---|
| **E2E success** | **67.0%** | **68.0%** | **+1.0%** |
| Merchant acc | 77.0% | 85.0% | +8.0% |
| Date acc | 81.0% | 84.0% | +3.0% |
| Total acc | 90.0% | 90.0% | +0.0% |
| Items F1 | — | 82.4% | — |
| Items precision | — | 86.5% | — |
| Items recall | — | 83.1% | — |
| Category acc | — | N/A (PhoBERT not built) | — |
| Latency p50 | — | 4.17s | — |
| Latency p95 | — | 7.78s | — |
| Cost / receipt | — | $0.0003 | — |
| Total API cost | — | $0.027 | — |
| Cache hits | — | 0/100 (0%) | — |

## Results — Per Difficulty Group

| Group | n | E2E (exp_004) | E2E (exp_005) | Delta |
|---|---|---|---|---|
| 01_pos_clean | 26 | 61.5% | 73.1% | +11.6% |
| 02_vat_invoice | 17 | 65.0% | 70.6% | +5.6% |
| 03_long_invoice | 16 | 75.0% | 68.8% | -6.2% ⚠ |
| 04_handwritten | 11 | 63.6% | 27.3% | -36.3% ⚠ |
| 05_low_quality | 30 | 73.3% | 76.7% | +3.4% |

*⚠ = group regresses more than 2pp vs baseline*

## Error Summary

- Total images processed: 100
- Images with any error: 2
- API errors (after retries): 1
- Extraction failures (parse/timeout): 4

## Top 5 Failure Cases

- **handwritten_0007.jpg** (04_handwritten): merchant=OK, date=FAIL, total=OK, items_f1=0.00
- **long_invoice_0032.jpg** (03_long_invoice): merchant=OK, date=FAIL, total=OK, items_f1=1.00
- **vat_invoice_0023.jpg** (02_vat_invoice): merchant=FAIL, date=OK, total=OK, items_f1=1.00
- **vat_invoice_0005.jpg** (02_vat_invoice): merchant=OK, date=FAIL, total=OK, items_f1=1.00
- **low_quality_0000.jpg** (05_low_quality): merchant=FAIL, date=OK, total=FAIL, items_f1=0.00

## Decision

**FAIL (regression)** — E2E delta +1.0% but at least one group regresses >2pp.

EXTRACTION_PROMPT_V1 is NOT promoted to production.
Next step: analyze failure cases above, create V2 addressing systematic errors.

## Next Steps

- Week 6: Build validation layer (math check, date sanity, merchant fuzzy match)
  to catch VLM errors before they reach the user.
- If FAIL: iterate prompt to V2 targeting the most common error patterns above.
- Week 7: Fine-tune PhoBERT on extracted merchant+item text for category classification.

## Notes

- Category accuracy is N/A in this experiment — PhoBERT classifier is built in Week 7.
  The V1 prompt intentionally does not ask for category to keep the task focused.
- exp_004 field accuracy numbers (merchant/date/total) are re-computed from the
  checkpoint file rather than the original report to ensure consistent measurement.
- All images processed through ImagePreprocessor v3 (same as exp_004) to isolate
  the prompt effect.