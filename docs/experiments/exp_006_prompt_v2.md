# exp_006 — Extraction Prompt V2: Handwritten + Long Invoice Fixes

**Date:** 2026-04-14 16:04
**Model:** gemini-2.5-flash-lite
**Prompt version:** EXTRACTION_PROMPT_V2
**Dataset:** golden_test_set.json (100 samples)
**Baseline reference:** exp_005 (EXTRACTION_PROMPT_V1) = 68.0% E2E

## Hypotheses

Two systematic failures were identified in exp_005 and targeted in V2:

**H1 — Handwritten regression** (04_handwritten: 63.6% → 27.3%, -36.3pp)
Root cause: V1 anti-hallucination rules ('return null if unclear') caused Gemini
to return null for ALL uncertain handwritten fields — correct for printed receipts,
but catastrophic for handwritten where uncertainty is expected.
V2 fix: Two-tier null policy. Handwritten receipts get 'best-effort' instructions:
'read every item even if partially legible', 'provide best reading rather than null'.
Expected: 04_handwritten recovers to ≥55%.

**H2 — Long invoice date/total failures** (03_long_invoice: 75.0% → 68.8%, -6.2pp)
Root cause: Supermarket receipts (BigC, Lotte, Co.opmart) print multiple dates
(invoice date vs transaction date) and multiple subtotals (per category).
V1 prompt gave no guidance, so Gemini picked the wrong date/total.
V2 fix: Explicit SUPERMARKET MULTI-DATE RULE and SUPERMARKET MULTI-TOTAL RULE
with named labels to look for ('Ngày mua', 'Tổng thanh toán') and chain store list.
Expected: 03_long_invoice recovers to ≥75%.

## What Changed (V1 → V2)

| Component | V1 | V2 |
|---|---|---|
| Preprocessing | ImagePreprocessor v3 | ImagePreprocessor v3 (unchanged) |
| Receipt classification (Step 1) | Binary (receipt / not) | 5-way type classification |
| Date extraction (Step 3) | Single date | + SUPERMARKET MULTI-DATE RULE |
| Item extraction (Step 4) | Strict null | + HANDWRITTEN RULE (best-effort) |
| Total extraction (Step 5) | Single total | + SUPERMARKET MULTI-TOTAL RULE |
| Few-shot Example 2 | VAT invoice | Supermarket multi-total (Lotte Mart) |
| Few-shot Example 3 | Handwritten with nulls | Handwritten with best-effort items |
| Hard rules | Single null policy | Two-tier: printed strict / handwritten lenient |

## Results — Overall

| Metric | exp_005 (V1 baseline) | exp_006 (V2 prompt) | Delta |
|---|---|---|---|
| **E2E success** | **68.0%** | **68.0%** | **+0.0%** |
| Merchant acc | 85.0% | 86.0% | +1.0% |
| Date acc | 84.0% | 83.0% | -1.0% |
| Total acc | 90.0% | 90.0% | +0.0% |
| Items F1 | — | 84.3% | — |
| Items precision | — | 88.6% | — |
| Items recall | — | 84.5% | — |
| Latency p50 | — | 4.56s | — |
| Latency p95 | — | 7.32s | — |
| Cost / receipt | — | $0.0003 | — |
| Total API cost | — | $0.031 | — |
| Cache hits | — | 0/100 (0%) | — |

## Results — Per Difficulty Group

| Group | n | E2E (V1, exp_005) | E2E (V2, exp_006) | Delta |
|---|---|---|---|---|
| 01_pos_clean | 26 | 73.1% | 76.9% | +3.8% |
| 02_vat_invoice | 17 | 70.0% | 58.8% | -11.2% ⚠ |
| 03_long_invoice | 16 | 68.8% | 68.8% | -0.0% |
| 04_handwritten | 11 | 27.3% | 27.3% | -0.0% |
| 05_low_quality | 30 | 76.7% | 80.0% | +3.3% |

*⚠ = group regresses more than 2pp  |  ✓ = group improves by ≥5pp*

## Hypothesis Verification

- H1 (handwritten recovery): 27.3% → 27.3% (-0.0%) — FAILED ✗
- H2 (long_invoice date/total fix): 68.8% → 68.8% (-0.0%) — FAILED ✗

## Error Summary

- Total images processed: 100
- Images with any error: 2
- API errors (after retries): 1
- Extraction failures: 4

## Top Failure Cases

- **handwritten_0007.jpg** (04_handwritten): merchant=OK, date=FAIL, total=OK, items_f1=0.00
- **long_invoice_0032.jpg** (03_long_invoice): merchant=OK, date=FAIL, total=OK, items_f1=1.00
- **vat_invoice_0023.jpg** (02_vat_invoice): merchant=FAIL, date=OK, total=OK, items_f1=1.00
- **vat_invoice_0005.jpg** (02_vat_invoice): merchant=OK, date=FAIL, total=OK, items_f1=1.00
- **low_quality_0000.jpg** (05_low_quality): merchant=FAIL, date=OK, total=FAIL, items_f1=0.00
- **low_quality_0038.jpg** (05_low_quality): merchant=OK, date=FAIL, total=OK, items_f1=0.50
- **vat_invoice_0020.jpg** (02_vat_invoice): merchant=FAIL, date=OK, total=OK, items_f1=1.00
- **pos_clean_0012.jpg** (01_pos_clean): merchant=OK, date=FAIL, total=OK, items_f1=1.00

## Decision

**FAIL** — E2E delta +0.0% is below the +1pp threshold. V2 prompt changes did not produce sufficient overall improvement.

EXTRACTION_PROMPT_V2 is NOT promoted. Analysis of remaining failures required.

## Root Cause Analysis

### 04_handwritten still underperforming (27.3%)

Possible causes:
- Gemini still defaults to null for very low-confidence handwritten items
- OCR noise too high for model to produce meaningful text
- Few-shot example not representative enough of actual handwritten samples

Suggested V3 fix: Add explicit instruction 'Write the characters you see,
even if you are only 30% confident — use [?] for unreadable single characters'.

### 03_long_invoice still underperforming (68.8%)

Possible causes:
- Multi-date / multi-total rule not specific enough for the failing images
- Some long invoices fail due to item extraction (sliding window needed?)
- Date format variant not covered (e.g., non-standard date position)

Suggested V3 fix: Review the actual failing images, check if date/total
fields are the remaining blockers or if item extraction is now the limit.

## Next Steps

1. Manually inspect the top failure cases above
2. Identify whether failures are: date format, total selection, or item hallucination
3. Build EXTRACTION_PROMPT_V3 targeting remaining systematic errors
4. Re-run exp_007 (prompt V3 benchmark)

## Notes

- Category accuracy is N/A in this experiment — PhoBERT classifier is built in Week 7.
- V2 prompt is ~20% longer than V1 (~840 tokens vs ~700), reflected in cost.
- exp_005 handwritten subgroup numbers (27.3%) include the ReceiptItem.total=null
  bug fix from mid-exp_005 run. All exp_006 numbers use the fixed schema.
- All images processed through ImagePreprocessor v3 (unchanged since exp_004).