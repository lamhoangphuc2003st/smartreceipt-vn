# exp_002 — Image Preprocessing Module

**Date:** 2026-04-13 18:06
**Model:** gemini-2.5-flash-lite
**Preprocessing version:** preprocessing_v1
**Dataset:** golden_test_set.json (100 samples evaluated)
**Baseline reference:** exp_001_naive_baseline.md (62.0% E2E)

## Hypothesis

Preprocessing (orientation fix, CLAHE contrast, deskew, denoise) will improve
Gemini extraction accuracy by correcting image quality issues before sending to the VLM.
Expected gain: +5–10% E2E, especially for groups 04_handwritten and 05_low_quality.

## Preprocessing Pipeline Config

| Parameter | Value |
|---|---|
| Target image size | 1600px (longest side) |
| CLAHE clip limit | 2.0 |
| Deskew max angle | 15° |
| NLM denoise strength | h=6.0 |
| Idempotency gate | skip heavy steps if brightness > 150 |
| Prompt | naive_v0 (identical to exp_001) |

## Results — Overall

| Metric | Baseline (exp_001) | Raw re-run | With Preprocessing | Delta (pre vs raw) |
|---|---|---|---|---|
| **E2E success** | **62.0%** | **63.0%** | **50.0%** | **-13.0%** |
| Merchant acc | 78.0% | 81.0% | 66.0% | -15.0% |
| Date acc | 81.0% | 80.0% | 75.0% | -5.0% |
| Total acc | 92.0% | 91.0% | 78.0% | -13.0% |
| Items F1 | 85.3% | 84.9% | 73.5% | -11.3% |
| Category acc | 75.0% | 74.0% | 67.0% | -7.0% |
| Latency p50 | 3.22s | 3.69s | 3.88s | — |
| Latency p95 | 6.44s | 7.37s | 6.03s | — |

## Results — Per Group

| Group | n | E2E baseline | E2E raw | E2E preprocessed | Delta |
|---|---|---|---|---|---|
| 01_pos_clean | 26 | 73.1% | 73.1% | 46.2% | -26.9% |
| 02_vat_invoice | 17 | 58.8% | 64.7% | 41.2% | -23.5% |
| 03_long_invoice | 16 | 75.0% | 75.0% | 68.8% | -6.2% |
| 04_handwritten | 11 | 27.3% | 36.4% | 27.3% | -9.1% |
| 05_low_quality | 30 | 60.0% | 56.7% | 56.7% | +0.0% |

## Preprocessing Stats

| Stat | Value |
|---|---|
| Images with orientation correction | 1/100 |
| Images with deskew applied | 8/100 |
| Images with document crop | 3/100 |
| Images with darkness correction | 0/100 |
| Mean preprocessing time | 70ms |
| Preprocessing p50 | 25ms |
| Preprocessing p95 | 261ms |
| Images with preprocessing warnings | 39/100 |

## Failure Analysis

- Raw failures: 37/100 (37.0%)
- Preprocessed failures: 50/100 (50.0%)

## Decision

**FAIL** — E2E delta -13.0% is below the +1pp threshold. Preprocessing module needs tuning before promotion to production.

*(Fill in qualitative observations after reviewing failure cases)*

## Notes

- Both raw and preprocessed evaluations use the identical naive_v0 prompt
  (same as exp_001) to isolate the effect of preprocessing from prompt changes.
- Preprocessing p95 latency must stay under 500ms to leave budget for
  the <5s total latency target (5000ms - 500ms = 4500ms for Gemini).