# exp_002 — Image Preprocessing Module

**Date:** 2026-04-13 15:10
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
| **E2E success** | **62.0%** | **65.0%** | **47.0%** | **-18.0%** |
| Merchant acc | 78.0% | 80.0% | 69.0% | -11.0% |
| Date acc | 81.0% | 84.0% | 71.0% | -13.0% |
| Total acc | 92.0% | 92.0% | 80.0% | -12.0% |
| Items F1 | 85.3% | 83.9% | 72.1% | -11.8% |
| Category acc | 75.0% | 76.0% | 71.0% | -5.0% |
| Latency p50 | 3.22s | 3.70s | 3.71s | — |
| Latency p95 | 6.44s | 16.48s | 17.21s | — |

## Results — Per Group

| Group | n | E2E baseline | E2E raw | E2E preprocessed | Delta |
|---|---|---|---|---|---|
| 01_pos_clean | 26 | 73.1% | 69.2% | 57.7% | -11.5% |
| 02_vat_invoice | 17 | 58.8% | 58.8% | 35.3% | -23.5% |
| 03_long_invoice | 16 | 75.0% | 81.2% | 37.5% | -43.8% |
| 04_handwritten | 11 | 27.3% | 36.4% | 36.4% | +0.0% |
| 05_low_quality | 30 | 60.0% | 66.7% | 53.3% | -13.3% |

## Preprocessing Stats

| Stat | Value |
|---|---|
| Images with orientation correction | 14/100 |
| Images with deskew applied | 5/100 |
| Images with document crop | 0/100 |
| Images with darkness correction | 0/100 |
| Mean preprocessing time | 1088ms |
| Preprocessing p50 | 638ms |
| Preprocessing p95 | 2933ms |
| Images with preprocessing warnings | 29/100 |

## Failure Analysis

- Raw failures: 35/100 (35.0%)
- Preprocessed failures: 53/100 (53.0%)

## Decision

**FAIL** — E2E delta -18.0% is below the +1pp threshold. Preprocessing module needs tuning before promotion to production.

*(Fill in qualitative observations after reviewing failure cases)*

## Notes

- Both raw and preprocessed evaluations use the identical naive_v0 prompt
  (same as exp_001) to isolate the effect of preprocessing from prompt changes.
- Preprocessing p95 latency must stay under 500ms to leave budget for
  the <5s total latency target (5000ms - 500ms = 4500ms for Gemini).