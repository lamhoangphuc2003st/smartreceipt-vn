# Evaluation: full-pipeline-v1-final

**Date:** 2026-04-15 21:25  
**Mode:** full  
**Samples:** 100  

## Summary

| Metric | Value |
|--------|-------|
| End-to-end success | 70.0% |
| Merchant accuracy | 84.0% |
| Date accuracy | 89.0% |
| Total accuracy | 93.0% |
| Items F1 | 88.4% |
| Category accuracy | 83.0% |
| Latency p50 | 198 ms |
| Latency p95 | 9838 ms |

## Per-difficulty breakdown

| Group | N | Merchant | Date | Total | Items F1 | Category | E2E |
|-------|---|----------|------|-------|----------|----------|-----|
| 01_pos_clean | 26 | 88.5% | 96.2% | 100.0% | 83.8% | 80.8% | 84.6% |
| 02_vat_invoice | 17 | 70.6% | 88.2% | 94.1% | 97.5% | 64.7% | 58.8% |
| 03_long_invoice | 16 | 87.5% | 93.8% | 87.5% | 96.6% | 100.0% | 68.8% |
| 04_handwritten | 11 | 81.8% | 63.6% | 81.8% | 87.9% | 81.8% | 36.4% |
| 05_low_quality | 30 | 86.7% | 90.0% | 93.3% | 83.1% | 86.7% | 76.7% |

## Category accuracy

| Category | Accuracy |
|----------|----------|
| Mua sắm | 100.0% |
| Nhà cửa | 100.0% |
| Sức khoẻ | 100.0% |
| Ăn uống | 95.1% |
| Giáo dục | 85.7% |
| Đi lại | 80.0% |
| Hoá đơn tiện ích | 80.0% |
| Giải trí | 50.0% |
| Du lịch | 40.0% |
| Khác | 14.3% |

## Failure analysis (end-to-end failures, first 10)

| Image | Group | Merchant? | Date? | Total? | Items F1 |
|-------|-------|-----------|-------|--------|----------|
| low_quality_0042.jpg | 05_low_quality | ✗ | ✓ | ✓ | 1.00 |
| vat_invoice_0023.jpg | 02_vat_invoice | ✓ | ✗ | ✓ | 1.00 |
| low_quality_0000.jpg | 05_low_quality | ✓ | ✓ | ✗ | 0.00 |
| long_invoice_0005.jpg | 03_long_invoice | ✗ | ✓ | ✓ | 1.00 |
| low_quality_0038.jpg | 05_low_quality | ✓ | ✗ | ✓ | 0.50 |
| vat_invoice_0020.jpg | 02_vat_invoice | ✗ | ✓ | ✓ | 1.00 |
| handwritten_0000.jpg | 04_handwritten | ✗ | ✓ | ✓ | 1.00 |
| handwritten_0004.jpg | 04_handwritten | ✓ | ✗ | ✓ | 1.00 |
| long_invoice_0015.jpg | 03_long_invoice | ✗ | ✓ | ✓ | 0.67 |
| vat_invoice_0009.jpg | 02_vat_invoice | ✓ | ✓ | ✗ | 1.00 |

## Notes

### Observations

**End-to-end 70% vs field-level accuracy (84–93%):** E2E yêu cầu *tất cả* critical fields đúng cùng lúc — field-level accuracy tốt nhưng E2E thấp hơn target 85%. Root cause chính là `04_handwritten` (36.4%) và `02_vat_invoice` (58.8%).

**Latency p95 = 9.8s:** Vượt target 5s. Nguyên nhân: Gemini free-tier (~7–11s/request), không phải preprocessing hay pipeline overhead (p50=198ms). Acceptable cho portfolio demo; production fix là paid tier hoặc caching.

**Handwritten (36.4% E2E):** Bottleneck chính. Date accuracy chỉ 63.6% — Gemini gặp khó với chữ viết tay không rõ. Known limitation, documented.

**VAT invoice (58.8% E2E):** Merchant accuracy thấp nhất (70.6%) — tên công ty dài, format đa dạng. VAT invoice có layout khác POS receipts.

**Category accuracy 83% (full) vs 91% (classification-only):** 8pp drop do extraction errors propagate — khi merchant/items extract sai thì category cũng sai theo. Bình thường.

**Data fix:** `low_quality_0003.gif` → `.jpg` (typo trong golden_test_set.json, file thực là jpg).

### Known limitations (accepted)
- `04_handwritten`: 36.4% E2E — sẽ cải thiện với binarization preprocessing hoặc accept as limitation
- `02_vat_invoice`: 58.8% E2E — merchant extraction khó vì tên công ty dài
- `p95 latency`: 9.8s — Gemini free-tier bottleneck, không phải pipeline