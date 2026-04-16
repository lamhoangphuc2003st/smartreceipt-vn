# exp_001 — Gemini Naive Baseline

**Date:** 2026-04-13 13:31  
**Model:** gemini-2.5-flash-lite  
**Prompt version:** naive_v0 (no role, no CoT, no few-shot, no structured output API)  
**Dataset:** golden_test_set.json (100 samples)

## Hypothesis

Gemini Vision với prompt tối giản (chỉ yêu cầu JSON, không có kỹ thuật prompting) sẽ cho kết quả ở mức ~50–65% end-to-end success — đủ để prove concept nhưng rõ ràng cần cải thiện.

## Prompt

```
Extract information from this receipt image and return a JSON object.

Return only a JSON object with these fields:
- merchant_name: string or null
- date: string in DD-MM-YYYY format, or null
- total_amount: integer (VND, no commas), or null
- items: list of {"name": string, "quantity": number, "unit_price": integer, "total": integer}
- category: one of ["Ăn uống", "Đi lại", "Mua sắm", "Giải trí", "Hoá đơn tiện ích", "Sức khoẻ", "Giáo dục", "Du lịch", "Nhà cửa", "Khác"]

Return only the JSON, no explanation.
```

## Results

### Overall

| Metric | Score |
|---|---|
| **End-to-end success** | **62.0%** |
| Merchant accuracy | 78.0% |
| Date accuracy | 81.0% |
| Total accuracy | 92.0% |
| Items F1 | 85.3% |
| Items Precision | 89.5% |
| Items Recall | 85.7% |
| Category accuracy | 75.0% |
| Latency p50 | 3.22s |
| Latency p95 | 6.44s |

### Per-group

| Group | n | E2E | Merchant | Date | Total | Items F1 | Category |
|---|---|---|---|---|---|---|---|
| 01_pos_clean | 26 | 73.1% | 88.5% | 88.5% | 96.2% | 75.9% | 88.5% |
| 02_vat_invoice | 17 | 58.8% | 76.5% | 82.4% | 100.0% | 98.0% | 82.4% |
| 03_long_invoice | 16 | 75.0% | 87.5% | 100.0% | 87.5% | 98.1% | 50.0% |
| 04_handwritten | 11 | 27.3% | 54.5% | 45.5% | 54.5% | 77.0% | 63.6% |
| 05_low_quality | 30 | 60.0% | 73.3% | 76.7% | 100.0% | 82.5% | 76.7% |

### Per-category

| Category | n | Accuracy |
|---|---|---|
| Giải trí | 2 | 100.0% |
| Hoá đơn tiện ích | 5 | 100.0% |
| Sức khoẻ | 3 | 100.0% |
| Đi lại | 15 | 93.3% |
| Mua sắm | 10 | 80.0% |
| Nhà cửa | 5 | 80.0% |
| Giáo dục | 7 | 71.4% |
| Ăn uống | 41 | 65.9% |
| Du lịch | 5 | 60.0% |
| Khác | 7 | 57.1% |

## Failure Analysis

**Total failures:** 38/100 (38.0%)

| Failure type | Count |
|---|---|
| Chỉ merchant sai | 14 |
| Chỉ date sai | 12 |
| Chỉ total sai | 4 |
| Nhiều fields sai | 8 |

## Observations

*(Điền sau khi đọc qua failure cases)*

- [ ] TODO: Gemini có bỏ dấu tiếng Việt không?
- [ ] TODO: Format date có nhất quán không (DD-MM-YYYY vs YYYY-MM-DD)?
- [ ] TODO: Nhóm nào fail nhiều nhất?
- [ ] TODO: Category confusion matrix — hay nhầm giữa nhóm nào?

## Decision

Baseline thiết lập. Các bước tiếp theo (prompt v1, notebook 03):
1. Thêm role definition — "You are a Vietnamese accounting expert..."
2. Thêm chain-of-thought steps
3. Thêm few-shot examples (2-3 hard cases)
4. Thêm anti-hallucination constraints
5. Dùng structured output API (Pydantic schema)
6. Benchmark v1 vs naive_v0 trên cùng golden set
