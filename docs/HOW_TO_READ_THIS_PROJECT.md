# Hướng dẫn đọc hiểu project SmartReceipt VN

Đây là hướng dẫn dành cho người muốn hiểu toàn bộ project từ đầu — bao gồm lý do tồn tại của từng file, thứ tự đọc tối ưu, và cách các thành phần kết nối với nhau.

---

## Trước khi đọc code: hiểu bức tranh lớn

**Project làm gì?**

Nhận một ảnh chụp hóa đơn Việt Nam (siêu thị, nhà hàng, hóa đơn VAT, viết tay...) và trả về JSON có cấu trúc:

```
Ảnh hóa đơn → Tên cửa hàng + Ngày + Danh sách hàng + Tổng tiền + Danh mục chi tiêu
```

**Tại sao dùng 2 model thay vì 1?**

| Vấn đề | Model được chọn | Lý do |
|--------|----------------|-------|
| Hiểu ảnh + layout hóa đơn | Gemini Vision API | Giỏi xử lý hình ảnh, không cần train |
| Phân loại danh mục chi tiêu | PhoBERT (fine-tuned) | Rẻ hơn LLM 100x, 87ms latency, 91% accuracy |

Đây là quyết định kiến trúc quan trọng nhất — không dùng 1 LLM làm tất cả.

---

## Thứ tự đọc file (từ tổng quan đến chi tiết)

### Giai đoạn 1 — Hiểu bối cảnh và mục tiêu

**Đọc theo thứ tự này:**

#### 1. `README.md` (root)
Bức tranh tổng quan: project làm gì, số liệu thực tế, cách chạy nhanh.
Chú ý phần **Evaluation Results** — đây là kết quả thật, không phóng đại.

#### 2. `docs/proposal.md`
Lý do project ra đời, quyết định kiến trúc ban đầu (tại sao không dùng Donut, tại sao không dùng LLM cho classification).
Quan trọng để hiểu **tư duy** đằng sau các lựa chọn.

#### 3. `CLAUDE.md` (root)
Bộ nguyên tắc kỹ thuật chi phối mọi quyết định trong project.
Đọc phần **"The six core principles"** và **"Anti-patterns"** — đây là kim chỉ nam.

#### 4. `smart_receiptvn_roadmap.md`
Thấy được plan tổng thể: cái gì đã làm, cái gì chưa làm, thứ tự ưu tiên.

---

### Giai đoạn 2 — Hiểu quá trình phát triển (quan trọng nhất)

Đọc **9 experiment logs** theo thứ tự. Đây là lịch sử thật của project — mỗi file cho thấy một quyết định kỹ thuật và kết quả đo được.

```
docs/experiments/
├── exp_001_naive_baseline.md        ← Bắt đầu từ 12% E2E (keyword-only)
├── exp_002_preprocessing.md         ← +deskew: 12% → 38%
├── exp_003_preprocessing_v2.md      ← +document detect: 38% → 51%
├── exp_004_preprocessing_v3.md      ← +perspective warp: 51% → 58%
├── exp_005_extraction_v1.md         ← Thêm Gemini: 58% → 64%
├── exp_006_prompt_v2.md             ← +chain-of-thought: 64% → 68%
├── exp_007_prompt_v3.md             ← +anti-hallucination: 68% → 70%
├── exp_008_phobert_v1_final.md      ← PhoBERT classifier: 91% isolated
└── exp_009_full_pipeline_v1_final.md ← Full pipeline: 70% E2E (hiện tại)
```

**Bài học từ chuỗi experiment này:**
- Preprocessing đơn giản (deskew, enhance) tăng 26 điểm phần trăm — quan trọng hơn model
- Prompt engineering tăng 6 điểm — đáng đầu tư hơn fine-tuning
- PhoBERT không thay đổi E2E (vẫn 70%) nhưng giảm chi phí và latency classification

---

### Giai đoạn 3 — Đọc data schemas (trước khi đọc code)

Hiểu cấu trúc dữ liệu trước sẽ giúp code dễ hiểu hơn nhiều.

#### 5. `src/extraction/schemas.py`
Định nghĩa các Pydantic model trung tâm của toàn hệ thống:
- `ReceiptItem` — một dòng hàng hóa
- `Receipt` — toàn bộ hóa đơn
- `ExtractionResult` — kết quả trả về từ Gemini (bao gồm metadata, success/fail)

#### 6. `api/models.py`
Request/Response schemas của API:
- `ReceiptResponse` — cái user nhận được cuối cùng
- Thấy được sự khác biệt giữa internal schema và public API schema

#### 7. `data/golden_test_set.json` (xem 5–10 entries đầu)
Hiểu format dữ liệu thật: hóa đơn trông như thế nào trong JSON, độ phức tạp của từng difficulty group.

#### 8. `data/merchants.json` (xem 10 entries đầu)
Thấy cách lưu canonical name + aliases để fuzzy matching — giải quyết vấn đề OCR sai tên cửa hàng.

---

### Giai đoạn 4 — Đọc source code theo luồng xử lý

Đọc code **theo đúng thứ tự dữ liệu chạy qua**, không đọc theo folder.

```
Ảnh đầu vào
    ↓
[1] src/preprocessing/pipeline.py       ← Entry point preprocessing
    ↓
[2] src/preprocessing/orientation.py    ← Sửa xoay ảnh
    ↓
[3] src/preprocessing/document_detect.py ← Đánh giá chất lượng + phát hiện vùng hóa đơn
    ↓
[4] src/preprocessing/perspective.py    ← Sửa góc chụp lệch
    ↓
[5] src/preprocessing/deskew.py         ← Sửa nghiêng nhỏ
    ↓
[6] src/preprocessing/enhance.py        ← Tăng độ tương phản, khử nhiễu
    ↓
[7] src/extraction/prompts.py           ← Đọc cả 3 version prompt, so sánh sự tiến hóa
    ↓
[8] src/extraction/gemini_extractor.py  ← Wrapper Gemini (retry, cache, cost tracking)
    ↓
[9] src/validation/validators.py        ← Kiểm tra toán học + sửa lỗi tự động
    ↓
[10] src/classification/keyword_classifier.py  ← Fallback đơn giản (đọc trước)
    ↓
[11] src/classification/phobert_classifier.py  ← Classifier chính
    ↓
[12] src/pipeline/receipt_pipeline.py   ← Orchestrator: kết nối tất cả
    ↓
[13] api/routes/receipts.py             ← Endpoint HTTP cuối cùng
    ↓
[14] api/main.py                        ← App factory, middleware
```

**Khi đọc mỗi file, chú ý:**
- Class/function chính là gì?
- Input và output là gì?
- Error handling ở đâu?
- Fallback khi thất bại là gì?

---

### Giai đoạn 5 — Đọc tests để hiểu contract

Tests là tài liệu tốt nhất — chúng cho thấy system được kỳ vọng làm gì trong mọi tình huống.

#### 9. `tests/test_preprocessing.py`
Thấy được: ảnh sạch không bị làm tệ hơn (idempotency), ảnh xoay được sửa đúng.

#### 10. `tests/test_extraction.py`
Thấy được: mock Gemini API, test retry logic, test cache hit/miss.

#### 11. `tests/test_validation.py`
Thấy được: các trường hợp math fail, date invalid, merchant fuzzy match.

#### 12. `tests/test_classification.py`
Thấy được: 10 category, edge cases khi confidence thấp.

#### 13. `tests/test_pipeline.py`
Thấy được: full integration test — system vẫn trả về kết quả dù Gemini fail.

---

### Giai đoạn 6 — Infrastructure và deployment

#### 14. `Dockerfile`
Multi-stage build: stage 1 install deps (nặng), stage 2 copy code (nhẹ).
Lý do dùng `torch==2.4.0+cpu` thay vì GPU — Railway không có GPU, không cần.

#### 15. `docker-compose.yml`
Local dev setup: API container + Streamlit container, shared volume.

#### 16. `.github/workflows/ci.yml`
CI pipeline: test → Docker build → health check verify.
Không có auto-deploy — deploy thủ công trên Railway.

#### 17. `hf-space/app.py`
Streamlit UI production (trên Hugging Face Spaces).
Khác với `app/streamlit_app.py` (legacy, chạy pipeline trực tiếp) — bản này gọi Railway API.

#### 18. `docs/deployment.md`
Chi tiết deploy Railway + HF Spaces, environment variables cần thiết.

---

### Giai đoạn 7 — Đọc những limitations thật

#### 19. `docs/edge_cases.md`
9 failure patterns được document rõ ràng. Đây là phần quan trọng cho interview — interviewer sẽ hỏi "system fail ở đâu?".

---

## Bản đồ dependencies giữa các module

```
api/routes/receipts.py
    └── src/pipeline/receipt_pipeline.py
            ├── src/preprocessing/pipeline.py
            │       ├── orientation.py
            │       ├── document_detect.py
            │       ├── perspective.py
            │       ├── deskew.py
            │       └── enhance.py
            ├── src/extraction/gemini_extractor.py
            │       ├── schemas.py          ← dùng chung khắp nơi
            │       ├── prompts.py
            │       └── sliding_window.py
            ├── src/validation/validators.py
            │       └── data/merchants.json
            └── src/classification/phobert_classifier.py
                    └── src/classification/keyword_classifier.py  (fallback)
```

**Rule quan trọng về dependencies:**
- Layer dưới KHÔNG biết về layer trên
- `preprocessing` không biết về `extraction`
- `extraction` không biết về `classification`
- Chỉ `pipeline` mới biết về tất cả — đây là điểm giao tiếp duy nhất

---

## Các file KHÔNG cần đọc kỹ

| File | Lý do bỏ qua |
|------|-------------|
| `hf_spaces/` (folder cũ) | Superseded bởi `hf-space/` |
| `app/streamlit_app.py` | Legacy, không dùng trong production |
| `notebooks/` | Exploration notebooks, không phải production code |
| `scripts/` | Utility scripts sinh training data |
| `models/` | Local model cache, có thể trống |
| `configs/` | Thường trống hoặc chứa config thực nghiệm |
| `tmp/` | Temporary files |
| `data/raw/` | Raw images (bị gitignore) |
| `data/*_checkpoint.json` | Experiment checkpoints trung gian |

---

## Những chỗ code thú vị nhất

### 1. `src/extraction/gemini_extractor.py` — production engineering mẫu mực
Xem cách implement: retry với exponential backoff, SHA256 cache, cost tracking, graceful failure — tất cả trong một class.

### 2. `src/extraction/prompts.py` — prompt engineering có hệ thống
So sánh V1, V2, V3. Thấy rõ: thêm chain-of-thought tăng accuracy, thêm anti-hallucination rules giảm invented data.

### 3. `src/preprocessing/document_detect.py` — quality-aware preprocessing
Cách system tự đánh giá ảnh đầu vào (blur score, darkness score) và quyết định có cần crop không — tránh làm hỏng ảnh đã tốt.

### 4. `src/pipeline/receipt_pipeline.py` — tiered fallback thực sự
Xem cách mỗi stage có try/except riêng, lỗi không bubble up, luôn trả về `ProcessingResult` dù fail ở đâu.

### 5. `src/validation/validators.py` — correction vs rejection
Cách system sửa lỗi tự động (fuzzy merchant match) thay vì reject, và log mọi auto-correction.

---

## Tóm tắt số liệu cần nhớ

| Metric | Giá trị |
|--------|---------|
| E2E success rate | 70% |
| Total amount accuracy | 93% |
| PhoBERT category accuracy | 91% |
| Handwritten receipt E2E | 36.4% (điểm yếu rõ nhất) |
| Latency p50 | 198ms |
| Latency p95 | 9.8s (Gemini API) |
| Số experiments | 9 |
| Số test cases golden set | 100 |
| Số merchants trong database | ~65 |
| Số expense categories | 10 |

---

## Câu hỏi phỏng vấn thường gặp và file nào trả lời

| Câu hỏi | File tham khảo |
|---------|---------------|
| "Tại sao dùng Gemini thay vì fine-tune model riêng?" | `docs/proposal.md`, `CLAUDE.md` |
| "Tại sao dùng PhoBERT thay vì GPT cho classification?" | `CLAUDE.md` → Anti-patterns |
| "Accuracy của system là bao nhiêu?" | `docs/experiments/exp_009_full_pipeline_v1_final.md` |
| "System fail ở trường hợp nào?" | `docs/edge_cases.md` |
| "Làm sao biết thay đổi có cải thiện không?" | `src/evaluation/run_eval.py`, `data/golden_test_set.json` |
| "Preprocessing làm những gì?" | `docs/experiments/exp_002–004`, `src/preprocessing/` |
| "Prompt thay đổi như thế nào qua các phiên bản?" | `docs/experiments/exp_005–007`, `src/extraction/prompts.py` |
| "System hoạt động khi Gemini down không?" | `src/pipeline/receipt_pipeline.py` |
| "Deploy như thế nào?" | `docs/deployment.md`, `Dockerfile`, `docker-compose.yml` |
