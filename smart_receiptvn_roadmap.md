# SmartReceipt VN - Kế hoạch thực hiện 12 tuần

> **Project:** Vietnamese Receipt Understanding System
> **Thời gian:** 12 tuần (~15-20 giờ/tuần)
> **Mục tiêu:** Build hệ thống end-to-end trích xuất + phân loại hoá đơn VN, đủ ấn tượng cho vị trí AI Engineer fresher/intern
> **Tech stack:** Python, FastAPI, Gemini Vision API, PhoBERT, OpenCV, Streamlit, Docker, Hugging Face

---

## Tổng quan các phase

| Phase | Tuần | Tên phase | Mục tiêu chính |
|-------|------|-----------|----------------|
| 1 | 1-3 | Foundation & Exploration | Hiểu bài toán, build baseline, có data |
| 2 | 4-7 | Core Development | Build pipeline hoàn chỉnh, fine-tune PhoBERT |
| 3 | 8-10 | Optimization & Polish | Cải thiện accuracy, handle edge cases, evaluation |
| 4 | 11-12 | Deployment & Presentation | Deploy, viết blog, chuẩn bị phỏng vấn |

---

## PHASE 1: FOUNDATION & EXPLORATION (Tuần 1-3)

**Mục tiêu phase:** Hiểu rõ bài toán, có môi trường dev, có baseline chạy được, có ~200 ảnh hoá đơn để bắt đầu.

### Tuần 1: Setup & Research

**Mục tiêu:** Xong môi trường dev, hiểu state-of-the-art, có plan rõ ràng.

**Tasks:**

- [ ] **Setup môi trường development**
  - Cài Python 3.10+, tạo virtual env (venv hoặc conda)
  - Cài các thư viện cơ bản: `torch`, `transformers`, `opencv-python`, `fastapi`, `google-generativeai`, `pydantic`, `streamlit`
  - Setup Git repo trên GitHub với README, .gitignore, LICENSE (MIT)
  - Setup structure folder:
    ```
    smartreceipt-vn/
    ├── data/              # Dataset (gitignore)
    ├── notebooks/         # Jupyter experiments
    ├── src/
    │   ├── preprocessing/
    │   ├── extraction/
    │   ├── classification/
    │   ├── validation/
    │   └── pipeline/
    ├── tests/
    ├── api/              # FastAPI backend
    ├── ui/               # Streamlit frontend
    ├── configs/
    └── docs/
    ```

- [ ] **Research & đọc tài liệu**
  - Đọc paper Donut (Kim et al., 2022) - hiểu cách document AI hoạt động
  - Đọc về LayoutLMv3 để biết alternative
  - Research Gemini Vision API documentation - các capabilities, pricing, rate limits
  - Đọc về PhoBERT (Nguyen & Nguyen, 2020) - hiểu tại sao dùng cho tiếng Việt
  - Xem 2-3 demo project OCR tiếng Việt trên GitHub để biết landscape

- [ ] **Đăng ký các API cần thiết**
  - Google AI Studio → lấy Gemini API key (free tier)
  - Hugging Face account → để upload model sau
  - Google Colab Pro (optional, ~10$/tháng) hoặc dùng free tier để train PhoBERT

- [ ] **Viết project proposal document**
  - Tạo `docs/proposal.md`
  - Mô tả bài toán, mục tiêu, scope, non-goals
  - Vẽ architecture diagram (dùng draw.io hoặc excalidraw)
  - Define success metrics cụ thể

**Deliverable tuần 1:**
- ✅ Repo GitHub với structure chuẩn
- ✅ Proposal document + architecture diagram
- ✅ Môi trường dev chạy được "Hello World" với Gemini API

**Học gì:** Document AI landscape, VLM capabilities, project planning

---

### Tuần 2: Data Collection - Phần khó nhất

**Mục tiêu:** Thu thập 200+ ảnh hoá đơn đa dạng. Đây là tuần vất vả nhất nhưng quan trọng nhất.

**Tasks:**

- [ ] **Lập kế hoạch thu thập data**
  - Target 200 ảnh chia theo 5 nhóm độ khó:
    - Nhóm 1 (POS hiện đại): 60 ảnh (Highlands, Circle K, GS25, Co.opmart...)
    - Nhóm 2 (VAT điện tử): 40 ảnh
    - Nhóm 3 (Siêu thị dài): 40 ảnh (BigC, Lotte, Mega Market)
    - Nhóm 4 (Viết tay/quán nhỏ): 30 ảnh
    - Nhóm 5 (Ảnh chất lượng kém): 30 ảnh

- [ ] **Thực hiện thu thập**
  - Xin hoá đơn từ bạn bè, gia đình (giải thích cho project học tập)
  - Tự chụp khi đi mua sắm, ăn uống
  - Crawl một số ảnh public từ Google Images (cẩn thận bản quyền, chỉ dùng để test)
  - Lưu theo format: `data/raw/{nhom}/{timestamp}_{merchant}.jpg`
  - **Quan trọng:** Blur/che thông tin cá nhân nhạy cảm (MST cá nhân, địa chỉ nhà)

- [ ] **Organize data**
  - Tạo spreadsheet `data/metadata.csv` với columns: filename, group, merchant_name, date, has_vat, notes
  - Phân nhóm vào folders theo độ khó
  - Backup lên Google Drive (phòng mất data)

- [ ] **Thử nghiệm Gemini Vision API với 10 ảnh random**
  - Viết script đơn giản `notebooks/01_gemini_baseline.ipynb`
  - Gửi 10 ảnh với prompt naive ("Read this receipt and extract info")
  - Ghi chú: cái nào OK, cái nào fail, fail như thế nào
  - Đây là BASELINE để so sánh sau này

**Deliverable tuần 2:**
- ✅ 200+ ảnh hoá đơn đã organize trong `data/raw/`
- ✅ Metadata CSV
- ✅ Notebook baseline với 10 ảnh test + nhận xét

**Học gì:** Data collection best practices, understanding data distribution

**⚠️ Lưu ý:** Nếu không đủ 200 ảnh thật, có thể bổ sung bằng:
- Dataset MC-OCR 2021 của VLSP (public)
- SROIE dataset (tiếng Anh, nhưng có format tương tự)
- Tuyệt đối không generate synthetic data ở giai đoạn này - phải có data thật

---

### Tuần 3: Labeling & Baseline Evaluation

**Mục tiêu:** Label 100 ảnh làm golden test set, chạy baseline evaluation.

**Tasks:**

- [ ] **Setup labeling tool**
  - Cài Label Studio: `pip install label-studio`
  - Config project với schema: merchant_name, date, time, items (list), subtotal, vat, total, payment_method, category
  - Import 100 ảnh vào Label Studio

- [ ] **Label 100 ảnh cẩn thận (golden test set)**
  - Mỗi ảnh label tất cả fields
  - Với items: label đầy đủ name + quantity + unit_price + total
  - Gán category (Ăn uống, Đi lại, Mua sắm, etc.)
  - Export ra JSON format: `data/golden_test_set.json`
  - **Đây là data CỰC KỲ quan trọng** - dùng để benchmark mọi cải tiến

- [ ] **Viết evaluation framework**
  - File `src/evaluation/metrics.py`
  - Implement các metrics:
    - Field-level accuracy (exact match + fuzzy match cho text fields)
    - Items F1 score (precision + recall)
    - Category accuracy
    - End-to-end success rate
  - Viết function `evaluate(predictions, ground_truth) -> dict`

- [ ] **Chạy baseline với Gemini API naive prompt**
  - Chạy Gemini trên toàn bộ 100 ảnh golden set
  - Dùng evaluation framework để đo
  - Ghi kết quả vào `docs/experiments/exp_001_baseline.md`
  - **Expected baseline:** ~70-75% field-level accuracy

- [ ] **Error analysis**
  - Xem 20 cases fail tồi tệ nhất
  - Tìm pattern: lỗi loại gì? Nhóm ảnh nào? Field nào?
  - Document trong `docs/error_analysis_v1.md`

**Deliverable tuần 3:**
- ✅ Golden test set 100 ảnh labeled (JSON)
- ✅ Evaluation framework hoạt động
- ✅ Baseline results + error analysis document

**Học gì:** Data labeling, evaluation design, error analysis methodology

---

## PHASE 2: CORE DEVELOPMENT (Tuần 4-7)

**Mục tiêu phase:** Build các component chính của pipeline, fine-tune PhoBERT, kết nối thành system hoàn chỉnh.

### Tuần 4: Preprocessing Module

**Mục tiêu:** Build preprocessing pipeline hoàn chỉnh, cải thiện chất lượng ảnh trước khi gửi VLM.

**Tasks:**

- [ ] **Implement các preprocessing functions**
  - `src/preprocessing/orientation.py`: detect + fix rotation (EXIF + Hough Transform)
  - `src/preprocessing/deskew.py`: nắn thẳng ảnh nghiêng
  - `src/preprocessing/document_detect.py`: detect bounding box hoá đơn, crop
  - `src/preprocessing/perspective.py`: perspective transform (warp về hình chữ nhật)
  - `src/preprocessing/enhance.py`: denoise, CLAHE contrast enhancement
  - `src/preprocessing/pipeline.py`: class `ImagePreprocessor` gom tất cả

- [ ] **Viết unit tests**
  - `tests/test_preprocessing.py`
  - Test với ảnh nghiêng, ảnh mờ, ảnh thiếu sáng
  - Đảm bảo output luôn valid image

- [ ] **Benchmark preprocessing**
  - Chạy Gemini trên 100 golden set, 2 version: có và không preprocessing
  - So sánh accuracy
  - **Expected improvement:** +5-10% accuracy
  - Document trong `docs/experiments/exp_002_preprocessing.md`

- [ ] **Handle edge cases**
  - Ảnh không phải hoá đơn → return error
  - Ảnh quá mờ (detect blur bằng Laplacian variance) → warning
  - Ảnh quá tối → auto brightness adjustment

**Deliverable tuần 4:**
- ✅ Preprocessing module hoàn chỉnh với tests
- ✅ Benchmark results cho thấy improvement
- ✅ Document trade-offs và limitations

**Học gì:** Computer vision cơ bản, OpenCV, image processing pipeline

---

### Tuần 5: VLM Extraction Module - Phần quan trọng nhất

**Mục tiêu:** Build VLM wrapper production-grade với advanced prompting.

**Tasks:**

- [ ] **Design prompt v1 - có CoT và few-shot**
  - Viết `src/extraction/prompts.py`
  - Prompt bao gồm: role, CoT steps, few-shot examples, constraints, self-check
  - Tham khảo lại phần "Chiến thuật 1" trong cuộc thảo luận trước

- [ ] **Implement Gemini wrapper**
  - `src/extraction/gemini_extractor.py` - class `GeminiExtractor`
  - Methods: `extract(image_path) -> Receipt`
  - Features:
    - Retry logic (3 lần với exponential backoff)
    - Timeout handling
    - Error handling (API down, rate limit, invalid response)
    - Cost tracking (log tokens used)
    - Structured output với Pydantic schema

- [ ] **Implement Pydantic schemas**
  - `src/extraction/schemas.py`
  - Classes: `ReceiptItem`, `Receipt`, `ExtractionResult`
  - Validators cho date format, currency, etc.

- [ ] **Benchmark prompt v1 vs baseline**
  - Chạy trên golden set
  - So sánh với baseline tuần 3
  - **Expected:** 75% → 85% field-level accuracy
  - Document `docs/experiments/exp_003_prompt_v1.md`

- [ ] **Iterate prompt v2, v3**
  - Analyze fail cases của v1
  - Cải thiện prompt dựa trên patterns
  - Mỗi version chạy evaluation và log lại
  - Chọn version tốt nhất làm production

**Deliverable tuần 5:**
- ✅ GeminiExtractor module production-ready
- ✅ Prompt library với nhiều versions
- ✅ Benchmark results cho từng version
- ✅ Accuracy tăng lên ~85%

**Học gì:** Prompt engineering, LLM API best practices, structured output

---

### Tuần 6: Validation Layer + Sliding Window cho ảnh dài

**Mục tiêu:** Build validation/correction layer, handle hoá đơn siêu thị dài.

**Tasks:**

- [ ] **Implement validation layer**
  - `src/validation/validators.py`
  - Math validator: check sum(items) + vat = total
  - Date validator: check hợp lý, normalize format
  - Currency validator: check VND format, detect outliers
  - Merchant validator: fuzzy match với database merchants phổ biến

- [ ] **Build merchant database**
  - `data/merchants.json` - khoảng 200-500 merchants phổ biến VN
  - Bao gồm: tên chính thức, các biến thể tên, category mặc định
  - Có thể crawl từ các website hoặc tự liệt kê

- [ ] **Implement auto-correction**
  - `src/validation/correctors.py`
  - Fix common OCR errors (số 0 vs chữ O, số 1 vs chữ l)
  - Normalize dates, currencies
  - Fuzzy match merchant names

- [ ] **Implement sliding window cho ảnh dài**
  - `src/extraction/sliding_window.py`
  - Detect ảnh dài (aspect ratio > 2.5)
  - Split thành 2-3 chunks overlap 20%
  - Extract từng chunk, merge items (dedup dựa trên position)

- [ ] **Benchmark trên golden set**
  - Đặc biệt focus nhóm 3 (siêu thị dài)
  - **Expected:** Nhóm 3 accuracy 65% → 80%
  - Document `docs/experiments/exp_004_validation_sliding.md`

**Deliverable tuần 6:**
- ✅ Validation + correction layer
- ✅ Sliding window implementation
- ✅ Merchant database
- ✅ Accuracy tổng tăng lên ~88-90%

**Học gì:** Data validation, fuzzy matching, handling long inputs cho LLM

---

### Tuần 7: Fine-tune PhoBERT Classifier

**Mục tiêu:** Train PhoBERT để phân loại category chi tiêu.

**Tasks:**

- [ ] **Build training data cho classification**
  - Target 2000-3000 samples
  - Sources:
    - Label từ 200 ảnh đã thu thập (200 samples)
    - Auto-label bằng Gemini: tạo prompt "Phân loại hoá đơn này vào category X" cho 1500 samples còn lại
    - Verify manual 300 samples random để đảm bảo quality
  - Format: CSV với columns `text` (merchant + items) và `label` (category)
  - Split: 70% train, 15% val, 15% test

- [ ] **Define categories**
  - 10 categories: Ăn uống, Đi lại, Mua sắm, Giải trí, Hoá đơn tiện ích, Sức khoẻ, Giáo dục, Du lịch, Nhà cửa, Khác
  - Viết guide định nghĩa rõ từng category

- [ ] **Fine-tune PhoBERT**
  - Notebook `notebooks/02_finetune_phobert.ipynb`
  - Base model: `vinai/phobert-base`
  - Add classification head
  - Hyperparameters: lr=2e-5, batch_size=32, epochs=5, max_length=128
  - Train trên Google Colab free GPU (T4 đủ)
  - Use early stopping với val loss
  - Save checkpoint tốt nhất

- [ ] **Evaluate model**
  - Accuracy, F1 per class, confusion matrix
  - **Expected:** 85-92% accuracy
  - Viết evaluation report

- [ ] **Export và integration**
  - Export model sang `src/classification/phobert_classifier.py`
  - Class `ExpenseClassifier` với method `classify(text) -> (category, confidence)`
  - Upload model lên Hugging Face Hub (public)

**Deliverable tuần 7:**
- ✅ Training dataset 2000+ samples
- ✅ Fine-tuned PhoBERT model trên HF Hub
- ✅ Classification module integrated vào pipeline
- ✅ Evaluation report

**Học gì:** Fine-tuning transformers, Hugging Face Hub, model evaluation

---

## PHASE 3: OPTIMIZATION & POLISH (Tuần 8-10)

**Mục tiêu phase:** Integrate tất cả thành system hoàn chỉnh, tối ưu, handle edge cases, chuẩn bị cho production.

### Tuần 8: Pipeline Integration + FastAPI Backend

**Mục tiêu:** Ghép mọi thứ thành pipeline duy nhất, build API.

**Tasks:**

- [ ] **Build main pipeline orchestrator**
  - `src/pipeline/receipt_pipeline.py`
  - Class `ReceiptPipeline` với method `process(image_path) -> ProcessingResult`
  - Flow: preprocess → extract → validate → classify → merge → return
  - Logging đầy đủ cho từng step
  - Error handling graceful (mỗi step fail → fallback)

- [ ] **Implement fallback strategy**
  - Tier 1: Gemini primary
  - Tier 2: Gemini với alternative prompt
  - Tier 3: PaddleOCR + rule-based (install PaddleOCR)
  - Tier 4: Return với flag "needs_human_review"

- [ ] **Build FastAPI backend**
  - `api/main.py` - FastAPI app
  - Endpoints:
    - `POST /api/v1/extract` - upload ảnh, return result
    - `GET /api/v1/health` - health check
    - `GET /api/v1/stats` - processing stats
    - `POST /api/v1/feedback` - user submit correction
  - Request validation với Pydantic
  - CORS config
  - Rate limiting (slowapi)

- [ ] **Setup SQLite database**
  - `src/database/models.py` với SQLAlchemy
  - Tables: receipts, items, user_feedback
  - Migrations với Alembic

- [ ] **Write integration tests**
  - `tests/test_pipeline.py`
  - Test end-to-end flow với 5 ảnh thật
  - Test error cases

**Deliverable tuần 8:**
- ✅ Pipeline orchestrator hoạt động end-to-end
- ✅ FastAPI backend với 4 endpoints
- ✅ Database schema + migrations
- ✅ Integration tests passing

**Học gì:** System design, FastAPI, async programming, database design

---

### Tuần 9: Streamlit Frontend + Caching + Monitoring

**Mục tiêu:** Build UI đẹp, implement caching, monitoring dashboard.

**Tasks:**

- [ ] **Build Streamlit frontend**
  - `ui/app.py` - main app
  - Features:
    - Upload ảnh (drag & drop)
    - Hiển thị ảnh preview
    - Gọi API extract
    - Hiển thị kết quả dạng form có thể edit
    - Button "Xác nhận" / "Sửa lại" → gửi feedback
    - History của các receipts đã xử lý
  - Styling đẹp, mobile-friendly

- [ ] **Implement caching layer**
  - `src/cache/image_cache.py`
  - Hash ảnh bằng SHA256
  - Cache kết quả vào Redis hoặc SQLite
  - TTL 7 ngày
  - Log cache hit rate

- [ ] **Build monitoring dashboard**
  - `ui/admin_dashboard.py` - Streamlit admin page
  - Hiển thị:
    - Số receipts xử lý theo ngày (chart)
    - Accuracy trung bình
    - Top merchants
    - Cost tracking (Gemini API usage)
    - Fail cases (để review)
    - Processing time distribution

- [ ] **Load testing**
  - Dùng `locust` để test API chịu được ~10 concurrent requests
  - Optimize nếu có bottleneck

**Deliverable tuần 9:**
- ✅ Streamlit frontend hoạt động mượt
- ✅ Caching layer reduce cost ~20-30%
- ✅ Monitoring dashboard
- ✅ Load test report

**Học gì:** Streamlit, caching strategies, monitoring, load testing

---

### Tuần 10: Final Evaluation + Edge Cases + Bug Fixing

**Mục tiêu:** Đo lường cuối cùng, fix mọi bug, optimize cho demo.

**Tasks:**

- [ ] **Chạy evaluation tổng thể**
  - Chạy pipeline trên toàn bộ golden test set
  - So sánh với baseline tuần 3
  - Đo tất cả metrics: field-level, item F1, category acc, end-to-end, latency, cost
  - Tạo bảng so sánh đẹp cho README
  - **Target cuối cùng:**
    - End-to-end success rate: 85%+
    - Nhóm 1: 95%+
    - Nhóm 2: 90%+
    - Nhóm 3: 80%+
    - Nhóm 4: 65%+
    - Nhóm 5: 45%+ (với fallback)

- [ ] **Handle 20 edge cases quan trọng**
  - Ảnh rỗng, ảnh không phải hoá đơn
  - Hoá đơn 2 ngôn ngữ (VN + EN)
  - Hoá đơn có nhiều trang
  - Hoá đơn bị mất góc
  - Số tiền âm (hoàn tiền)
  - Discount/voucher
  - Tip/service charge
  - Hoá đơn bằng USD (hotel, du lịch)
  - Document tất cả trong `docs/edge_cases.md`

- [ ] **Performance optimization**
  - Profile pipeline, tìm bottleneck
  - Optimize image size trước khi gửi API
  - Parallel processing nếu cần
  - Target: < 5s cho mỗi receipt (p95)

- [ ] **Fix all bugs**
  - Chạy full test suite
  - Fix tất cả warnings, errors
  - Code review cho chính mình (clean code, naming, comments)

- [ ] **Security review**
  - API key không hardcode (dùng env vars)
  - Sanitize user input
  - Rate limiting hoạt động
  - File upload validation (size, type)

**Deliverable tuần 10:**
- ✅ Final evaluation report với đầy đủ metrics
- ✅ Edge cases documentation
- ✅ Clean codebase, no bugs, no warnings
- ✅ Performance meets targets

**Học gì:** Evaluation at scale, debugging, performance optimization, security

---

## PHASE 4: DEPLOYMENT & PRESENTATION (Tuần 11-12)

**Mục tiêu phase:** Deploy public, viết documentation xuất sắc, chuẩn bị cho phỏng vấn.

### Tuần 11: Deployment

**Mục tiêu:** Deploy toàn bộ system lên public, ai cũng dùng được.

**Tasks:**

- [ ] **Dockerize application**
  - Viết `Dockerfile` cho backend
  - Viết `Dockerfile` cho frontend
  - `docker-compose.yml` để chạy cả hệ thống
  - Test local với docker-compose

- [ ] **Deploy backend**
  - Option 1: Railway (free tier, dễ) - recommended
  - Option 2: Render (free tier)
  - Option 3: Google Cloud Run (pay as you go, rẻ)
  - Setup environment variables (Gemini API key)
  - Setup custom domain (optional, dùng free subdomain cũng OK)

- [ ] **Deploy frontend lên Hugging Face Spaces**
  - HF Spaces miễn phí, support Streamlit native
  - Setup secrets (API endpoint)
  - Test deployment
  - Đảm bảo có link public chia sẻ được

- [ ] **Setup CI/CD cơ bản**
  - GitHub Actions workflow
  - Trigger: push to main
  - Jobs: run tests → build docker → deploy
  - Đây là điểm cộng lớn với recruiter

- [ ] **Tạo demo data**
  - Chuẩn bị 10 ảnh sample để demo
  - Đảm bảo các ảnh này cover nhiều trường hợp khác nhau
  - Hoặc có thể mở upload cho bất kỳ ai thử

- [ ] **Monitoring production**
  - Setup basic logging (Sentry free tier hoặc simple logs)
  - Track: số requests, errors, latency

**Deliverable tuần 11:**
- ✅ Backend deployed, API public accessible
- ✅ Frontend deployed trên HF Spaces
- ✅ CI/CD pipeline hoạt động
- ✅ Link demo public share được

**Học gì:** Docker, deployment, CI/CD, cloud platforms

---

### Tuần 12: Documentation & Interview Prep

**Mục tiêu:** Viết documentation xuất sắc, blog post, chuẩn bị cho phỏng vấn.

**Tasks:**

- [ ] **Viết README chuyên nghiệp**
  - File `README.md` phải bao gồm:
    - Tên project + badge status
    - Demo GIF/video (dùng asciinema hoặc record screen)
    - Live demo link
    - Problem statement
    - Architecture diagram
    - Tech stack
    - Key features
    - Results table (với metrics)
    - Quick start guide
    - API documentation
    - Limitations & future work
    - Credits & references
  - **Quan trọng:** README phải "sell" project của bạn trong 30 giây đầu

- [ ] **Viết blog post kỹ thuật**
  - Publish trên Viblo (tiếng Việt) VÀ Medium (tiếng Anh)
  - Title gợi ý: "Build hệ thống trích xuất hoá đơn tiếng Việt với VLM và PhoBERT - Bài học từ 12 tuần"
  - Nội dung:
    - Motivation
    - Problem analysis
    - Approach comparison (tại sao chọn hybrid)
    - Key technical decisions với trade-offs
    - Results + benchmarks
    - Challenges faced
    - Lessons learned
  - Độ dài: 2000-3000 từ
  - Include code snippets + diagrams + charts

- [ ] **Viết technical documentation**
  - `docs/architecture.md` - chi tiết architecture
  - `docs/api.md` - API reference
  - `docs/evaluation.md` - evaluation methodology + results
  - `docs/deployment.md` - deployment guide

- [ ] **Record demo video**
  - Video 3-5 phút giới thiệu project
  - Upload lên YouTube
  - Embed vào README
  - Cấu trúc: problem → demo → architecture → results

- [ ] **Chuẩn bị pitch cho phỏng vấn**
  - "Elevator pitch" 30 giây về project
  - Pitch chi tiết 3 phút
  - Prepare câu trả lời cho 15 câu hỏi khả năng cao (xem phần dưới)
  - Practice nói trước gương hoặc record lại

- [ ] **Polish final touches**
  - Check lại tất cả links
  - Screenshot đẹp cho README
  - LinkedIn post announce project
  - Update CV với project này ở vị trí nổi bật

**Deliverable tuần 12:**
- ✅ README xuất sắc
- ✅ Blog post published (Viblo + Medium)
- ✅ Full documentation
- ✅ Demo video trên YouTube
- ✅ LinkedIn post
- ✅ CV updated
- ✅ Interview prep notes

**Học gì:** Technical writing, storytelling, self-marketing

---

## Các câu hỏi phỏng vấn khả năng cao (prepare trước)

1. "Giới thiệu project của em trong 1 phút"
2. "Tại sao em chọn approach hybrid thay vì fine-tune Donut?"
3. "Làm sao em đảm bảo VLM đọc đúng hoá đơn phức tạp?"
4. "Nếu phải scale lên 10,000 receipts/ngày thì em optimize thế nào?"
5. "Trade-off giữa accuracy và cost trong project của em?"
6. "Em đã gặp bug/thử thách gì lớn nhất và giải quyết ra sao?"
7. "Tại sao dùng PhoBERT mà không dùng LLM cho classification?"
8. "Em đo accuracy như thế nào? Metrics nào quan trọng nhất?"
9. "Nếu Gemini API down thì system của em có chạy được không?"
10. "Em handle hoá đơn viết tay thế nào?"
11. "Dataset của em có bao nhiêu ảnh? Build như thế nào?"
12. "Em đã thử những prompt engineering technique nào?"
13. "System design của em có gì đặc biệt?"
14. "Nếu có 1 tháng nữa, em sẽ cải thiện gì?"
15. "Em học được gì lớn nhất từ project này?"

---

## Milestones tổng quan

| Cuối tuần | Milestone |
|-----------|-----------|
| 3 | Có baseline + golden test set + error analysis |
| 7 | Pipeline core hoàn chỉnh, accuracy ~88% |
| 10 | System production-ready, full evaluation |
| 12 | Deployed + documented + ready for interview |

---

## Time budget ước tính

- **Tổng thời gian:** ~200 giờ (12 tuần × ~17 giờ/tuần)
- **Breakdown:**
  - Learning & research: 20%
  - Coding: 45%
  - Data work (collect, label): 15%
  - Documentation & writing: 10%
  - Debugging & optimization: 10%

---

## Risk management

**Risk 1: Không đủ data**
- Mitigation: Bổ sung bằng MC-OCR dataset, SROIE

**Risk 2: Gemini API hết free tier**
- Mitigation: Chuẩn bị budget ~$20-30, hoặc dùng Qwen2-VL self-host backup

**Risk 3: PhoBERT accuracy thấp**
- Mitigation: Augment data, thử XLM-R làm alternative

**Risk 4: Bị chậm tiến độ**
- Mitigation: Mỗi tuần review, nếu chậm thì cắt scope (skip caching, skip CI/CD) nhưng không skip evaluation và deployment

**Risk 5: Deploy thất bại**
- Mitigation: Luôn có local demo backup, record video demo từ sớm

---

## Resources & Links hữu ích

**Documentation:**
- Gemini API: https://ai.google.dev/docs
- PhoBERT: https://github.com/VinAIResearch/PhoBERT
- FastAPI: https://fastapi.tiangolo.com
- Streamlit: https://docs.streamlit.io

**Datasets:**
- MC-OCR 2021: https://aihub.ml/competitions/1
- SROIE: https://rrc.cvc.uab.es/?ch=13

**Learning:**
- HuggingFace NLP Course: https://huggingface.co/learn/nlp-course
- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
- Docker basics: https://docs.docker.com/get-started/

**Deployment:**
- Railway: https://railway.app
- Hugging Face Spaces: https://huggingface.co/spaces
- Google Cloud Run: https://cloud.google.com/run

---

## Final checklist trước khi đi phỏng vấn

- [ ] Project có live demo link hoạt động
- [ ] GitHub repo public, README xuất sắc
- [ ] Blog post đã publish (có ít nhất 100 views)
- [ ] Demo video trên YouTube
- [ ] Model trên Hugging Face Hub
- [ ] CV có link đến project
- [ ] LinkedIn có post về project
- [ ] Có thể giải thích mọi technical decision
- [ ] Có thể demo live trong 5 phút
- [ ] Prepare 15 câu Q&A
- [ ] Đã practice pitch 3 lần

---