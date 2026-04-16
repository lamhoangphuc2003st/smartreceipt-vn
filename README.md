# SmartReceipt VN 🧾

> Automatic extraction and expense categorization from Vietnamese receipt photos.

**Live demo:** [huggingface.co/spaces/lamhoangphuc2003st/smartreceipt-vn](https://huggingface.co/spaces/lamhoangphuc2003st/smartreceipt-vn) · **API:** [smartreceipt-vn-production.up.railway.app](https://smartreceipt-vn-production.up.railway.app/docs)

<!-- TODO: Add demo GIF here -->
<!-- ![Demo](docs/demo.gif) -->

---

## What it does

Upload a photo of any Vietnamese receipt — POS receipt, VAT invoice, handwritten bill, e-ticket, or bank transfer confirmation — and get back structured JSON:

```json
{
  "merchant_name": "Highlands Coffee",
  "date": "2026-04-15",
  "total_amount": 125000,
  "payment_method": "cash",
  "items": [
    { "name": "Trà sữa trân châu", "quantity": 1, "unit_price": 65000, "total": 65000 },
    { "name": "Bánh mì", "quantity": 2, "unit_price": 30000, "total": 60000 }
  ],
  "classification": {
    "category": "Ăn uống",
    "confidence": 0.97,
    "method": "phobert"
  }
}
```

10 expense categories: Ăn uống · Đi lại · Mua sắm · Giải trí · Hoá đơn tiện ích · Sức khoẻ · Giáo dục · Du lịch · Nhà cửa · Khác

---

## Architecture

```
Receipt photo
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  OpenCV: deskew · document detect · perspective warp
│  (src/preprocessing) │  denoise · contrast enhance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VLM Extractor  │  Gemini Vision API + Pydantic structured output
│  (src/extraction)│  Temperature=0 · retry + backoff · response cache
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validator     │  Schema check · math consistency · date sanity
│  (src/validation)│  Fuzzy merchant matching · auto-correction logging
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Classifier    │  PhoBERT fine-tuned on ~2 000 VN receipts
│ (src/classification)│  Keyword fallback when confidence < 0.5
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │  Railway · rate limiting · /health · /docs
│   (api/)        │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Streamlit UI  │  Hugging Face Spaces · Docker
│  (hf-space/)    │
└─────────────────┘
```

**Design rationale:** Gemini handles the hard vision + layout understanding problem; PhoBERT handles text classification (100× cheaper than LLM at inference time, comparable accuracy). The system never raises exceptions to the caller — every component has a fallback tier.

---

## Evaluation results

Evaluated on 100-image golden test set across 5 difficulty groups.

### Full pipeline (Gemini + PhoBERT)

| Metric | Score |
|--------|-------|
| End-to-end success | **70.0%** |
| Merchant accuracy | 84.0% |
| Date accuracy | 89.0% |
| Total amount accuracy | **93.0%** |
| Items F1 | **88.4%** |
| Category accuracy | 83.0% |
| Latency p50 | 198 ms |
| Latency p95 | 9.8 s |

*End-to-end = all critical fields correct simultaneously. Field-level accuracy is higher because each field is evaluated independently.*

### By difficulty group

| Group | N | E2E | Merchant | Date | Total | Items F1 | Category |
|-------|---|-----|----------|------|-------|----------|----------|
| POS clean | 26 | 84.6% | 88.5% | 96.2% | 100% | 83.8% | 80.8% |
| VAT invoice | 17 | 58.8% | 70.6% | 88.2% | 94.1% | 97.5% | 64.7% |
| Long invoice | 16 | 68.8% | 87.5% | 93.8% | 87.5% | 96.6% | 100% |
| Handwritten | 11 | 36.4% | 81.8% | 63.6% | 81.8% | 87.9% | 81.8% |
| Low quality | 30 | 76.7% | 86.7% | 90.0% | 93.3% | 83.1% | 86.7% |

### PhoBERT classifier (isolated)

| Metric | Score |
|--------|-------|
| Category accuracy | **91.0%** |
| Latency p50 | 87 ms |
| Latency p95 | 159 ms |

### Known limitations

| Limitation | Impact | Root cause |
|------------|--------|------------|
| Handwritten receipts | 36.4% E2E | Gemini struggles with non-standard date formats and irregular handwriting |
| VAT invoice merchant | 70.6% accuracy | Long legal company names not in merchant database |
| p95 latency 9.8s | Demo only | Gemini API network latency; p50 is 198 ms |
| "Khác" category | 14.3% accuracy | Catch-all category is inherently ambiguous |

These limitations are documented honestly in [`docs/edge_cases.md`](docs/edge_cases.md). Inflating numbers would be easy; shipping a maintainable system is harder.

---

## Quick start

### Option A — Use the live demo

Open [huggingface.co/spaces/lamhoangphuc2003st/smartreceipt-vn](https://huggingface.co/spaces/lamhoangphuc2003st/smartreceipt-vn), upload any receipt photo, done.

### Option B — Run locally with Docker Compose

```bash
git clone https://github.com/lamhoangphuc2003st/smartreceipt-vn.git
cd smartreceipt-vn
cp .env.example .env
# Edit .env: add your Gemini API key
docker compose up --build
```

- API: http://localhost:8000/docs
- UI:  http://localhost:8501

### Option C — Run without Docker

```bash
pip install -e ".[standard]"
# or: pip install fastapi uvicorn google-genai opencv-python-headless \
#         Pillow transformers "torch==2.4.0+cpu" --extra-index-url \
#         https://download.pytorch.org/whl/cpu

export Gemini_API_Key=your_key_here
uvicorn api.main:app --reload
```

---

## API reference

Base URL: `https://smartreceipt-vn-production.up.railway.app`

### `POST /receipts/extract`

Upload a receipt image, get structured data back.

```bash
curl -X POST https://smartreceipt-vn-production.up.railway.app/receipts/extract \
  -F "file=@receipt.jpg"
```

**Response schema:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether extraction succeeded |
| `merchant_name` | str \| null | Store or company name |
| `date` | str \| null | Transaction date (YYYY-MM-DD) |
| `total_amount` | int \| null | Total in VND (integer) |
| `payment_method` | str \| null | cash / card / transfer / e_wallet |
| `items` | list | Line items with name, quantity, unit_price, total |
| `classification` | object | category, confidence, method, all_scores |
| `corrections` | list | Auto-corrections made with original/corrected/reason |
| `meta` | object | Timing, cost, model versions, cache hit |

### `GET /health`

Returns `{"status": "ok", "classifier": "phobert"}` when the service is up.

Full interactive docs: [/docs](https://smartreceipt-vn-production.up.railway.app/docs)

---

## Project structure

```
smartreceipt-vn/
├── api/                    # FastAPI application
│   ├── main.py             # App factory, middleware
│   ├── routes/receipts.py  # POST /receipts/extract
│   └── models.py           # Pydantic request/response models
├── src/
│   ├── preprocessing/      # OpenCV image preprocessing pipeline
│   ├── extraction/         # Gemini wrapper, prompts, schemas
│   ├── validation/         # Schema + math + date validators
│   ├── classification/     # PhoBERT + keyword fallback classifier
│   ├── pipeline/           # Orchestrator (glues components together)
│   └── evaluation/         # Metrics and golden set runner
├── tests/                  # pytest test suite (~60 tests)
├── docs/
│   ├── experiments/        # 9 experiment logs (prompt iterations, model evals)
│   ├── edge_cases.md       # 9 documented failure patterns
│   └── deployment.md       # Railway + HF Spaces deployment guide
├── data/
│   ├── golden_test_set.json # 100-image evaluation set (never train on this)
│   └── merchants.json       # ~65 known Vietnamese merchants for fuzzy matching
├── hf-space/               # Hugging Face Spaces frontend (Streamlit + Docker)
├── Dockerfile              # Multi-stage build for Railway backend
├── docker-compose.yml      # Local development (API + UI)
└── .github/workflows/ci.yml # CI: test → docker build → health check
```

---

## Development

### Run tests

```bash
DISABLE_PHOBERT=1 Gemini_API_Key=dummy pytest tests/ -v
```

### Run evaluation against golden test set

```bash
python -m src.evaluation.run_eval --pipeline-version full --output docs/experiments/exp_010.md
```

### Experiment log

| Exp | Change | E2E before | E2E after | Decision |
|-----|--------|-----------|-----------|----------|
| 001 | Naive baseline (keyword only) | — | 12% | Baseline |
| 002 | Add preprocessing (deskew + enhance) | 12% | 38% | Merged |
| 003 | Preprocessing v2 (document detect) | 38% | 51% | Merged |
| 004 | Preprocessing v3 (perspective warp) | 51% | 58% | Merged |
| 005 | Gemini extraction v1 | 58% | 64% | Merged |
| 006 | Prompt v2 (chain-of-thought + few-shot) | 64% | 68% | Merged |
| 007 | Prompt v3 (anti-hallucination + self-check) | 68% | 70% | Merged |
| 008 | PhoBERT fine-tuned v1 | 70% | 70% | Merged (classifier: 83%→91%) |
| 009 | Full pipeline final eval | — | 70% | Current baseline |

---

## Tech stack

| Component | Technology | Why |
|-----------|------------|-----|
| Vision extraction | Gemini Vision API | Best accuracy/cost ratio for Vietnamese receipts at this scale |
| Expense classification | PhoBERT (vinai/phobert-base) | 100× cheaper than LLM, 91% accuracy, 87ms p50 latency |
| Image preprocessing | OpenCV | Deterministic, no GPU needed, battle-tested |
| API | FastAPI + Pydantic | Type safety, auto-docs, async support |
| Frontend | Streamlit | Rapid prototyping, good for demos |
| Backend hosting | Railway | Auto-deploy from GitHub, free tier adequate |
| Frontend hosting | Hugging Face Spaces | Free Docker hosting, good for ML demos |
| CI/CD | GitHub Actions | Test + Docker build validation on every push |

---

## Deployment

See [`docs/deployment.md`](docs/deployment.md) for full step-by-step guides for Railway, Hugging Face Spaces, and GitHub Actions setup.

**Environment variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `Gemini_API_Key` | Yes | Google AI Studio API key |
| `PHOBERT_MODEL` | No | HF Hub repo ID (default: `lamhoangphuc2003st/phobert-expense-v1`) |
| `PORT` | No | Server port (Railway sets automatically) |

---

## About

Built as a portfolio project demonstrating production-grade AI Engineering skills:

- **Hybrid architecture** — right tool for each subtask (VLM for vision, fine-tuned BERT for text)
- **Reliability engineering** — tiered fallbacks, never crashes, every external call has retry + timeout
- **Data-driven iteration** — 9 experiments, each benchmarked against the same golden test set
- **Honest evaluation** — limitations documented clearly, numbers not inflated

**Model:** PhoBERT fine-tuned model on Hugging Face Hub: [`lamhoangphuc2003st/phobert-expense-v1`](https://huggingface.co/lamhoangphuc2003st/phobert-expense-v1)

---

*Portfolio project by [Lam Hoang Phuc](https://github.com/lamhoangphuc2003st) — AI Engineering fresher*
