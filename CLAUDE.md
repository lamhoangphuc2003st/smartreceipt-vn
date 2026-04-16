---
name: smartreceipt-vn
description: Guidance for building and maintaining SmartReceipt VN - a Vietnamese receipt understanding system that extracts structured data and classifies expenses from receipt images using Gemini Vision API and fine-tuned PhoBERT. Use this skill whenever the user is working on receipt extraction, OCR for Vietnamese documents, VLM prompt engineering for structured output, fine-tuning PhoBERT for classification, handling edge cases in Vietnamese receipts (VAT invoices, handwritten bills, long supermarket receipts), evaluating extraction accuracy, or making architectural decisions for the smartreceipt-vn project. Also trigger when the user mentions "hoá đơn", "receipt pipeline", "Gemini Vision", "PhoBERT classifier", or is editing any file under the smartreceipt-vn repository.
---

# SmartReceipt VN Development Skill

A skill for making consistent, high-quality technical decisions when building SmartReceipt VN - a Vietnamese receipt extraction and expense classification system.

## What this project is

SmartReceipt VN takes a photo of a Vietnamese receipt (grocery, restaurant, VAT invoice, handwritten bill, etc.) and returns structured data (merchant, date, items, total) plus an expense category (Ăn uống, Đi lại, Mua sắm...). The system is designed as a portfolio project demonstrating production-grade AI Engineering skills for fresher/intern interviews.

**Core architecture:** Hybrid approach combining a Vision-Language Model (Gemini) for image-to-JSON extraction and a fine-tuned PhoBERT classifier for expense categorization, wrapped in a FastAPI backend with Streamlit frontend.

## The six core principles

These principles guide every technical decision. When in doubt, return to these.

### 1. Hybrid over monolithic

Use the right tool for each subtask. VLM handles vision + layout understanding (hard). PhoBERT handles text classification (easy, cheap, controllable). Never use a single model for everything just because it's simpler — the cost/latency/accuracy trade-offs matter. When adding a new capability, first ask: "Is this a vision problem or a text problem?" and pick the corresponding tool.

### 2. Production-ready over research-y

This is an *AI Engineer* portfolio project, not a research project. Prioritize reliability, cost-awareness, error handling, and deployability over novel techniques. A boring solution that works beats a fancy solution that's flaky. When choosing between two approaches, pick the one that's easier to operate in production.

### 3. Data-driven decisions, always

Never claim an improvement without measuring it on the golden test set. Every prompt change, every preprocessing tweak, every model update must be benchmarked against `data/golden_test_set.json` before merging. If you can't measure it, don't ship it.

### 4. Fail gracefully with tiered fallbacks

The system must never crash. Design every component with a fallback. Gemini down? Try alternative prompt. That fails? Use PaddleOCR. That fails? Flag for human review. Return *something* useful, always. This reliability engineering mindset is the single most impressive thing for interviews.

### 5. Honesty about limitations

Don't inflate accuracy numbers. Don't claim to handle cases you can't. Document known failures in `docs/edge_cases.md`. Interviewers can smell exaggeration immediately — honest numbers with clear error analysis beat inflated numbers every time.

### 6. Explain decisions, not just code

Every non-trivial technical choice must have a written rationale (in code comments, ADR documents, or the blog post). The question "why did you choose X over Y?" will come up in every interview — the rationale should already exist in the repo.

## Architecture at a glance

```
User uploads image
       ↓
[Preprocessing] → deskew, document detect, enhance
       ↓
[VLM Extractor] → Gemini Vision API with structured output
       ↓
[Validator] → schema check, math check, auto-correction
       ↓
[Classifier] → PhoBERT fine-tuned on ~2000 VN receipts
       ↓
[Merger] → combine extraction + classification + metadata
       ↓
Structured JSON result
```

Layer boundaries matter. Never mix concerns across layers. Preprocessing doesn't know about categories. The classifier doesn't know about images. This separation is what makes the system debuggable and testable.

## Working on specific components

### When editing preprocessing (`src/preprocessing/`)

Preprocessing exists to make VLM's job easier, nothing more. Don't try to do OCR here, don't try to extract text here. The only job is: take a messy photo, output a clean image ready for the VLM.

**Standard pipeline order:** orientation fix → document detect & crop → perspective warp → denoise → contrast enhance → resize. Don't reorder without a reason — each step assumes the previous step ran.

**Critical rule:** Preprocessing must be *idempotent and safe*. If the image is already clean, preprocessing should return something equivalent, not worse. Always test with both good and bad images. An aggressive filter that ruins clean images is a bug, not a feature.

**When to add a new preprocessing step:** Only if you can show it improves accuracy on the golden set by ≥1% without regressing any subgroup. Run the benchmark before and after. Log it in `docs/experiments/`.

### When writing or editing VLM prompts (`src/extraction/prompts.py`)

This is the most impactful component. A 5% prompt improvement beats a week of model fine-tuning effort. Take prompt engineering seriously.

**Prompt structure to follow:**

1. **Role definition** — "You are a Vietnamese accounting expert with experience reading receipts including POS, VAT invoices, and handwritten bills."
2. **Task description** — Clear, specific, one task per prompt.
3. **Output schema** — Always use Pydantic + Gemini structured output, never rely on prompt-only JSON.
4. **Chain-of-thought steps** — Force the model to reason: identify receipt type → find merchant → extract items line by line → validate math → output JSON.
5. **Few-shot examples** — 2-3 examples covering the hardest cases (not the easiest).
6. **Explicit constraints** — "Keep Vietnamese diacritics", "Return integer VND, no commas", "Return null if unreadable, never guess".
7. **Self-check instruction** — "Before returning, verify sum(items) + vat = total".

**Prompt iteration discipline:** Every prompt change gets a version number. Save old versions. Benchmark new vs old on the golden set. Only promote to production if accuracy improves *and* no subgroup regresses. Document the change in `docs/experiments/exp_XXX_prompt_vN.md`.

**Anti-hallucination rules** — always include in prompts:
- "Do not invent items not visible in the image"
- "If a field is unclear, return null — never guess"
- "If the image is not a receipt, return {\"error\": \"not_a_receipt\"}"

**Temperature = 0, always.** Extraction is not a creative task. Deterministic output is non-negotiable.

### When working on the VLM extractor (`src/extraction/gemini_extractor.py`)

This wrapper must be production-grade. Never call the Gemini SDK directly from anywhere else — always go through this wrapper.

**Required features in the wrapper:**
- Retry with exponential backoff (3 attempts: 1s, 2s, 4s)
- Timeout (30s hard limit per request)
- Error classification (transient vs permanent — only retry transient)
- Cost tracking (log tokens + estimated USD per call)
- Response caching by image hash (SHA256) with 7-day TTL
- Structured output enforcement via Pydantic schema
- Graceful degradation signal (return `ExtractionResult` with `success=False` + reason, never raise to caller)

**When Gemini returns suspicious output:** Don't silently accept. Run it through the validator. If validation fails, retry once with a "corrective" prompt that shows the model what was wrong. If still fails, mark as low-confidence and let the fallback tier handle it.

### When working on validation (`src/validation/`)

The validator is the safety net. Its job is to catch VLM mistakes before they reach the user.

**Validation must include:**
- Schema validation (Pydantic)
- Math consistency: `abs(sum(items.total) + vat - total_amount) <= 1` (allow 1 VND rounding)
- Date sanity: within [2020-01-01, today + 1 day]
- Currency sanity: each item total < 100M VND, total < 1B VND (flag anomalies, don't reject)
- Merchant name fuzzy match against `data/merchants.json` database (correct common OCR errors)

**Correction vs rejection philosophy:** Prefer correction over rejection. If the VLM misreads "Higlands" as the merchant, fuzzy-match it to "Highlands Coffee" and log the correction. Only mark as error if the data is clearly unusable.

**Never silently modify data.** Every auto-correction must be logged with the original value, the corrected value, and the reason. Users should be able to see what was changed.

### When fine-tuning or updating PhoBERT (`src/classification/`)

PhoBERT is trained once per release. Don't fine-tune ad-hoc.

**Training protocol:**
- Base model: `vinai/phobert-base` (not base-v2 unless benchmarked first)
- Input format: `f"{merchant_name} | {', '.join(item_names[:10])}"` (truncate long item lists)
- Max length: 128 tokens (receipts are short)
- Learning rate: 2e-5, batch size: 32, epochs: 5 with early stopping
- Split: 70% train / 15% val / 15% test, stratified by category
- Always report per-class F1, never just overall accuracy (class imbalance is real here)

**When retraining:** The new model must beat the old one on the held-out test set *and* on a held-out set of real receipts from the golden set. Log both numbers. Version the model: `phobert-expense-v1.2` etc. Upload to Hugging Face Hub.

**Category taxonomy is stable.** Don't add or remove categories without a migration plan for historical data. The 10 categories (Ăn uống, Đi lại, Mua sắm, Giải trí, Hoá đơn tiện ích, Sức khoẻ, Giáo dục, Du lịch, Nhà cửa, Khác) are fixed unless there's a strong product reason.

### When editing the pipeline orchestrator (`src/pipeline/receipt_pipeline.py`)

The orchestrator glues components together. Keep it thin — no business logic, only flow control and error handling.

**Required behavior:**
- Log every stage entry/exit with timing
- Each stage wrapped in try/except with graceful fallback
- Return `ProcessingResult` object, never raise to the API layer
- Include metadata: which model versions were used, processing time, whether any fallback tier was triggered

**Golden rule:** If a new orchestrator version can't process every image in the golden set without crashing, it's not ready to merge. Crashes are worse than wrong answers.

## Evaluation methodology — how to know if a change is good

Every change that touches extraction, validation, or classification must be evaluated against the golden test set before merging.

**Standard evaluation command:**
```bash
python -m src.evaluation.run_eval --pipeline-version <new> --output docs/experiments/exp_XXX.md
```

**Metrics to report for every experiment:**
- Field-level accuracy per field (merchant, date, total, items)
- Item F1 score (precision and recall separately)
- Category accuracy
- End-to-end success rate (all critical fields correct)
- Accuracy per difficulty group (1-5)
- p50 / p95 latency
- Cost per receipt (USD)

**A change is "good" only if:**
- End-to-end success rate improves by ≥1 percentage point, AND
- No difficulty group regresses by more than 2 percentage points, AND
- p95 latency stays under 5 seconds, AND
- Cost per receipt doesn't increase by more than 20% (unless justified)

If any condition fails, the change is not merged without explicit rationale in the experiment document.

**The golden test set is sacred.** Never train on it. Never prompt-engineer while looking at it (that's overfitting). Only use it for final evaluation. If you need to iterate on a specific subset, create a separate dev set.

## Data handling conventions

**Raw images:** `data/raw/<difficulty_group>/<timestamp>_<merchant>.jpg`. Never edit raw files. If you need a processed version, save it elsewhere.

**Golden test set:** `data/golden_test_set.json`. Append-only. If you find a labeling error, fix it in a separate commit with clear justification. Never silently rewrite history.

**Training data for PhoBERT:** `data/training/phobert_classifier.csv`. Every sample must have a source label (manual, auto-labeled-verified, auto-labeled-unverified). Only `manual` and `auto-labeled-verified` are used for final training.

**Privacy:** Blur or remove personal info (addresses, phone numbers, personal tax codes) from any image before committing. Use `data/raw/` in `.gitignore` — never push raw receipts to GitHub.

**Synthetic data:** Allowed for augmentation but must be labeled clearly as synthetic. Never mix with real data in the golden set.

## Anti-patterns — things to avoid

These are mistakes that would hurt the project or the interview narrative.

**Don't fine-tune Donut or LayoutLMv3 "just to have it".** It was considered and rejected: VLM gives better accuracy per unit effort at this scale. Only revisit if there's concrete evidence VLM is insufficient.

**Don't use LLM for classification.** It was considered and rejected: PhoBERT is 100x cheaper and faster, with comparable accuracy. Only use LLM classification as a fallback when PhoBERT confidence is below 0.5.

**Don't add features without evaluation.** Every feature must be benchmarked. "I think this will help" is not evidence. Run the eval.

**Don't skip error handling because "it usually works".** Production reliability is the main interview story. Every external call needs retry + timeout + fallback.

**Don't over-optimize for the golden set.** If a prompt works great on the 100 golden images but fails on new real-world images, it's overfit. Periodically test on fresh unseen images.

**Don't hide failures.** If the system fails on 15% of handwritten receipts, document it clearly. Interviewers respect honesty; they distrust polished numbers without caveats.

**Don't skip documentation.** Code without rationale documents is half-done. Every experiment, every decision, every trade-off needs a written record.

## Common tasks — quick workflows

### Adding a new preprocessing step

1. Create `src/preprocessing/<step_name>.py` with a single function `apply(image) -> image`
2. Add unit test in `tests/test_preprocessing.py` with at least 3 cases (clean image, target case, edge case)
3. Insert into pipeline in the correct position (see "Standard pipeline order" above)
4. Run golden set benchmark before and after
5. If accuracy improves, merge. If not, document why it was tried in `docs/experiments/` and revert.

### Improving a prompt

1. Copy current prompt to a new version in `src/extraction/prompts.py` (e.g., `EXTRACTION_PROMPT_V3`)
2. Make changes to the new version, leaving the old one intact
3. Run A/B benchmark on golden set
4. If new version wins on all metrics, update the default import
5. Document the change in `docs/experiments/exp_XXX_prompt_vN.md` with: hypothesis, change, result, decision

### Debugging a specific failing receipt

1. Add the image to `data/debug/` (not golden set)
2. Run pipeline with `--debug` flag to log intermediate outputs
3. Identify which stage the error happens at (preprocessing? extraction? validation?)
4. Don't "fix" by special-casing the image. Find the general pattern it represents
5. Add a similar case to the golden set (if a real pattern) and fix at the general level

### Deploying a new version

1. All tests pass (`pytest`)
2. Golden set benchmark meets release criteria (see "Evaluation methodology")
3. Update version in `pyproject.toml` and `CHANGELOG.md`
4. Tag the commit: `git tag v0.X.Y`
5. CI/CD deploys automatically
6. Verify live demo works with 3 sample receipts
7. Update blog post / README if public-facing changes

## References

When you need more depth, look at these documents in the repo (create them if missing):

- `docs/architecture.md` — Detailed system architecture with diagrams
- `docs/experiments/` — All benchmarking experiments, one file per change
- `docs/edge_cases.md` — Known failure modes and how the system handles them
- `docs/api.md` — FastAPI endpoint reference
- `docs/evaluation.md` — Full evaluation methodology and current metrics
- `docs/weekly_log.md` — Development journal (useful for interview storytelling)
- `data/merchants.json` — Known Vietnamese merchant database
- `README.md` — Public-facing project documentation

## Interview preparation notes

When the user asks for help preparing for interviews about this project, remind them of the key talking points this architecture supports:

- **Pragmatic tool selection** — "I used VLM for the hard vision problem and PhoBERT for the easy text problem because..."
- **Cost-awareness** — "Running classification through LLM would cost 100x more at scale, so..."
- **Reliability engineering** — "I designed 4-tier fallback because in production, you never want hard failures..."
- **Honest evaluation** — "Accuracy is 85% overall, but drops to 65% for handwritten receipts — here's the error analysis..."
- **Data-driven iteration** — "I ran 23 experiments, each logged — here's how prompt v3 improved accuracy 8 points..."

The project is designed to support these narratives. Keep development aligned with them.

---

**Final reminder:** This project is ultimately a demonstration of *engineering judgment*, not ML virtuosity. Every decision — architecture, code, documentation — should reinforce the image of a thoughtful, pragmatic AI engineer who ships reliable systems. When a decision is ambiguous, ask: "which choice makes for a better interview story?" That's usually the right one.
