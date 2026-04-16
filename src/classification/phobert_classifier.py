"""
src/classification/phobert_classifier.py
------------------------------------------
Expense category classifier backed by fine-tuned PhoBERT.

Architecture:
  - Base: vinai/phobert-base (pre-trained Vietnamese BERT)
  - Head: linear classification layer → 10 expense categories
  - Input: "{merchant_name} | {item1}, {item2}, ..." (max 128 tokens)
  - Fine-tuned on ~2000+ labeled Vietnamese receipts

Fallback strategy:
  If PhoBERT model is not loaded (not yet trained, or HF Hub unavailable),
  falls back to KeywordClassifier — rule-based, ~70% accuracy, instant.

Usage:
    clf = ExpenseClassifier()                    # keyword fallback only
    clf = ExpenseClassifier.from_pretrained()    # load from HF Hub (needs training first)
    result = clf.classify("Highlands Coffee", ["Cà phê sữa", "Bánh mì"])
    print(result.category, result.confidence)    # "Ăn uống", 0.97

Training:
    See scripts/generate_training_data.py to build training CSV.
    See notebooks/02_finetune_phobert.ipynb for the Colab training notebook.
    Upload fine-tuned model to HF Hub, then:
        clf = ExpenseClassifier.from_pretrained("your-hf-username/phobert-expense-v1")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .keyword_classifier import CATEGORIES, CATEGORY_INDEX, KeywordClassifier

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_INPUT_TOKENS = 128
MAX_ITEM_COUNT   = 10    # truncate long item lists
PHOBERT_CONFIDENCE_THRESHOLD = 0.50  # below this → escalate to LLM fallback


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Output of ExpenseClassifier.classify()."""

    category: str
    confidence: float
    method: str                          # "phobert" | "keyword" | "default"
    all_scores: dict[str, float] = field(default_factory=dict)

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < PHOBERT_CONFIDENCE_THRESHOLD


# ── Main classifier ───────────────────────────────────────────────────────────

class ExpenseClassifier:
    """
    Vietnamese expense classifier: PhoBERT + keyword fallback.

    The class is designed for two operating modes:
      A) Production (fine-tuned model available):
            clf = ExpenseClassifier.from_pretrained("username/phobert-expense-v1")
      B) Development / fallback (no trained model yet):
            clf = ExpenseClassifier()
            # Uses keyword rules (~70% accuracy) — sufficient for pipeline integration
    """

    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._model: Any = None
        self._model_name: str | None = None
        self._keyword = KeywordClassifier()

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "lamhoangphuc/phobert-expense-v1",
        device: str = "cpu",
    ) -> "ExpenseClassifier":
        """
        Load fine-tuned PhoBERT from Hugging Face Hub or local path.

        Falls back to keyword classifier if loading fails (e.g., model not yet
        uploaded, no internet, or HF Hub unavailable).

        Parameters
        ----------
        model_name_or_path : HF Hub repo ID or local directory path.
        device             : 'cpu' | 'cuda' | 'mps'
        """
        instance = cls()
        instance._try_load(model_name_or_path, device)
        return instance

    def _try_load(self, model_name_or_path: str, device: str) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            log.info("Loading PhoBERT from %s ...", model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            model.eval()
            model.to(device)

            self._tokenizer = tokenizer
            self._model = model
            self._model_name = model_name_or_path
            self._device = device
            log.info("PhoBERT loaded successfully (%s)", model_name_or_path)

        except Exception as exc:
            log.warning(
                "Could not load PhoBERT from '%s': %s — falling back to keyword classifier",
                model_name_or_path, exc,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(
        self,
        merchant_name: str | None,
        item_names: list[str] | None = None,
    ) -> ClassificationResult:
        """
        Classify a receipt into an expense category.

        Parameters
        ----------
        merchant_name : Store/merchant name from extraction (may be None).
        item_names    : List of item names from extraction (may be empty).

        Returns
        -------
        ClassificationResult with category, confidence, and method used.
        """
        text = self._build_input(merchant_name, item_names or [])

        if self._model is not None:
            return self._phobert_classify(text)
        return self._keyword_classify(merchant_name, item_names or [])

    def classify_batch(
        self,
        receipts: list[tuple[str | None, list[str]]],
    ) -> list[ClassificationResult]:
        """
        Classify multiple receipts. PhoBERT batches internally for efficiency.
        Falls back to per-item keyword classification if model not loaded.
        """
        if self._model is not None:
            return self._phobert_classify_batch(receipts)
        return [self._keyword_classify(m, items) for m, items in receipts]

    @property
    def using_phobert(self) -> bool:
        """True if PhoBERT model is loaded; False if using keyword fallback."""
        return self._model is not None

    # ── Input formatting ──────────────────────────────────────────────────────

    @staticmethod
    def _build_input(merchant_name: str | None, item_names: list[str]) -> str:
        """
        Format receipt fields into a single classifier input string.

        Format: "{merchant} | {item1}, {item2}, ..."
        Truncated to MAX_ITEM_COUNT items to stay within token budget.

        Examples:
            "Highlands Coffee | Cà phê sữa, Bánh mì"
            "KFC | Gà rán, Khoai tây chiên, Pepsi"
            "Unknown | Thuốc paracetamol, Vitamin C"
        """
        parts: list[str] = []
        if merchant_name:
            parts.append(merchant_name.strip())
        if item_names:
            truncated = item_names[:MAX_ITEM_COUNT]
            parts.append(", ".join(n.strip() for n in truncated if n.strip()))
        return " | ".join(parts) if parts else ""

    # ── PhoBERT inference ─────────────────────────────────────────────────────

    def _phobert_classify(self, text: str) -> ClassificationResult:
        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return ClassificationResult(
            category=CATEGORIES[best_idx],
            confidence=round(probs[best_idx], 4),
            method="phobert",
            all_scores={cat: round(p, 4) for cat, p in zip(CATEGORIES, probs)},
        )

    def _phobert_classify_batch(
        self,
        receipts: list[tuple[str | None, list[str]]],
    ) -> list[ClassificationResult]:
        import torch
        import torch.nn.functional as F

        texts = [self._build_input(m, items) for m, items in receipts]

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()

        results = []
        for row in probs:
            best_idx = max(range(len(row)), key=lambda i: row[i])
            results.append(ClassificationResult(
                category=CATEGORIES[best_idx],
                confidence=round(row[best_idx], 4),
                method="phobert",
                all_scores={cat: round(p, 4) for cat, p in zip(CATEGORIES, row)},
            ))
        return results

    # ── Keyword fallback ──────────────────────────────────────────────────────

    def _keyword_classify(
        self,
        merchant_name: str | None,
        item_names: list[str],
    ) -> ClassificationResult:
        category, confidence = self._keyword.classify(merchant_name, item_names)
        return ClassificationResult(
            category=category,
            confidence=confidence,
            method="keyword",
        )
