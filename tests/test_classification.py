"""
tests/test_classification.py
------------------------------
Unit tests for the expense classification module.

All tests use keyword classifier (no model download required).
PhoBERT-specific paths are tested with a mock to avoid HF Hub dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.classification.keyword_classifier import CATEGORIES, CATEGORY_INDEX, KeywordClassifier
from src.classification.phobert_classifier import (
    PHOBERT_CONFIDENCE_THRESHOLD,
    ClassificationResult,
    ExpenseClassifier,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def keyword_clf() -> KeywordClassifier:
    return KeywordClassifier()


@pytest.fixture
def expense_clf() -> ExpenseClassifier:
    """ExpenseClassifier without PhoBERT (keyword fallback)."""
    return ExpenseClassifier()


# ── Category constants ────────────────────────────────────────────────────────

class TestCategories:
    def test_ten_categories(self):
        assert len(CATEGORIES) == 10

    def test_all_required_categories_present(self):
        required = {
            "Ăn uống", "Đi lại", "Mua sắm", "Giải trí",
            "Hoá đơn tiện ích", "Sức khoẻ", "Giáo dục",
            "Du lịch", "Nhà cửa", "Khác",
        }
        assert set(CATEGORIES) == required

    def test_category_index_consistent(self):
        for i, cat in enumerate(CATEGORIES):
            assert CATEGORY_INDEX[cat] == i


# ── KeywordClassifier — merchant matching ─────────────────────────────────────

class TestKeywordClassifierMerchant:
    def test_highlands_coffee_food(self, keyword_clf):
        cat, conf = keyword_clf.classify("Highlands Coffee", [])
        assert cat == "Ăn uống"
        assert conf >= 0.75

    def test_grab_transport(self, keyword_clf):
        cat, conf = keyword_clf.classify("Grab", [])
        assert cat == "Đi lại"
        assert conf >= 0.75

    def test_pharmacity_health(self, keyword_clf):
        cat, conf = keyword_clf.classify("Pharmacity", [])
        assert cat == "Sức khoẻ"
        assert conf >= 0.75

    def test_vinmart_shopping(self, keyword_clf):
        cat, conf = keyword_clf.classify("VinMart", [])
        assert cat == "Mua sắm"
        assert conf >= 0.75

    def test_case_insensitive_merchant(self, keyword_clf):
        cat1, _ = keyword_clf.classify("highlands coffee", [])
        cat2, _ = keyword_clf.classify("HIGHLANDS COFFEE", [])
        assert cat1 == cat2 == "Ăn uống"

    def test_typo_in_merchant_still_matches(self, keyword_clf):
        """Common OCR typo in chain store name should still match."""
        cat, conf = keyword_clf.classify("Higlands Coffee", [])
        assert cat == "Ăn uống"
        assert conf >= 0.70

    def test_unknown_merchant_falls_through(self, keyword_clf):
        """Completely unknown merchant should not return a merchant match."""
        cat, conf = keyword_clf.classify("Quán Bà Năm Đường Lê Lợi", [])
        # May match via keyword fallback or return Khác — just not high confidence
        assert conf <= 0.92


# ── KeywordClassifier — item keyword matching ─────────────────────────────────

class TestKeywordClassifierItems:
    def test_coffee_items_food(self, keyword_clf):
        cat, conf = keyword_clf.classify(None, ["Cà phê sữa", "Bánh mì"])
        assert cat == "Ăn uống"
        assert conf > 0.5

    def test_fuel_items_transport(self, keyword_clf):
        cat, conf = keyword_clf.classify(None, ["Xăng RON95", "Dầu diesel"])
        assert cat == "Đi lại"
        assert conf > 0.5

    def test_medicine_items_health(self, keyword_clf):
        cat, conf = keyword_clf.classify(None, ["Thuốc paracetamol 500mg", "Vitamin C"])
        assert cat == "Sức khoẻ"
        assert conf > 0.5

    def test_electricity_items_utility(self, keyword_clf):
        cat, conf = keyword_clf.classify(None, ["Tiền điện tháng 3", "EVN"])
        assert cat == "Hoá đơn tiện ích"
        assert conf > 0.5

    def test_no_merchant_no_items_returns_khac(self, keyword_clf):
        cat, conf = keyword_clf.classify(None, [])
        assert cat == "Khác"
        assert conf == 0.40

    def test_empty_string_merchant_falls_through(self, keyword_clf):
        cat, conf = keyword_clf.classify("", [])
        assert cat == "Khác"

    def test_merchant_takes_priority_over_items(self, keyword_clf):
        """If merchant matches clearly, don't let ambiguous items override."""
        # Grab is Đi lại, but items say food — merchant should win
        cat, conf = keyword_clf.classify("Grab", ["Cơm gà", "Nước ngọt"])
        assert cat == "Đi lại"


# ── ExpenseClassifier — interface ─────────────────────────────────────────────

class TestExpenseClassifierInterface:
    def test_classify_returns_classification_result(self, expense_clf):
        result = expense_clf.classify("Highlands Coffee", ["Cà phê"])
        assert isinstance(result, ClassificationResult)

    def test_classify_result_has_valid_category(self, expense_clf):
        result = expense_clf.classify("KFC", ["Gà rán", "Khoai tây"])
        assert result.category in CATEGORIES

    def test_classify_result_confidence_in_range(self, expense_clf):
        result = expense_clf.classify("Pharmacity", ["Thuốc"])
        assert 0.0 <= result.confidence <= 1.0

    def test_keyword_fallback_method_label(self, expense_clf):
        result = expense_clf.classify("Highlands Coffee", [])
        assert result.method == "keyword"

    def test_using_phobert_false_without_model(self, expense_clf):
        assert expense_clf.using_phobert is False

    def test_classify_none_merchant(self, expense_clf):
        result = expense_clf.classify(None, ["Cà phê sữa"])
        assert result.category in CATEGORIES

    def test_classify_empty_items(self, expense_clf):
        result = expense_clf.classify("Vinmart+", [])
        assert result.category in CATEGORIES

    def test_classify_both_none(self, expense_clf):
        result = expense_clf.classify(None, [])
        assert result.category == "Khác"

    def test_classify_batch_returns_list(self, expense_clf):
        receipts = [
            ("Highlands Coffee", ["Cà phê"]),
            ("Grab", ["Chuyến đi"]),
            (None, []),
        ]
        results = expense_clf.classify_batch(receipts)
        assert len(results) == 3
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_classify_batch_correct_categories(self, expense_clf):
        receipts = [
            ("Highlands Coffee", []),
            ("Pharmacity", []),
        ]
        results = expense_clf.classify_batch(receipts)
        assert results[0].category == "Ăn uống"
        assert results[1].category == "Sức khoẻ"


# ── ExpenseClassifier — input formatting ──────────────────────────────────────

class TestInputFormatting:
    def test_basic_format(self):
        text = ExpenseClassifier._build_input("Highlands Coffee", ["Cà phê sữa", "Bánh mì"])
        assert text == "Highlands Coffee | Cà phê sữa, Bánh mì"

    def test_no_items(self):
        text = ExpenseClassifier._build_input("KFC", [])
        assert text == "KFC"

    def test_no_merchant(self):
        text = ExpenseClassifier._build_input(None, ["Cà phê", "Bánh"])
        assert text == "Cà phê, Bánh"

    def test_both_none_empty_string(self):
        text = ExpenseClassifier._build_input(None, [])
        assert text == ""

    def test_truncates_at_ten_items(self):
        items = [f"item_{i}" for i in range(15)]
        text = ExpenseClassifier._build_input("Merchant", items)
        item_part = text.split(" | ")[1]
        assert item_part.count(",") == 9   # 10 items → 9 commas

    def test_strips_whitespace(self):
        text = ExpenseClassifier._build_input("  Grab  ", ["  Xăng  ", "  Dầu  "])
        assert "  " not in text


# ── ClassificationResult ──────────────────────────────────────────────────────

class TestClassificationResult:
    def test_low_confidence_flag(self):
        r = ClassificationResult(category="Khác", confidence=0.30, method="keyword")
        assert r.is_low_confidence is True

    def test_high_confidence_not_low(self):
        r = ClassificationResult(category="Ăn uống", confidence=0.95, method="phobert")
        assert r.is_low_confidence is False

    def test_threshold_boundary(self):
        r_at = ClassificationResult(category="Ăn uống", confidence=PHOBERT_CONFIDENCE_THRESHOLD, method="phobert")
        r_below = ClassificationResult(category="Ăn uống", confidence=PHOBERT_CONFIDENCE_THRESHOLD - 0.01, method="phobert")
        assert r_at.is_low_confidence is False
        assert r_below.is_low_confidence is True

    def test_all_scores_default_empty(self):
        r = ClassificationResult(category="Ăn uống", confidence=0.9, method="keyword")
        assert r.all_scores == {}


# ── PhoBERT mock tests ────────────────────────────────────────────────────────

class TestPhoBERTMocked:
    def test_from_pretrained_falls_back_on_error(self):
        """If AutoTokenizer raises during _try_load, should not propagate — use keyword fallback."""
        clf = ExpenseClassifier()
        # Patch the internal imports to raise — _try_load must catch this
        with patch("src.classification.phobert_classifier.ExpenseClassifier._try_load",
                   wraps=lambda self_inner, path, dev: None):
            pass  # wraps is just to verify signature
        # Direct test: patch transformers import inside _try_load
        with patch.dict("sys.modules", {"transformers": None}):
            clf._try_load("some/model", "cpu")
        assert clf.using_phobert is False

    def test_phobert_classify_uses_model_when_loaded(self, expense_clf):
        """When model IS loaded, classify() should call _phobert_classify."""
        import torch

        # Build mock tokenizer that returns a MagicMock with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # .to(device) returns itself

        mock_tokenizer = MagicMock(return_value=mock_inputs)

        # Build mock model that returns logits
        logits = torch.zeros(1, 10)
        logits[0][0] = 5.0   # Ăn uống wins
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model = MagicMock(return_value=mock_output)

        expense_clf._model = mock_model
        expense_clf._tokenizer = mock_tokenizer
        expense_clf._device = "cpu"

        result = expense_clf.classify("Test Merchant", ["item"])
        assert result.method == "phobert"
        assert result.category == CATEGORIES[0]   # Ăn uống

        # cleanup
        expense_clf._model = None
        expense_clf._tokenizer = None
