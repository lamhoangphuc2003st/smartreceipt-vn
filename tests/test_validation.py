"""
tests/test_validation.py
--------------------------
Unit tests for the validation layer.

Covers:
  - Merchant fuzzy matching (correct, no-match, exact, edge cases)
  - Date sanity (valid, future, pre-2020, unparseable)
  - Currency anomaly warnings (normal, large total, large item)
  - Math checks (correct sum, wrong sum, subtotal derivation, vat math)
  - ValidationResult flags (is_valid for usable/unusable receipts)
  - No mutation of original receipt
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.extraction.schemas import Receipt, ReceiptItem
from src.validation import Correction, ReceiptValidator, ValidationResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_receipt(**kwargs) -> Receipt:
    """Build a Receipt with sensible defaults, overriding with kwargs."""
    defaults = dict(
        merchant_name="Test Merchant",
        date=date.today().isoformat(),
        items=[
            ReceiptItem(name="Cà phê", quantity=2, unit_price=45000, total=90000),
            ReceiptItem(name="Bánh mì", quantity=1, unit_price=25000, total=25000),
        ],
        subtotal=115000,
        total_amount=115000,
    )
    defaults.update(kwargs)
    return Receipt.model_validate(defaults)


def make_validator(merchants_path: Path | None = None) -> ReceiptValidator:
    """Build validator using the real merchants.json from data/."""
    if merchants_path is not None:
        return ReceiptValidator(merchants_path=merchants_path)
    real_path = Path(__file__).parent.parent / "data" / "merchants.json"
    return ReceiptValidator(merchants_path=real_path)


# ── Merchant validation ────────────────────────────────────────────────────────

class TestMerchantValidation:
    def test_no_correction_for_unknown_merchant(self):
        """Merchants not in the database must not be corrected."""
        receipt = make_receipt(merchant_name="Quán Bà Bảy")
        result = make_validator().validate(receipt)
        assert result.receipt.merchant_name == "Quán Bà Bảy"
        assert not any(c.field == "merchant_name" for c in result.corrections)

    def test_exact_match_no_correction(self):
        """Exact match (case-insensitive) should not produce a correction."""
        receipt = make_receipt(merchant_name="Highlands Coffee")
        result = make_validator().validate(receipt)
        assert result.receipt.merchant_name == "Highlands Coffee"
        assert not any(c.field == "merchant_name" for c in result.corrections)

    def test_typo_corrected(self):
        """Known typo alias 'Higlands Coffee' → 'Highlands Coffee'."""
        receipt = make_receipt(merchant_name="Higlands Coffee")
        result = make_validator().validate(receipt)
        assert result.receipt.merchant_name == "Highlands Coffee"
        corrections = [c for c in result.corrections if c.field == "merchant_name"]
        assert len(corrections) == 1
        assert corrections[0].original == "Higlands Coffee"
        assert corrections[0].corrected == "Highlands Coffee"

    def test_generic_name_not_corrected(self):
        """
        Generic names that share words with known merchants must NOT be corrected
        (false positive guard). Threshold=90 blocks these.
        """
        receipt = make_receipt(merchant_name="The Noodle House")
        result = make_validator().validate(receipt)
        assert result.receipt.merchant_name == "The Noodle House"
        assert not any(c.field == "merchant_name" for c in result.corrections)

    def test_alias_corrected(self):
        """Known alias 'VinMart' → canonical 'WinMart'."""
        receipt = make_receipt(merchant_name="VinMart")
        result = make_validator().validate(receipt)
        assert result.receipt.merchant_name == "WinMart"

    def test_null_merchant_skipped(self):
        """None merchant_name must not crash and must not produce a correction."""
        receipt = make_receipt(merchant_name=None)
        result = make_validator().validate(receipt)
        assert result.receipt.merchant_name is None
        assert not any(c.field == "merchant_name" for c in result.corrections)

    def test_original_receipt_not_mutated(self):
        """Validator must never mutate the input receipt object."""
        receipt = make_receipt(merchant_name="Higlands Coffee")
        _ = make_validator().validate(receipt)
        assert receipt.merchant_name == "Higlands Coffee"  # unchanged


# ── Date validation ────────────────────────────────────────────────────────────

class TestDateValidation:
    def test_valid_date_unchanged(self):
        receipt = make_receipt(date="2024-03-15")
        result = make_validator().validate(receipt)
        assert result.receipt.date == "2024-03-15"
        assert not any(c.field == "date" for c in result.corrections)

    def test_future_date_nullified(self):
        """Date more than 1 day in the future must be nullified."""
        future = (date.today() + timedelta(days=5)).isoformat()
        receipt = make_receipt(date=future)
        result = make_validator().validate(receipt)
        assert result.receipt.date is None
        corrections = [c for c in result.corrections if c.field == "date"]
        assert len(corrections) == 1
        assert "future" in corrections[0].reason

    def test_tomorrow_allowed(self):
        """Date 1 day in the future is within the buffer and must pass."""
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        receipt = make_receipt(date=tomorrow)
        result = make_validator().validate(receipt)
        assert result.receipt.date == tomorrow
        assert not any(c.field == "date" for c in result.corrections)

    def test_pre_2020_produces_warning(self):
        """Date before 2020-01-01 should produce a warning, not nullify."""
        receipt = make_receipt(date="2019-12-31")
        result = make_validator().validate(receipt)
        assert result.receipt.date == "2019-12-31"  # not corrected
        assert any("2019" in w for w in result.warnings)

    def test_unparseable_date_already_null(self):
        """
        Malformed date strings (e.g. '32/13/2024') are nullified by the Receipt
        schema validator before reaching ReceiptValidator, so date arrives as None.
        Validator must handle None date gracefully (no crash, no correction).
        """
        # Receipt schema's validate_date_format already converts bad dates to None
        receipt = make_receipt(date=None)
        result = make_validator().validate(receipt)
        assert result.receipt.date is None
        assert not any(c.field == "date" for c in result.corrections)

    def test_null_date_skipped(self):
        receipt = make_receipt(date=None)
        result = make_validator().validate(receipt)
        assert result.receipt.date is None
        assert not any(c.field == "date" for c in result.corrections)


# ── Currency anomaly warnings ──────────────────────────────────────────────────

class TestCurrencyValidation:
    def test_normal_amounts_no_warning(self):
        receipt = make_receipt(total_amount=115000)
        result = make_validator().validate(receipt)
        assert not result.warnings

    def test_large_total_warns(self):
        """Total > 1B VND must produce a warning (not reject)."""
        receipt = make_receipt(total_amount=1_500_000_000, subtotal=1_500_000_000)
        result = make_validator().validate(receipt)
        assert any("1,500,000,000" in w or "total_amount" in w for w in result.warnings)
        assert result.is_valid  # still valid — just flagged

    def test_large_item_warns(self):
        """Item > 100M VND must produce a warning."""
        receipt = make_receipt(
            items=[ReceiptItem(name="Laptop", quantity=1, unit_price=120_000_000, total=120_000_000)],
            total_amount=120_000_000,
            subtotal=120_000_000,
        )
        result = make_validator().validate(receipt)
        assert any("Laptop" in w or "120,000,000" in w for w in result.warnings)

    def test_boundary_amount_no_warning(self):
        """Exactly 1B VND total must NOT warn (only strictly above)."""
        receipt = make_receipt(total_amount=1_000_000_000, subtotal=1_000_000_000)
        result = make_validator().validate(receipt)
        # 1_000_000_000 == _TOTAL_AMOUNT_WARN_VND so no warning
        assert not any("total_amount" in w for w in result.warnings)


# ── Math validation ────────────────────────────────────────────────────────────

class TestMathValidation:
    def test_correct_math_no_warning(self):
        """Correct sum produces no warnings or corrections."""
        receipt = make_receipt(
            items=[
                ReceiptItem(name="A", total=50000),
                ReceiptItem(name="B", total=65000),
            ],
            subtotal=115000,
            total_amount=115000,
        )
        result = make_validator().validate(receipt)
        math_warnings = [w for w in result.warnings if "math" in w.lower() or "sum" in w.lower() or "≠" in w]
        assert not math_warnings

    def test_wrong_items_sum_warns(self):
        """When sum(items) ≠ subtotal, produce a warning."""
        receipt = make_receipt(
            items=[
                ReceiptItem(name="A", total=50000),
                ReceiptItem(name="B", total=50000),  # sum=100000
            ],
            subtotal=115000,   # wrong
            total_amount=115000,
        )
        result = make_validator().validate(receipt)
        assert any("100,000" in w or "items sum" in w.lower() for w in result.warnings)

    def test_subtotal_derived_from_items(self):
        """When subtotal is missing but all items have totals, derive subtotal."""
        receipt = make_receipt(
            items=[
                ReceiptItem(name="A", total=50000),
                ReceiptItem(name="B", total=65000),
            ],
            subtotal=None,
            total_amount=115000,
        )
        result = make_validator().validate(receipt)
        assert result.receipt.subtotal == 115000
        corrections = [c for c in result.corrections if c.field == "subtotal"]
        assert len(corrections) == 1
        assert corrections[0].original is None
        assert corrections[0].corrected == 115000

    def test_subtotal_not_derived_when_items_incomplete(self):
        """Don't derive subtotal if some items have null total."""
        receipt = make_receipt(
            items=[
                ReceiptItem(name="A", total=50000),
                ReceiptItem(name="B", total=None),  # incomplete
            ],
            subtotal=None,
            total_amount=115000,
        )
        result = make_validator().validate(receipt)
        assert result.receipt.subtotal is None  # not derived
        assert not any(c.field == "subtotal" for c in result.corrections)

    def test_vat_math_mismatch_warns(self):
        """subtotal + vat ≠ total should produce a warning."""
        receipt = make_receipt(
            items=[ReceiptItem(name="X", total=100000)],
            subtotal=100000,
            vat_amount=8000,    # 8% VAT
            total_amount=110000,  # wrong: should be 108000
        )
        result = make_validator().validate(receipt)
        assert any("math" in w.lower() or "≠" in w for w in result.warnings)

    def test_one_vnd_rounding_ok(self):
        """1 VND rounding difference must NOT produce a warning."""
        receipt = make_receipt(
            items=[ReceiptItem(name="X", total=99999)],
            subtotal=99999,
            total_amount=100000,  # 1 VND difference
        )
        result = make_validator().validate(receipt)
        math_warnings = [w for w in result.warnings if "≠" in w]
        assert not math_warnings

    def test_no_items_skips_math(self):
        """Receipt with no items must skip math check without error."""
        receipt = make_receipt(items=[], subtotal=None, total_amount=50000)
        result = make_validator().validate(receipt)
        assert result.is_valid  # still usable


# ── ValidationResult.is_valid ──────────────────────────────────────────────────

class TestIsValid:
    def test_receipt_with_total_is_valid(self):
        receipt = make_receipt(items=[], total_amount=50000)
        result = make_validator().validate(receipt)
        assert result.is_valid

    def test_receipt_with_items_is_valid(self):
        receipt = make_receipt(total_amount=None)
        result = make_validator().validate(receipt)
        assert result.is_valid

    def test_no_total_no_items_is_invalid(self):
        """Receipt missing both total_amount and items is unusable."""
        receipt = make_receipt(items=[], total_amount=None, subtotal=None)
        result = make_validator().validate(receipt)
        assert not result.is_valid

    def test_multiple_corrections_still_valid(self):
        """Multiple corrections must not affect is_valid if data is usable."""
        receipt = make_receipt(
            merchant_name="Higlands Coffee",
            date=(date.today() + timedelta(days=5)).isoformat(),
            total_amount=90000,
        )
        result = make_validator().validate(receipt)
        assert result.is_valid
        assert len(result.corrections) >= 2  # merchant + date

    def test_correction_log_has_reason(self):
        """Every correction must have a non-empty reason string."""
        receipt = make_receipt(merchant_name="Higlands Coffee")
        result = make_validator().validate(receipt)
        for c in result.corrections:
            assert c.reason, f"Correction for {c.field} has empty reason"
