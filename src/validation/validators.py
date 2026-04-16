"""
src/validation/validators.py
------------------------------
Validation and auto-correction layer for extracted Receipt objects.

Responsibilities:
  - Math check: sum(items) + vat ≈ total (within 1 VND rounding)
  - Date sanity: within [2020-01-01, today + 1 day]
  - Currency sanity: flag anomalous amounts (don't reject)
  - Merchant fuzzy match: correct OCR errors against data/merchants.json

Design principles (from CLAUDE.md):
  - Prefer correction over rejection — return something useful always.
  - Never silently modify data — every auto-correction is logged in ValidationResult.
  - Only mark is_valid=False if data is clearly unusable (missing total on all items, etc.)
  - Anomalies (e.g. suspiciously large total) are warnings, not rejections.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process as fuzz_process

from src.extraction.schemas import Receipt, ReceiptItem

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MATH_TOLERANCE_VND      = 1        # allow 1 VND rounding error
_DATE_EARLIEST           = date(2020, 1, 1)
_DATE_LATEST_BUFFER_DAYS = 1        # allow up to 1 day in the future
_ITEM_AMOUNT_WARN_VND    = 100_000_000   # 100M VND per item — flag as anomaly
_TOTAL_AMOUNT_WARN_VND   = 1_000_000_000 # 1B VND total — flag as anomaly
_MERCHANT_MATCH_THRESHOLD = 90      # rapidfuzz token_set_ratio (0–100)
# 90 chosen empirically: all legitimate alias corrections score 100 (exact alias match).
# Threshold 90 blocks generic-word false positives (score 80–89) while passing
# real OCR typos like "Higlands Coffee" (100) and "Loteria" (100).

_MERCHANTS_PATH = Path(__file__).parent.parent.parent / "data" / "merchants.json"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Correction:
    """
    Records a single auto-correction made to the Receipt.

    Every change to extracted data must produce a Correction entry.
    Users can review all corrections in ValidationResult.corrections.
    """
    field: str          # e.g. "merchant_name", "date", "items[2].total"
    original: Any       # value before correction
    corrected: Any      # value after correction
    reason: str         # human-readable explanation


@dataclass
class ValidationResult:
    """
    Output of ReceiptValidator.validate().

    receipt     — possibly corrected Receipt (original is never mutated)
    corrections — all auto-corrections applied (empty if none)
    warnings    — anomalies flagged but NOT corrected (e.g. very large total)
    is_valid    — False only if data is clearly unusable
    """
    receipt: Receipt
    corrections: list[Correction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True


# ── Merchant database ─────────────────────────────────────────────────────────

class _MerchantDB:
    """
    Loads merchants.json and provides fuzzy-match lookup.

    Internally builds a flat alias → canonical_name map for fast lookup.
    Falls back gracefully if the file is missing.
    """

    def __init__(self, path: Path = _MERCHANTS_PATH) -> None:
        self._alias_map: dict[str, str] = {}   # alias_lower → canonical_name
        self._category_map: dict[str, str] = {} # canonical_name → category
        self._load(path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            log.warning("merchants.json not found at %s — merchant correction disabled", path)
            return
        try:
            merchants = json.loads(path.read_text(encoding="utf-8"))
            for entry in merchants:
                canonical = entry["canonical_name"]
                self._category_map[canonical] = entry.get("category", "Khác")
                # Index canonical name itself
                self._alias_map[canonical.lower().strip()] = canonical
                # Index all aliases
                for alias in entry.get("aliases", []):
                    self._alias_map[alias.lower().strip()] = canonical
            log.debug("Loaded %d merchant entries (%d aliases)", len(merchants), len(self._alias_map))
        except Exception as exc:
            log.warning("Failed to load merchants.json: %s — merchant correction disabled", exc)

    def fuzzy_match(self, name: str) -> tuple[str | None, float]:
        """
        Find the best canonical merchant name for the given input.

        Returns (canonical_name, score) or (None, 0.0) if no match above threshold.
        Score is rapidfuzz token_set_ratio (0–100).
        """
        if not name or not self._alias_map:
            return None, 0.0

        name_lower = name.lower().strip()

        # Exact match first (fast path)
        if name_lower in self._alias_map:
            return self._alias_map[name_lower], 100.0

        # Fuzzy match against all known aliases
        result = fuzz_process.extractOne(
            name_lower,
            list(self._alias_map.keys()),
            scorer=fuzz.token_set_ratio,
            score_cutoff=_MERCHANT_MATCH_THRESHOLD,
        )
        if result is None:
            return None, 0.0

        matched_alias, score, _ = result
        canonical = self._alias_map[matched_alias]
        return canonical, float(score)

    def get_category(self, canonical_name: str) -> str | None:
        return self._category_map.get(canonical_name)


# ── Validator ─────────────────────────────────────────────────────────────────

class ReceiptValidator:
    """
    Validates and auto-corrects a Receipt extracted by GeminiExtractor.

    Usage:
        validator = ReceiptValidator()
        result = validator.validate(receipt)
        if result.corrections:
            for c in result.corrections:
                print(f"Corrected {c.field}: {c.original!r} → {c.corrected!r} ({c.reason})")

    Thread-safe — validate() is stateless beyond shared _MerchantDB (read-only).
    """

    def __init__(self, merchants_path: Path = _MERCHANTS_PATH) -> None:
        self._merchant_db = _MerchantDB(merchants_path)

    def validate(self, receipt: Receipt) -> ValidationResult:
        """
        Run all validation checks on the receipt.

        Returns a ValidationResult with a (possibly corrected) receipt copy.
        The original receipt object is never mutated.
        """
        # Work on a mutable copy so we never mutate the input
        data = receipt.model_dump()
        corrections: list[Correction] = []
        warnings: list[str] = []

        # Run each check in order — each may modify data and append corrections/warnings
        self._check_merchant(data, corrections)
        self._check_date(data, corrections, warnings)
        self._check_currency(data, warnings)
        self._check_math(data, corrections, warnings)

        # Reconstruct Receipt from (possibly modified) data
        try:
            corrected_receipt = Receipt.model_validate(data)
        except Exception as exc:
            log.warning("Could not reconstruct Receipt after corrections: %s", exc)
            corrected_receipt = receipt  # fall back to original

        # Determine validity: unusable only if total is missing AND items are empty
        is_valid = not (
            corrected_receipt.total_amount is None
            and not corrected_receipt.items
        )

        return ValidationResult(
            receipt=corrected_receipt,
            corrections=corrections,
            warnings=warnings,
            is_valid=is_valid,
        )

    # ── Individual checks ──────────────────────────────────────────────────────

    def _check_merchant(self, data: dict, corrections: list[Correction]) -> None:
        """
        Fuzzy-match merchant name against known merchants.
        Corrects if score >= threshold and name differs from canonical.
        """
        name = data.get("merchant_name")
        if not name:
            return

        canonical, score = self._merchant_db.fuzzy_match(name)
        if canonical is None:
            return  # Unknown merchant — not an error

        if canonical.lower() == name.lower():
            return  # Already correct

        # Score above threshold and name differs → correct
        corrections.append(Correction(
            field="merchant_name",
            original=name,
            corrected=canonical,
            reason=f"fuzzy match to known merchant (score={score:.0f})",
        ))
        data["merchant_name"] = canonical
        log.debug("Merchant corrected: %r → %r (score=%.0f)", name, canonical, score)

    def _check_date(
        self,
        data: dict,
        corrections: list[Correction],
        warnings: list[str],
    ) -> None:
        """
        Validate date is within [2020-01-01, today + 1 day].
        Nullify clearly invalid dates (don't guess).
        """
        date_str = data.get("date")
        if not date_str:
            return

        try:
            parsed = date.fromisoformat(str(date_str))
        except (ValueError, TypeError):
            corrections.append(Correction(
                field="date",
                original=date_str,
                corrected=None,
                reason="could not parse as ISO date (YYYY-MM-DD)",
            ))
            data["date"] = None
            return

        today = date.today()
        latest_allowed = today + timedelta(days=_DATE_LATEST_BUFFER_DAYS)

        if parsed < _DATE_EARLIEST:
            warnings.append(
                f"date {date_str!r} is before {_DATE_EARLIEST} — "
                f"may be a scanning error or very old receipt"
            )
        elif parsed > latest_allowed:
            corrections.append(Correction(
                field="date",
                original=date_str,
                corrected=None,
                reason=f"date {date_str!r} is in the future (today={today}) — nullified",
            ))
            data["date"] = None

    def _check_currency(self, data: dict, warnings: list[str]) -> None:
        """
        Flag anomalously large amounts as warnings (don't reject or modify).
        These are legitimate in some cases (bulk B2B purchases, hotel bills).
        """
        total = data.get("total_amount")
        if total is not None and isinstance(total, int) and total > _TOTAL_AMOUNT_WARN_VND:
            warnings.append(
                f"total_amount {total:,} VND exceeds {_TOTAL_AMOUNT_WARN_VND:,} VND — "
                f"verify this is correct"
            )

        for i, item in enumerate(data.get("items") or []):
            item_total = item.get("total")
            if item_total is not None and isinstance(item_total, int) and item_total > _ITEM_AMOUNT_WARN_VND:
                warnings.append(
                    f"items[{i}] ({item.get('name', '?')!r}) total {item_total:,} VND "
                    f"exceeds {_ITEM_AMOUNT_WARN_VND:,} VND — verify"
                )

    def _check_math(
        self,
        data: dict,
        corrections: list[Correction],
        warnings: list[str],
    ) -> None:
        """
        Check math consistency: sum(items.total) + vat_amount ≈ total_amount.

        Rules:
          - If items sum ≠ subtotal (within 1 VND): warn (items may be incomplete).
          - If subtotal + vat ≠ total (within 1 VND): warn (don't correct — could be
            discount or rounding we don't know about).
          - If items sum + vat ≈ total but subtotal is missing: fill in subtotal.
        """
        items = data.get("items") or []
        items_with_total = [it for it in items if it.get("total") is not None]
        total_amount = data.get("total_amount")
        subtotal = data.get("subtotal")
        vat_amount = data.get("vat_amount") or 0
        discount = data.get("discount") or 0

        if not items_with_total:
            return  # Nothing to check

        items_sum = sum(it["total"] for it in items_with_total)

        # 1. Check items sum vs subtotal (if both present)
        if subtotal is not None:
            diff = abs(items_sum - subtotal)
            if diff > _MATH_TOLERANCE_VND:
                warnings.append(
                    f"items sum ({items_sum:,}) ≠ subtotal ({subtotal:,}), "
                    f"diff={diff:,} VND — items may be incomplete or subtotal includes fees"
                )

        # 2. If subtotal missing but items sum is known: fill it in
        elif len(items_with_total) == len(items):
            # All items have totals — safe to derive subtotal
            corrections.append(Correction(
                field="subtotal",
                original=None,
                corrected=items_sum,
                reason=f"derived from sum of {len(items_with_total)} item totals",
            ))
            data["subtotal"] = items_sum
            subtotal = items_sum

        # 3. Check total math: subtotal - discount + vat = total
        if total_amount is not None and subtotal is not None:
            expected_total = subtotal - discount + vat_amount
            diff = abs(expected_total - total_amount)
            if diff > _MATH_TOLERANCE_VND:
                warnings.append(
                    f"math check failed: subtotal({subtotal:,}) - discount({discount:,}) "
                    f"+ vat({vat_amount:,}) = {expected_total:,} ≠ total({total_amount:,}), "
                    f"diff={diff:,} VND — may include service charge, rounding, or other fees"
                )

        # 4. Check items sum directly against total (common for receipts without subtotal line)
        elif total_amount is not None and subtotal is None:
            expected = items_sum + vat_amount - discount
            diff = abs(expected - total_amount)
            if diff > _MATH_TOLERANCE_VND:
                warnings.append(
                    f"items sum ({items_sum:,}) + vat ({vat_amount:,}) - discount ({discount:,}) "
                    f"= {expected:,} ≠ total ({total_amount:,}), diff={diff:,} VND"
                )
