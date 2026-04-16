"""
src/validation/__init__.py
---------------------------
Public API for the validation layer.

Usage:
    from src.validation import ReceiptValidator, ValidationResult, Correction

    validator = ReceiptValidator()
    result = validator.validate(receipt)
    print(result.is_valid, result.corrections, result.warnings)
"""

from .validators import Correction, ReceiptValidator, ValidationResult

__all__ = ["ReceiptValidator", "ValidationResult", "Correction"]
