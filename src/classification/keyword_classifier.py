"""
src/classification/keyword_classifier.py
-----------------------------------------
Rule-based expense classifier using merchant name matching and item keywords.

Serves two roles:
  1. Immediate baseline — works without any training, accuracy ~70-75%.
  2. Fallback — used by ExpenseClassifier when PhoBERT model is not loaded.

Classification strategy (in priority order):
  1. Merchant name → fuzzy-match against merchants.json (rapidfuzz, threshold 80)
     High confidence (0.90) when matched — merchants.json is curated.
  2. Item keywords → scan item names for category-specific Vietnamese keywords.
     Medium confidence (0.60-0.75).
  3. Default → "Khác" with low confidence (0.40).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

# ── Categories ────────────────────────────────────────────────────────────────

CATEGORIES: list[str] = [
    "Ăn uống",
    "Đi lại",
    "Mua sắm",
    "Giải trí",
    "Hoá đơn tiện ích",
    "Sức khoẻ",
    "Giáo dục",
    "Du lịch",
    "Nhà cửa",
    "Khác",
]

CATEGORY_INDEX: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

# ── Item keyword rules ────────────────────────────────────────────────────────
# Each entry: (regex_pattern, category, confidence)
# Patterns are matched against item names (lowercased, no diacritics stripped).

_KEYWORD_RULES: list[tuple[str, str, float]] = [
    # Ăn uống
    (r"cà phê|cafe|coffee|trà|nước|bia|rượu|cơm|phở|bún|bánh|gà|thịt|cá|tôm|rau|salad|pizza|burger|sandwich|mì|hủ tiếu|lẩu|nướng|sushi|dim sum|ăn|uống|snack|kẹo|đồ ăn|thức ăn|đồ uống|suất", "Ăn uống", 0.65),
    # Đi lại
    (r"xăng|dầu|petro|nhiên liệu|vé|taxi|grab|be |gojek|xe|tàu|máy bay|vé máy bay|vé xe|vé tàu|parking|đỗ xe|bãi xe|gửi xe|phí cầu đường|toll|uber", "Đi lại", 0.70),
    # Sức khoẻ
    (r"thuốc|vitamin|dược|khám|bệnh viện|phòng khám|xét nghiệm|siêu âm|nha|răng|mắt|tai mũi họng|vật lý trị liệu|spa|massage|thể hình|gym|fitness|y tế|sức khỏe|kháng sinh|paracetamol|ibuprofen", "Sức khoẻ", 0.70),
    # Giáo dục
    (r"học phí|học|khóa học|course|sách|vở|bút|stationery|văn phòng phẩm|trường|lớp|gia sư|luyện thi|ielts|toeic|anh ngữ|tiếng anh|tiếng nhật|tiếng hàn|tiếng trung|chứng chỉ|bằng|giáo trình", "Giáo dục", 0.70),
    # Giải trí
    (r"phim|cinema|rạp|game|trò chơi|nhạc|concert|sự kiện|event|vé xem|karaoke|bowling|billiard|bi a|vui chơi|giải trí|theme park|công viên|zoo|bảo tàng|triển lãm|netflix|spotify|steam", "Giải trí", 0.65),
    # Hoá đơn tiện ích
    (r"điện|điện lực|evn|nước|tiền nước|gas|internet|wifi|viễn thông|viettel|vnpt|mobifone|điện thoại|cước|phí dịch vụ|bảo hiểm|nhà mạng|data|sim|truyền hình", "Hoá đơn tiện ích", 0.72),
    # Du lịch
    (r"khách sạn|hotel|resort|homestay|villa|phòng|check.?in|check.?out|tour|du lịch|travel|vé tham quan|airbnb|booking|agoda|traveloka", "Du lịch", 0.70),
    # Nhà cửa
    (r"nội thất|đồ gia dụng|bếp|tủ lạnh|máy giặt|điều hòa|quạt|đèn|sơn|vật liệu xây dựng|sửa chữa|thợ|ống nước|điện tử|ikea|điện máy|cleaning|vệ sinh|tẩy rửa|nước rửa|bột giặt|chổi|lau|mop", "Nhà cửa", 0.65),
    # Mua sắm (catch-all for retail)
    (r"quần|áo|giày|dép|túi|ví|phụ kiện|mỹ phẩm|son|kem|nước hoa|thời trang|fashion|clothes|shoes|bag|shopee|lazada|tiki|sendo|điện tử|điện thoại|laptop|máy tính|tablet|tai nghe|sạc|cáp", "Mua sắm", 0.60),
]

_COMPILED_RULES = [(re.compile(p, re.IGNORECASE), cat, conf) for p, cat, conf in _KEYWORD_RULES]

# ── Default merchants.json path ───────────────────────────────────────────────

_DEFAULT_MERCHANTS_PATH = Path(__file__).parent.parent.parent / "data" / "merchants.json"


class KeywordClassifier:
    """
    Fast rule-based classifier. No model loading, instant startup.

    Usage:
        clf = KeywordClassifier()
        category, confidence = clf.classify("Highlands Coffee", ["Cà phê sữa", "Bánh mì"])
    """

    def __init__(self, merchants_path: Path | str | None = None) -> None:
        path = Path(merchants_path) if merchants_path else _DEFAULT_MERCHANTS_PATH
        self._merchant_map: dict[str, str] = {}   # lowercased name → category
        self._load_merchants(path)

    def _load_merchants(self, path: Path) -> None:
        if not path.exists():
            log.warning("merchants.json not found at %s — merchant matching disabled", path)
            return
        try:
            with open(path, encoding="utf-8") as f:
                merchants: list[dict] = json.load(f)
            for m in merchants:
                cat = m.get("category")
                if not cat:
                    continue
                for name in [m.get("canonical_name", "")] + m.get("aliases", []):
                    if name:
                        self._merchant_map[name.lower().strip()] = cat
            log.debug("Loaded %d merchant name→category mappings", len(self._merchant_map))
        except Exception as exc:
            log.warning("Failed to load merchants.json: %s", exc)

    def classify(
        self,
        merchant_name: str | None,
        item_names: list[str] | None = None,
    ) -> tuple[str, float]:
        """
        Returns (category, confidence).

        Priority: merchant match > item keywords > default "Khác".
        """
        # 1. Merchant name matching (fuzzy)
        if merchant_name:
            cat, conf = self._match_merchant(merchant_name)
            if cat:
                return cat, conf

        # 2. Item keyword matching
        if item_names:
            cat, conf = self._match_keywords(item_names)
            if cat:
                return cat, conf

        return "Khác", 0.40

    def _match_merchant(self, name: str) -> tuple[str | None, float]:
        """Fuzzy-match merchant name against loaded merchant map."""
        try:
            from rapidfuzz import fuzz, process as rfprocess
        except ImportError:
            return None, 0.0

        query = name.lower().strip()
        if not query:
            return None, 0.0

        # Exact or near-exact match first
        if query in self._merchant_map:
            return self._merchant_map[query], 0.92

        # Fuzzy match — token set ratio handles word-order variance
        result = rfprocess.extractOne(
            query,
            self._merchant_map.keys(),
            scorer=fuzz.token_set_ratio,
            score_cutoff=80,
        )
        if result:
            matched_name, score, _ = result
            confidence = 0.75 + (score - 80) / 100   # 0.75 at 80, up to 0.95 at 100
            return self._merchant_map[matched_name], round(min(confidence, 0.95), 2)

        return None, 0.0

    def _match_keywords(self, item_names: list[str]) -> tuple[str | None, float]:
        """Scan item names against keyword rules. Return highest-confidence match."""
        combined = " ".join(item_names).lower()
        best_cat: str | None = None
        best_conf = 0.0

        for pattern, category, confidence in _COMPILED_RULES:
            if pattern.search(combined):
                if confidence > best_conf:
                    best_cat = category
                    best_conf = confidence

        return best_cat, best_conf
