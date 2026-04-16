"""
src/evaluation/metrics.py
--------------------------
Evaluation framework cho SmartReceipt VN.

Dùng để benchmark mọi thay đổi (prompt, preprocessing, model) trên golden test set.
Không bao giờ train hoặc prompt-engineer trực tiếp trên golden set.

Entry point chính:
    evaluate(predictions, ground_truth) -> dict

Schema của mỗi record (pred hoặc gt):
    image, group, merchant_name, date (DD-MM-YYYY), time,
    items: [{name, quantity, unit_price, total}],
    subtotal, vat, total_amount, payment_method, category
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from rapidfuzz import fuzz

# ── Matching thresholds ────────────────────────────────────────────────────────
_MERCHANT_THRESHOLD = 80   # token_set_ratio để coi merchant là đúng
_ITEM_THRESHOLD     = 70   # token_set_ratio để pair hai item lại với nhau
_TOTAL_TOLERANCE    = 1    # VND — cho phép sai lệch làm tròn

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    """Kết quả đánh giá một sample (một ảnh)."""
    image: str
    group: str

    # Per-field correctness
    merchant_ok: bool    # fuzzy(pred, gt) >= _MERCHANT_THRESHOLD
    date_ok: bool        # exact match sau khi normalize về YYYY-MM-DD
    total_ok: bool       # abs(pred - gt) <= _TOTAL_TOLERANCE

    # Items matching
    items_precision: float   # trong số pred items, bao nhiêu % match được gt
    items_recall: float      # trong số gt items, bao nhiêu % được pred tìm thấy
    items_f1: float

    # Category
    category_ok: bool

    # End-to-end: TẤT CẢ 3 critical fields (merchant + date + total) đều đúng
    end_to_end_ok: bool

    # Latency (giây) — 0.0 nếu không đo
    latency_sec: float = 0.0


@dataclass
class GroupStats:
    """Thống kê cho một nhóm độ khó."""
    n: int
    merchant_accuracy: float
    date_accuracy: float
    total_accuracy: float
    items_f1: float
    category_accuracy: float
    end_to_end_success: float


@dataclass
class EvalReport:
    """Kết quả tổng hợp trên toàn bộ dataset."""
    n_samples: int

    # Field-level accuracy
    field_accuracy: dict[str, float]   # {"merchant": 0.85, "date": 0.90, ...}

    # Items
    items_precision: float
    items_recall: float
    items_f1: float

    # Category
    category_accuracy: float

    # End-to-end
    end_to_end_success: float

    # Per difficulty group
    per_group: dict[str, GroupStats]

    # Per category accuracy
    per_category_accuracy: dict[str, float]

    # Latency (giây) — NaN nếu không có dữ liệu latency
    latency_p50: float
    latency_p95: float

    # Raw per-sample results — để drill down vào failure cases
    sample_results: list[SampleResult] = field(default_factory=list)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _normalize_date(s: Any) -> str | None:
    """
    Chuyển DD-MM-YYYY → YYYY-MM-DD. Trả về None nếu không parse được.
    Chấp nhận cả YYYY-MM-DD (đã chuẩn) và D/M/YYYY.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()

    # Thử DD-MM-YYYY hoặc D-M-YYYY (dấu gạch ngang)
    for sep in ("-", "/"):
        parts = s.split(sep)
        if len(parts) == 3:
            a, b, c = parts
            # DD-MM-YYYY
            if len(c) == 4 and c.isdigit():
                try:
                    return f"{int(c):04d}-{int(b):02d}-{int(a):02d}"
                except ValueError:
                    pass
            # YYYY-MM-DD đã chuẩn
            if len(a) == 4 and a.isdigit():
                try:
                    return f"{int(a):04d}-{int(b):02d}-{int(c):02d}"
                except ValueError:
                    pass
    return None


def _fuzzy_score(a: Any, b: Any) -> float:
    """
    Tính token_set_ratio giữa hai chuỗi (0–100).
    - Cả hai None/rỗng → 100.0 (đồng ý là không có giá trị)
    - Một bên None/rỗng, bên kia có giá trị → 0.0
    """
    a_empty = not a or (isinstance(a, str) and not a.strip())
    b_empty = not b or (isinstance(b, str) and not b.strip())
    if a_empty and b_empty:
        return 100.0
    if a_empty or b_empty:
        return 0.0
    return fuzz.token_set_ratio(str(a).lower().strip(), str(b).lower().strip())


def _safe_int(v: Any) -> int | None:
    """Ép kiểu về int an toàn, trả về None nếu không hợp lệ."""
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _match_items(
    pred_items: list[dict],
    gt_items: list[dict],
) -> tuple[float, float, float]:
    """
    Ghép pred items với gt items bằng greedy matching trên tên (fuzzy).

    Thuật toán:
      - Với mỗi pred item, tìm gt item chưa được dùng có score cao nhất.
      - Nếu score >= _ITEM_THRESHOLD → coi là match, đánh dấu gt item đã dùng.
      - Precision = matched / len(pred_items)
      - Recall    = matched / len(gt_items)
      - F1        = harmonic mean(precision, recall)

    Returns: (precision, recall, f1)
    """
    if not gt_items and not pred_items:
        return 1.0, 1.0, 1.0
    if not gt_items:
        return 0.0, 1.0, 0.0   # pred có items nhưng gt không có → precision = 0
    if not pred_items:
        return 1.0, 0.0, 0.0   # gt có items nhưng pred không có → recall = 0

    used_gt = [False] * len(gt_items)
    matched = 0

    for p_item in pred_items:
        p_name = str(p_item.get("name", "")).lower().strip()
        best_score = 0.0
        best_idx = -1

        for i, g_item in enumerate(gt_items):
            if used_gt[i]:
                continue
            g_name = str(g_item.get("name", "")).lower().strip()
            score = fuzz.token_set_ratio(p_name, g_name)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0 and best_score >= _ITEM_THRESHOLD:
            matched += 1
            used_gt[best_idx] = True

    precision = matched / len(pred_items)
    recall    = matched / len(gt_items)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


# ── Per-sample evaluation ─────────────────────────────────────────────────────

def evaluate_sample(
    pred: dict,
    gt: dict,
    latency_sec: float = 0.0,
) -> SampleResult:
    """
    So sánh một prediction với ground truth, trả về SampleResult.

    pred và gt phải cùng schema (xem module docstring).
    Nếu pred thiếu field nào thì field đó được coi là None → fail.
    """
    # Merchant
    merchant_score = _fuzzy_score(pred.get("merchant_name"), gt.get("merchant_name"))
    merchant_ok = merchant_score >= _MERCHANT_THRESHOLD

    # Date — None vs None = match (cả hai đồng ý không đọc được ngày)
    pred_date = _normalize_date(pred.get("date"))
    gt_date   = _normalize_date(gt.get("date"))
    date_ok   = pred_date == gt_date   # None == None là True

    # Total amount
    pred_total = _safe_int(pred.get("total_amount"))
    gt_total   = _safe_int(gt.get("total_amount"))
    if pred_total is not None and gt_total is not None:
        total_ok = abs(pred_total - gt_total) <= _TOTAL_TOLERANCE
    else:
        total_ok = False

    # Items
    prec, rec, f1 = _match_items(
        pred.get("items") or [],
        gt.get("items") or [],
    )

    # Category (exact match)
    category_ok = (
        pred.get("category") is not None
        and str(pred.get("category")).strip() == str(gt.get("category", "")).strip()
    )

    # End-to-end: 3 critical fields
    end_to_end_ok = merchant_ok and date_ok and total_ok

    return SampleResult(
        image=gt.get("image", ""),
        group=gt.get("group", "unknown"),
        merchant_ok=merchant_ok,
        date_ok=date_ok,
        total_ok=total_ok,
        items_precision=prec,
        items_recall=rec,
        items_f1=f1,
        category_ok=category_ok,
        end_to_end_ok=end_to_end_ok,
        latency_sec=latency_sec,
    )


# ── Dataset-level evaluation ──────────────────────────────────────────────────

def evaluate_dataset(
    preds: list[dict],
    gts: list[dict],
    latencies: list[float] | None = None,
) -> EvalReport:
    """
    Đánh giá toàn bộ dataset.

    preds và gts phải cùng thứ tự (index i tương ứng với nhau).
    latencies: thời gian xử lý (giây) cho từng sample, None nếu không có.
    """
    if len(preds) != len(gts):
        raise ValueError(
            f"preds ({len(preds)}) và gts ({len(gts)}) phải cùng số lượng"
        )

    if latencies is None:
        latencies = [0.0] * len(preds)

    if len(latencies) != len(preds):
        raise ValueError("latencies phải cùng độ dài với preds/gts")

    # Tính per-sample
    results: list[SampleResult] = [
        evaluate_sample(p, g, lat)
        for p, g, lat in zip(preds, gts, latencies)
    ]

    n = len(results)
    if n == 0:
        raise ValueError("Dataset rỗng")

    # ── Aggregate field accuracy ──
    field_accuracy = {
        "merchant" : sum(r.merchant_ok  for r in results) / n,
        "date"     : sum(r.date_ok      for r in results) / n,
        "total"    : sum(r.total_ok     for r in results) / n,
        "items_f1" : sum(r.items_f1     for r in results) / n,
        "category" : sum(r.category_ok  for r in results) / n,
    }

    items_precision = sum(r.items_precision for r in results) / n
    items_recall    = sum(r.items_recall    for r in results) / n
    items_f1        = sum(r.items_f1        for r in results) / n
    category_accuracy   = sum(r.category_ok   for r in results) / n
    end_to_end_success  = sum(r.end_to_end_ok for r in results) / n

    # ── Per-group breakdown ──
    groups: dict[str, list[SampleResult]] = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)

    per_group: dict[str, GroupStats] = {}
    for g_name, g_results in sorted(groups.items()):
        ng = len(g_results)
        per_group[g_name] = GroupStats(
            n=ng,
            merchant_accuracy  = sum(r.merchant_ok  for r in g_results) / ng,
            date_accuracy      = sum(r.date_ok      for r in g_results) / ng,
            total_accuracy     = sum(r.total_ok     for r in g_results) / ng,
            items_f1           = sum(r.items_f1     for r in g_results) / ng,
            category_accuracy  = sum(r.category_ok  for r in g_results) / ng,
            end_to_end_success = sum(r.end_to_end_ok for r in g_results) / ng,
        )

    # ── Per-category accuracy ──
    cat_correct: dict[str, int] = {}
    cat_total:   dict[str, int] = {}
    for r, gt in zip(results, gts):
        cat = gt.get("category", "Khác")
        cat_total[cat]   = cat_total.get(cat, 0) + 1
        cat_correct[cat] = cat_correct.get(cat, 0) + (1 if r.category_ok else 0)

    per_category_accuracy = {
        cat: cat_correct.get(cat, 0) / cat_total[cat]
        for cat in cat_total
    }

    # ── Latency percentiles ──
    non_zero_latencies = [lat for lat in latencies if lat > 0]
    if non_zero_latencies:
        sorted_lat = sorted(non_zero_latencies)
        p50 = statistics.median(sorted_lat)
        p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        p95 = sorted_lat[p95_idx]
    else:
        p50 = float("nan")
        p95 = float("nan")

    return EvalReport(
        n_samples=n,
        field_accuracy=field_accuracy,
        items_precision=items_precision,
        items_recall=items_recall,
        items_f1=items_f1,
        category_accuracy=category_accuracy,
        end_to_end_success=end_to_end_success,
        per_group=per_group,
        per_category_accuracy=per_category_accuracy,
        latency_p50=p50,
        latency_p95=p95,
        sample_results=results,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def evaluate(
    predictions: list[dict],
    ground_truth: list[dict],
    latencies: list[float] | None = None,
) -> dict:
    """
    Entry point chính — evaluate(predictions, ground_truth) -> dict.

    Trả về EvalReport dưới dạng dict JSON-serializable (dùng cho logging,
    lưu experiment docs, so sánh across versions).

    Ví dụ sử dụng:
        result = evaluate(preds, gts)
        print(f"End-to-end: {result['end_to_end_success']:.1%}")
        print(f"Merchant:   {result['field_accuracy']['merchant']:.1%}")
    """
    report = evaluate_dataset(predictions, ground_truth, latencies)

    # Chuyển dataclasses → dict (JSON-serializable)
    d = asdict(report)

    # Bỏ sample_results khỏi top-level dict (quá dài cho logging)
    # Caller có thể access qua evaluate_dataset() nếu cần drill down
    d.pop("sample_results", None)

    return d


def get_failures(
    predictions: list[dict],
    ground_truth: list[dict],
    latencies: list[float] | None = None,
) -> list[dict]:
    """
    Trả về danh sách các sample có end_to_end_ok = False, kèm chi tiết lỗi.
    Dùng cho error analysis.

    Returns: list of dicts với keys: image, group, merchant_ok, date_ok,
             total_ok, items_f1, category_ok, pred, gt
    """
    report = evaluate_dataset(predictions, ground_truth, latencies)
    failures = []
    for r, pred, gt in zip(report.sample_results, predictions, ground_truth):
        if not r.end_to_end_ok:
            failures.append({
                **asdict(r),
                "pred_merchant" : pred.get("merchant_name"),
                "gt_merchant"   : gt.get("merchant_name"),
                "pred_date"     : pred.get("date"),
                "gt_date"       : gt.get("date"),
                "pred_total"    : pred.get("total_amount"),
                "gt_total"      : gt.get("total_amount"),
            })
    return failures
