"""
app/streamlit_app.py
---------------------
SmartReceipt VN — Streamlit demo UI.

Runs the full pipeline in-process (no separate API server needed).

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when running from app/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SmartReceipt VN",
    page_icon="🧾",
    layout="wide",
)

# ── Category colours ─────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "Ăn uống":           "#FF6B6B",
    "Đi lại":            "#4ECDC4",
    "Mua sắm":           "#45B7D1",
    "Giải trí":          "#96CEB4",
    "Hoá đơn tiện ích":  "#FFEAA7",
    "Sức khoẻ":          "#DDA0DD",
    "Giáo dục":          "#98D8C8",
    "Du lịch":           "#F7DC6F",
    "Nhà cửa":           "#A29BFE",
    "Khác":              "#B2BEC3",
}

CATEGORY_ICONS = {
    "Ăn uống":           "🍜",
    "Đi lại":            "🚗",
    "Mua sắm":           "🛍️",
    "Giải trí":          "🎭",
    "Hoá đơn tiện ích":  "💡",
    "Sức khoẻ":          "💊",
    "Giáo dục":          "📚",
    "Du lịch":           "✈️",
    "Nhà cửa":           "🏠",
    "Khác":              "📦",
}

# ── Pipeline (cached — load once) ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang tải model PhoBERT...")
def load_pipeline():
    from src.pipeline.receipt_pipeline import ReceiptPipeline
    return ReceiptPipeline(
        classifier_model="models/phobert-expense-v1",
        classifier_device="cpu",
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_vnd(amount: int | None) -> str:
    if amount is None:
        return "—"
    return f"{amount:,.0f} ₫".replace(",", ".")


def _fmt_date(date_str: str | None) -> str:
    """Convert any date string to DD/MM/YYYY for display."""
    if not date_str:
        return "—"
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return date_str  # fallback: return as-is if unparseable


def _confidence_bar(confidence: float, color: str) -> str:
    pct = int(confidence * 100)
    return f"""
    <div style="background:#eee;border-radius:4px;height:10px;margin-top:4px">
      <div style="background:{color};width:{pct}%;height:10px;border-radius:4px"></div>
    </div>
    <small style="color:#666">{pct}% confidence</small>
    """

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🧾 SmartReceipt VN")
st.caption("Trích xuất và phân loại chi tiêu từ hoá đơn Việt Nam")

# Sidebar — about
with st.sidebar:
    st.header("Về dự án")
    st.markdown("""
    **SmartReceipt VN** trích xuất thông tin có cấu trúc từ ảnh hoá đơn Việt Nam và phân loại chi tiêu tự động.

    **Pipeline:**
    1. 🖼️ Tiền xử lý ảnh
    2. 🤖 Gemini Vision (trích xuất)
    3. ✅ Validation
    4. 🏷️ PhoBERT (phân loại)

    **Accuracy (golden set):**
    - Category: **91%**
    - Merchant: **~85%**
    """)

    st.divider()
    st.caption("Portfolio project — AI Engineering")

# Main content
uploaded = st.file_uploader(
    "Tải lên ảnh hoá đơn",
    type=["jpg", "jpeg", "png", "webp"],
    help="Hỗ trợ: POS receipt, VAT invoice, hoá đơn viết tay",
)

if uploaded is None:
    st.info("Tải lên một ảnh hoá đơn để bắt đầu.")
    st.stop()

image_bytes = uploaded.read()

col_img, col_result = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("Ảnh hoá đơn")
    st.image(image_bytes, width="stretch")

with col_result:
    st.subheader("Kết quả trích xuất")

    with st.spinner("Đang xử lý..."):
        pipeline = load_pipeline()
        result = pipeline.process(image_bytes)

    if not result.success:
        st.error(f"Xử lý thất bại: {result.error_message}")
        st.stop()

    # ── Category badge ────────────────────────────────────────────────────
    category = result.category or "Khác"
    confidence = result.category_confidence or 0.0
    color = CATEGORY_COLORS.get(category, "#B2BEC3")
    icon  = CATEGORY_ICONS.get(category, "📦")
    method = result.category_method or "keyword"

    st.markdown(
        f"""
        <div style="background:{color}22;border:2px solid {color};border-radius:12px;
                    padding:16px;margin-bottom:16px;text-align:center">
          <div style="font-size:2.5rem">{icon}</div>
          <div style="font-size:1.4rem;font-weight:bold;color:{color}">{category}</div>
          <div style="color:#666;font-size:0.85rem">Phân loại bởi {method}</div>
          {_confidence_bar(confidence, color)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key fields ────────────────────────────────────────────────────────
    st.markdown("**Thông tin chính**")
    kv_col1, kv_col2 = st.columns(2)
    with kv_col1:
        st.metric("Cửa hàng", result.merchant_name or "—")
        st.metric("Ngày", _fmt_date(result.date))
    with kv_col2:
        st.metric("Tổng tiền", _fmt_vnd(result.total_amount))
        st.metric("Thanh toán", result.payment_method or "—")

    # ── Items table ───────────────────────────────────────────────────────
    if result.items:
        st.markdown("**Danh sách mặt hàng**")
        import pandas as pd
        df = pd.DataFrame([
            {
                "Tên hàng": it.get("name", ""),
                "SL": it.get("quantity", ""),
                "Đơn giá": _fmt_vnd(it.get("unit_price")),
                "Thành tiền": _fmt_vnd(it.get("total")),
            }
            for it in result.items
        ])
        st.dataframe(df, width="stretch", hide_index=True)

    # ── Subtotals ─────────────────────────────────────────────────────────
    if any([result.subtotal, result.vat_amount, result.discount]):
        with st.expander("Chi tiết thanh toán"):
            if result.subtotal:
                st.write(f"Tạm tính: {_fmt_vnd(result.subtotal)}")
            if result.discount:
                st.write(f"Giảm giá: -{_fmt_vnd(result.discount)}")
            if result.vat_amount:
                rate = f" ({result.vat_rate:.0%})" if result.vat_rate else ""
                st.write(f"VAT{rate}: {_fmt_vnd(result.vat_amount)}")
            st.write(f"**Tổng cộng: {_fmt_vnd(result.total_amount)}**")

    # ── All category scores ────────────────────────────────────────────────
    if result.category_all_scores:
        with st.expander("Điểm phân loại tất cả danh mục"):
            scores = sorted(result.category_all_scores.items(), key=lambda x: -x[1])
            for cat, score in scores:
                bar_color = CATEGORY_COLORS.get(cat, "#B2BEC3")
                icon_cat  = CATEGORY_ICONS.get(cat, "📦")
                st.markdown(
                    f"{icon_cat} **{cat}** — {score:.1%} "
                    + f'<span style="display:inline-block;width:{int(score*120)}px;'
                    + f'height:8px;background:{bar_color};border-radius:4px;vertical-align:middle"></span>',
                    unsafe_allow_html=True,
                )

    # ── Validation corrections ─────────────────────────────────────────────
    if result.validation_corrections:
        with st.expander(f"Auto-corrections ({len(result.validation_corrections)})"):
            for c in result.validation_corrections:
                st.warning(
                    f"**{c['field']}**: `{c['original']}` → `{c['corrected']}`  \n"
                    f"_{c['reason']}_"
                )

    # ── Pipeline timing ───────────────────────────────────────────────────
    with st.expander("Pipeline metadata"):
        timing = result.stage_times_ms or {}
        cols = st.columns(len(timing) if timing else 1)
        for i, (stage, ms) in enumerate(timing.items()):
            cols[i].metric(stage.capitalize(), f"{ms:.0f} ms")

        st.caption(f"Total: {result.total_time_ms:.0f} ms | "
                   f"Cost: ${result.extraction_cost_usd:.5f} | "
                   f"Cache: {'hit' if result.extraction_from_cache else 'miss'}")

        if result.fallbacks_triggered:
            st.warning("Fallbacks: " + ", ".join(result.fallbacks_triggered))
