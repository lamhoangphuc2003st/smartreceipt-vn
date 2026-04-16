"""
hf_spaces/app.py
-----------------
SmartReceipt VN — Streamlit UI for Hugging Face Spaces.

This version calls the Railway-deployed FastAPI backend instead of running
the pipeline in-process, so it works within HF Spaces free-tier memory limits.

Environment variable (set in HF Spaces Secrets):
  API_BASE_URL — e.g. https://smartreceipt-vn.up.railway.app
"""

import os
import io
import requests
import streamlit as st
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SmartReceipt VN",
    page_icon="🧾",
    layout="wide",
)

# ── Category styling ──────────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_vnd(amount: int | None) -> str:
    if amount is None:
        return "—"
    return f"{amount:,.0f} ₫".replace(",", ".")


def _fmt_date(date_str: str | None) -> str:
    if not date_str:
        return "—"
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return date_str


def _call_api(image_bytes: bytes, filename: str, content_type: str) -> dict:
    """POST image to backend /receipts/extract. Returns parsed JSON."""
    url = f"{API_BASE_URL}/receipts/extract"
    try:
        resp = requests.post(
            url,
            files={"file": (filename, image_bytes, content_type)},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"success": False, "error_type": "connection_error",
                "error_message": f"Không thể kết nối tới API: {url}"}
    except requests.exceptions.Timeout:
        return {"success": False, "error_type": "timeout",
                "error_message": "API timeout sau 60 giây."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error_type": "http_error",
                "error_message": str(e)}


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🧾 SmartReceipt VN")
st.caption("Trích xuất và phân loại chi tiêu từ hoá đơn Việt Nam")

# Sidebar
with st.sidebar:
    st.header("Về dự án")
    st.markdown("""
    **SmartReceipt VN** trích xuất thông tin có cấu trúc từ ảnh hoá đơn Việt Nam
    và phân loại chi tiêu tự động.

    **Pipeline:**
    1. 🖼️ Tiền xử lý ảnh (OpenCV)
    2. 🤖 Gemini Vision (trích xuất JSON)
    3. ✅ Validation & auto-correction
    4. 🏷️ PhoBERT fine-tuned (phân loại)

    **Accuracy (100 ảnh test):**
    - Category: **91%**
    - Tổng tiền: **93%**
    - Tên cửa hàng: **~85%**
    """)
    st.divider()

    # API status
    if API_BASE_URL:
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=5)
            health = r.json()
            status = health.get("status", "unknown")
            if status == "ok":
                st.success(f"API: {status} ✓")
            else:
                st.warning(f"API: {status}")
        except Exception:
            st.error("API: unreachable")
    else:
        st.warning("API_BASE_URL chưa được cấu hình.\nSet trong HF Spaces Secrets.")

    st.caption("Portfolio project — AI Engineering")

# Guard: no API URL
if not API_BASE_URL:
    st.error(
        "**API_BASE_URL chưa được cấu hình.**\n\n"
        "Vào Settings → Secrets và thêm:\n"
        "`API_BASE_URL = https://<your-railway-app>.up.railway.app`"
    )
    st.stop()

# File uploader
uploaded = st.file_uploader(
    "Tải lên ảnh hoá đơn",
    type=["jpg", "jpeg", "png", "webp"],
    help="Hỗ trợ: POS receipt, VAT invoice, hoá đơn viết tay",
)

if uploaded is None:
    st.info("Tải lên một ảnh hoá đơn để bắt đầu.")

    # Sample images hint
    st.markdown("---")
    st.markdown("**Thử với ảnh mẫu:** Xem thư mục `data/samples/` trong repo.")
    st.stop()

image_bytes = uploaded.read()

# Detect content type from file extension
ext = uploaded.name.rsplit(".", 1)[-1].lower()
content_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp"}
content_type = content_type_map.get(ext, "image/jpeg")

col_img, col_result = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("Ảnh hoá đơn")
    st.image(image_bytes, use_container_width=True)

with col_result:
    st.subheader("Kết quả trích xuất")

    with st.spinner("Đang xử lý... (có thể mất 5-10 giây)"):
        data = _call_api(image_bytes, uploaded.name, content_type)

    if not data.get("success"):
        st.error(f"Lỗi: {data.get('error_message', 'Unknown error')}")
        with st.expander("Chi tiết lỗi"):
            st.json(data)
        st.stop()

    # ── Category badge ─────────────────────────────────────────────────────
    classification = data.get("classification", {})
    category = classification.get("category") or "Khác"
    confidence = classification.get("confidence") or 0.0
    method = classification.get("method") or "keyword"
    color = CATEGORY_COLORS.get(category, "#B2BEC3")
    icon  = CATEGORY_ICONS.get(category, "📦")
    pct   = int(confidence * 100)

    st.markdown(
        f"""
        <div style="background:{color}22;border:2px solid {color};border-radius:12px;
                    padding:16px;margin-bottom:16px;text-align:center">
          <div style="font-size:2.5rem">{icon}</div>
          <div style="font-size:1.4rem;font-weight:bold;color:{color}">{category}</div>
          <div style="color:#666;font-size:0.85rem">Phân loại bởi {method}</div>
          <div style="background:#eee;border-radius:4px;height:10px;margin-top:8px">
            <div style="background:{color};width:{pct}%;height:10px;border-radius:4px"></div>
          </div>
          <small style="color:#666">{pct}% confidence</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key fields ──────────────────────────────────────────────────────────
    st.markdown("**Thông tin chính**")
    kv1, kv2 = st.columns(2)
    with kv1:
        st.metric("Cửa hàng", data.get("merchant_name") or "—")
        st.metric("Ngày", _fmt_date(data.get("date")))
    with kv2:
        st.metric("Tổng tiền", _fmt_vnd(data.get("total_amount")))
        st.metric("Thanh toán", data.get("payment_method") or "—")

    # ── Items table ─────────────────────────────────────────────────────────
    items = data.get("items", [])
    if items:
        st.markdown("**Danh sách mặt hàng**")
        import pandas as pd
        df = pd.DataFrame([
            {
                "Tên hàng":   it.get("name", ""),
                "SL":         it.get("quantity", ""),
                "Đơn giá":    _fmt_vnd(it.get("unit_price")),
                "Thành tiền": _fmt_vnd(it.get("total")),
            }
            for it in items
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── All category scores ─────────────────────────────────────────────────
    all_scores = classification.get("all_scores", {})
    if all_scores:
        with st.expander("Điểm phân loại tất cả danh mục"):
            for cat, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                bar_color = CATEGORY_COLORS.get(cat, "#B2BEC3")
                icon_cat  = CATEGORY_ICONS.get(cat, "📦")
                st.markdown(
                    f"{icon_cat} **{cat}** — {score:.1%} "
                    + f'<span style="display:inline-block;width:{int(score*120)}px;'
                    + f'height:8px;background:{bar_color};border-radius:4px;vertical-align:middle"></span>',
                    unsafe_allow_html=True,
                )

    # ── Corrections ─────────────────────────────────────────────────────────
    corrections = data.get("corrections", [])
    if corrections:
        with st.expander(f"Auto-corrections ({len(corrections)})"):
            for c in corrections:
                st.warning(
                    f"**{c['field']}**: `{c['original']}` → `{c['corrected']}`  \n"
                    f"_{c['reason']}_"
                )

    # ── Metadata ────────────────────────────────────────────────────────────
    meta = data.get("meta", {})
    if meta:
        with st.expander("Pipeline metadata"):
            st.caption(
                f"Total: {meta.get('total_time_ms', 0):.0f} ms | "
                f"Cost: ${meta.get('extraction_cost_usd', 0):.5f} | "
                f"Cache: {'hit' if meta.get('extraction_from_cache') else 'miss'}"
            )
