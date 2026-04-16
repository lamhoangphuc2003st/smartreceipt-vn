---
title: SmartReceipt VN
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# SmartReceipt VN

Trích xuất và phân loại chi tiêu từ hoá đơn Việt Nam tự động.

**Pipeline:** Gemini Vision (trích xuất) + PhoBERT (phân loại)

## Usage

Tải lên ảnh hoá đơn → nhận JSON có cấu trúc với tên cửa hàng, ngày, danh sách mặt hàng, tổng tiền và danh mục chi tiêu.

## Tech Stack

- Gemini Vision API (VLM extraction)
- PhoBERT fine-tuned (expense classification)
- FastAPI backend (Railway)
- Streamlit frontend (HF Spaces)
