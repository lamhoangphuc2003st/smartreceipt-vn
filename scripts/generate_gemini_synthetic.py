"""
scripts/generate_gemini_synthetic.py
--------------------------------------
Sinh thêm ~1500 mẫu training tổng hợp bằng Gemini 2.0 Flash.

Chiến lược:
  - Phân tích phân phối hiện tại → sinh thêm cho các class yếu để đạt target/class.
  - Mỗi batch: yêu cầu Gemini sinh N dòng định dạng "{merchant} | {item1}, {item2}, ..."
  - Prompt bao gồm tiêu chí phân loại chi tiết từ proposal + ví dụ thực tế VN.
  - Model: gemini-2.0-flash (cân bằng quality / chi phí).
  - Output: data/training/phobert_synthetic_gemini.csv (APPEND vào existing nếu chạy lại)

Usage:
    python scripts/generate_gemini_synthetic.py
    python scripts/generate_gemini_synthetic.py --target 300 --batch-size 40
    python scripts/generate_gemini_synthetic.py --category "Giải trí" --count 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TRAIN_CSV     = PROJECT_ROOT / "data" / "training" / "phobert_training.csv"
OUTPUT_CSV    = PROJECT_ROOT / "data" / "training" / "phobert_synthetic_gemini.csv"
GEMINI_MODEL  = "gemini-2.5-flash"

CATEGORIES = [
    "Ăn uống", "Đi lại", "Mua sắm", "Giải trí",
    "Hoá đơn tiện ích", "Sức khoẻ", "Giáo dục",
    "Du lịch", "Nhà cửa", "Khác",
]

# ── Tiêu chí phân loại + gợi ý sản phẩm/merchant theo từng category ──────────

CATEGORY_PROMPTS: dict[str, dict] = {
    "Ăn uống": {
        "definition": (
            "Chi tiêu cho thức ăn, đồ uống tiêu thụ trực tiếp — ăn tại chỗ, mang về, "
            "đặt app (GrabFood/ShopeeFood), siêu thị khi items là thực phẩm."
        ),
        "merchants": [
            "Highlands Coffee", "Phúc Long", "The Coffee House", "Cộng Cà Phê",
            "Starbucks", "Phở Hùng", "Bún bò Huế Dì Ba", "KFC", "McDonald's",
            "Lotteria", "Jollibee", "Pizza Hut", "Domino's Pizza", "Gà Rán Texas",
            "Bún Đậu Mộc", "Nhà hàng Hải Sản", "Quán nhậu", "Trà sữa Tocotoco",
            "Gong Cha", "Tiger Sugar", "Quán cơm Bà Hoa", "Bánh mì Hà Nội",
            "GrabFood", "ShopeeFood", "WinMart", "Co.opmart", "AEON Việt Nam",
            "BigC", "Lotte Mart", "Bach Hoa Xanh", "Circle K", "GS25", "FamilyMart",
        ],
        "items": [
            "Cà phê sữa đá", "Trà sữa trân châu", "Phở bò tái", "Bún bò Huế",
            "Cơm tấm sườn", "Bánh mì thịt", "Gà rán", "Pizza hải sản",
            "Nước ép cam", "Sinh tố bơ", "Bánh ngọt", "Kem tươi",
            "Bò lúc lắc", "Lẩu thái", "Hải sản tươi sống",
            "Rau củ quả", "Thịt heo", "Cá tươi", "Trứng gà", "Sữa tươi",
            "Bia Tiger", "Pepsi", "Nước suối Aquafina",
        ],
    },
    "Đi lại": {
        "definition": (
            "Di chuyển hàng ngày: xăng dầu, taxi/grab xe, vé xe buýt/xe khách nội tỉnh, "
            "bãi đỗ xe, phí BOT, bảo dưỡng xe."
        ),
        "merchants": [
            "Petrolimex", "Shell", "Caltex", "PV Oil", "Saigon Petro",
            "Grab", "Be", "Gojek", "Mai Linh Taxi", "Vinasun Taxi",
            "Phương Trang", "Thành Bưởi", "xe buýt tuyến 12",
            "Bãi xe Bitexco", "Giữ xe nhà xe", "Trạm BOT cao tốc TP.HCM-Long Thành",
            "Honda Head", "Yamaha Town", "Tiệm vá xe Hùng",
            "Rửa xe Thành Đạt", "Dầu nhớt Castrol",
        ],
        "items": [
            "Xăng RON95", "Xăng RON92", "Dầu diesel B5",
            "Cước phí chuyến xe", "Vé xe buýt tháng",
            "Phí đậu xe máy", "Phí đậu ô tô",
            "Phí cầu đường", "Vé qua trạm BOT",
            "Thay dầu nhớt", "Vá xe lốp trước", "Rửa xe máy",
            "Nhớt động cơ 10W40", "Lọc gió xe máy",
        ],
    },
    "Mua sắm": {
        "definition": (
            "Mua hàng hoá phi thực phẩm cho cá nhân: quần áo, điện tử, mỹ phẩm, "
            "đồ dùng sinh hoạt (giấy vệ sinh, xà phòng, bột giặt), đồ chơi, quà tặng."
        ),
        "merchants": [
            "Zara", "H&M", "Uniqlo", "Routine", "Canifa", "Owen", "NEM",
            "Thế Giới Di Động", "Điện Máy Xanh", "FPT Shop",
            "Guardian", "Watsons", "Hasaki Beauty", "Innisfree",
            "Laneige", "The Face Shop",
            "Fahasa", "Nhà Sách Tiền Phong",
            "Shopee", "Lazada", "Tiki",
            "Bách Hoá Xanh", "WinMart", "Co.opmart",
            "Decathlon", "Vascara", "Bata",
        ],
        "items": [
            "Áo thun nam size L", "Quần jeans nữ", "Giày thể thao", "Túi xách da",
            "iPhone 15 Pro Max", "Tai nghe AirPods", "Sạc dự phòng Anker",
            "Kem dưỡng da Laneige", "Son môi 3CE", "Serum Vitamin C",
            "Nước hoa Chanel", "Mặt nạ dưỡng da",
            "Giấy vệ sinh Pulppy 10 cuộn", "Bột giặt Omo 3kg",
            "Nước rửa bát Sunlight", "Dầu gội Rejoice",
            "Đồ chơi Lego", "Búp bê Barbie",
        ],
    },
    "Giải trí": {
        "definition": (
            "Hoạt động vui chơi giải trí: rạp phim, karaoke, billiards, bowling, game, "
            "khu vui chơi, vé concert/thể thao. Kể cả đồ ăn/uống ĐI KÈM hoạt động giải trí."
        ),
        "merchants": [
            "CGV Crescent Mall", "CGV Vincom", "Lotte Cinema",
            "Karaoke Luxury", "Karaoke Doraemon", "Karaoke New Sky",
            "Billiards New Vip", "Bi-a Club 68",
            "Bowling Saigon Pearl", "Kingfun",
            "Dream City", "Kidzone", "Timezone",
            "GamePop", "Garena nạp thẻ", "VNG Store",
            "Sun World Bà Nà Hills", "Vinpearl Land",
            "SVĐ Thống Nhất", "Nhà hát TP.HCM",
        ],
        "items": [
            "Vé xem phim Avengers 2D", "Vé xem phim 3D",
            "Phòng karaoke VIP 2 giờ", "Tiền phòng karaoke",
            "Đồ uống trong karaoke (Pepsi, bia)",
            "Giờ chơi billiards", "Vé bowling 2 ván",
            "Thẻ game Timezone 200k", "Nạp Kim Cương Free Fire",
            "Vé vào cổng Vinpearl", "Vé tàu lượn siêu tốc",
            "Vé xem bóng đá AFF Cup", "Vé concert Sơn Tùng MTP",
        ],
    },
    "Hoá đơn tiện ích": {
        "definition": (
            "Hoá đơn dịch vụ thiết yếu hàng tháng: điện (EVN), nước, gas, internet, "
            "cáp truyền hình, điện thoại trả sau, phí quản lý chung cư."
        ),
        "merchants": [
            "EVN TP.HCM", "EVN Hà Nội", "EVNHCMC",
            "Công ty TNHH MTV Nước Sạch Hà Nội", "Sawaco",
            "PetroVietnam Gas", "Saigon Gas", "Thành Hà Gas",
            "VNPT", "Viettel", "FPT Telecom", "CMC Telecom",
            "VTVcab", "K+", "SCTV",
            "Ban Quản Lý Toà Nhà Masteri",
            "Điện lực Bình Thạnh", "Điện lực Cầu Giấy",
        ],
        "items": [
            "Tiền điện tháng 3/2024", "Điện năng tiêu thụ 215 kWh",
            "Tiền nước sinh hoạt tháng 2", "Nước sạch 12m3",
            "Gas bình 12kg", "Gas đường ống tháng 4",
            "Cước internet FPT 100Mbps tháng 5",
            "Cước viễn thông Viettel trả sau",
            "Cước truyền hình VTVcab",
            "Phí quản lý chung cư tháng 6",
            "Cước điện thoại cố định",
        ],
    },
    "Sức khoẻ": {
        "definition": (
            "Duy trì và cải thiện sức khoẻ: nhà thuốc, phòng khám/bệnh viện, "
            "thực phẩm chức năng, gym/yoga/bơi lội, dụng cụ y tế."
        ),
        "merchants": [
            "Pharmacity", "Long Châu", "An Khang",
            "Trung Tâm Y Tế Quận 1", "Bệnh Viện Đa Khoa Tâm Anh",
            "Phòng Khám Đa Khoa Hoàn Mỹ", "Nha Khoa Kim",
            "Mắt Kính 365", "Specsavers",
            "Gymmax Fitness", "California Fitness & Yoga",
            "Elite Fitness", "Club 4000",
            "Hồ Bơi Rạch Miễu",
        ],
        "items": [
            "Paracetamol 500mg hộp 10 vỉ", "Vitamin C 1000mg",
            "Amoxicillin 500mg", "Thuốc hạ sốt trẻ em",
            "Omega-3 Fish Oil 1000mg", "Collagen Peptide 10000mg",
            "Whey Protein Gold Standard 1kg",
            "Khám tổng quát", "Xét nghiệm máu cơ bản",
            "Siêu âm bụng tổng quát", "Chụp X-quang ngực",
            "Phí điều trị nha khoa", "Trám răng",
            "Tập gym tháng 1 tháng", "Gói yoga 10 buổi",
            "Nhiệt kế điện tử", "Máy đo huyết áp Omron",
        ],
    },
    "Giáo dục": {
        "definition": (
            "Học tập và nâng cao kiến thức: học phí trường/trung tâm, sách giáo khoa, "
            "khóa học online, văn phòng phẩm dùng cho học, thiết bị học tập."
        ),
        "merchants": [
            "ILA", "Anh văn Hội Việt Mỹ", "ACET Language School",
            "Trung Tâm Anh Ngữ YOLA", "Apax English",
            "Udemy", "Coursera",
            "Trường Đại Học FPT", "RMIT Việt Nam",
            "Trường THPT Lê Quý Đôn",
            "Nhà Sách Fahasa", "Nhà Sách Tiền Phong",
            "Thiên Long", "Bút Bi Hà Nội",
            "Topica", "MindX Technology School",
        ],
        "items": [
            "Học phí tiếng Anh tháng 4", "Học phí khóa IELTS 7.0",
            "Học phí HK1 năm 2024-2025",
            "Sách Tiếng Anh lớp 10", "Sách Toán nâng cao",
            "Giáo trình Marketing cơ bản",
            "Khóa học Python trên Udemy", "Khóa Data Science Coursera",
            "Vở ô ly 200 trang", "Bút bi Thiên Long 10 cây",
            "Thước kẻ, compa, ê ke bộ",
            "Máy tính Casio FX-580",
            "Học phí lớp học thêm Toán",
        ],
    },
    "Du lịch": {
        "definition": (
            "Chuyến đi xa khỏi nơi thường trú: vé máy bay, khách sạn/resort, "
            "tour trọn gói, vé tàu/xe đường dài liên tỉnh xa."
        ),
        "merchants": [
            "Vietnam Airlines", "VietJet Air", "Bamboo Airways",
            "Vinpearl Resort & Spa Nha Trang",
            "Mường Thanh Luxury Đà Nẵng",
            "Novotel Phú Quốc Resort",
            "Khách sạn Majestic Sài Gòn",
            "Saigontourist", "Vietravel",
            "Phương Trang (tuyến HCM-Đà Lạt)",
            "Tàu SE3 Thống Nhất", "Xe khách Kumho Samco",
        ],
        "items": [
            "Vé máy bay HAN-SGN khứ hồi", "Vé Vietnam Airlines HAN-DAD",
            "Phòng Deluxe 2 đêm", "Phòng Superior 1 giường đôi",
            "Tour Đà Lạt 3 ngày 2 đêm", "Tour Phú Quốc 4N3Đ trọn gói",
            "Vé xe khách HCM-Nha Trang giường nằm",
            "Vé tàu SE2 Hà Nội-TP.HCM",
            "Dịch vụ đưa đón sân bay",
            "Bảo hiểm du lịch quốc tế",
        ],
    },
    "Nhà cửa": {
        "definition": (
            "Nơi ở: nội thất (bàn ghế giường tủ), điện máy gia dụng lớn (tủ lạnh máy giặt điều hoà), "
            "vật tư sửa chữa, dịch vụ vệ sinh, đồ dùng bếp, cây cảnh trang trí."
        ),
        "merchants": [
            "IKEA Việt Nam", "Nội Thất Hoà Phát",
            "Nội Thất Dũng Quỳnh", "Phương Hoa Furniture",
            "Điện Máy Xanh", "Nguyễn Kim", "MediaMart",
            "Tiểu thủ công nghiệp Việt Nam",
            "Sơn Dulux", "Sơn Jotun",
            "Chợ Vật Liệu Xây Dựng Bình Dương",
            "Dịch vụ vệ sinh Sạch Sẽ",
            "Lò Vi Sóng Electrolux Service",
        ],
        "items": [
            "Bàn ăn 6 ghế gỗ sồi", "Ghế sofa da 3 chỗ",
            "Giường ngủ 1m8 đầu bọc da", "Tủ quần áo 4 cánh",
            "Tủ lạnh Samsung 300L inverter",
            "Máy giặt LG 9kg lồng ngang",
            "Điều hoà Daikin 1.5HP inverter",
            "Lò vi sóng Panasonic 25L",
            "Máy hút bụi Dyson V11",
            "Bộ nồi chảo chống dính Tefal",
            "Sơn nội thất Dulux 18L", "Gạch ốp lát 60x60",
            "Dịch vụ vệ sinh nhà 3 tầng",
            "Chậu cây sen đá", "Đèn trang trí LED",
        ],
    },
    "Khác": {
        "definition": (
            "Không thuộc 9 nhóm trên: nộp thuế/phí hành chính, giao dịch ngân hàng, "
            "từ thiện, bảo hiểm nhân thọ/xe/tài sản, dịch vụ pháp lý/công chứng, "
            "dịch vụ thú cưng, dịch vụ tang lễ, phí hội viên tổ chức, "
            "hoá đơn không rõ nội dung."
        ),
        "merchants": [
            "Cục Thuế Quận Bình Thạnh", "Cục Thuế TP.HCM",
            "Ngân hàng BIDV", "Vietcombank", "ACB", "Agribank",
            "Bảo hiểm Nhân thọ Prudential", "Bảo hiểm Bảo Việt", "Manulife VN",
            "Bảo hiểm PVI", "Bảo hiểm AAA xe ô tô",
            "Văn phòng Công chứng số 1", "Văn phòng Luật sư Minh Đức",
            "Tổ chức từ thiện Hoa Mai", "Quỹ Bảo trợ Trẻ em VN",
            "UBND Phường 12", "Kho bạc Nhà Nước",
            "Bưu điện Việt Nam", "ViettelPost", "GHTK",
            "Phòng khám thú y Doctor Pet",
            "Dịch vụ in ấn Minh Khoa",
            "Dịch vụ rửa xe chuyên nghiệp",
            "Hội Cựu chiến binh phường",
        ],
        "items": [
            "Thuế thu nhập cá nhân tháng 3", "Thuế VAT quý 1",
            "Lệ phí đăng ký kinh doanh hộ cá thể",
            "Phí công chứng hợp đồng mua bán nhà",
            "Phí làm hộ chiếu phổ thông",
            "Phí cấp lại CMND/CCCD",
            "Bảo hiểm nhân thọ năm 2024 - hợp đồng HP001",
            "Phí bảo hiểm xe ô tô bắt buộc",
            "Phí tư vấn pháp lý hợp đồng lao động",
            "Ủng hộ quỹ từ thiện miền Trung lũ lụt",
            "Tiêm phòng chó mèo - vaccine dại",
            "Khám và điều trị bệnh cho chó",
            "Phí gửi bưu kiện EMS 2kg",
            "Phí chuyển tiền ngân hàng quốc tế",
            "Phí thường niên thẻ tín dụng Visa",
            "Hội phí câu lạc bộ golf năm 2024",
            "Phí đỗ xe tháng toà nhà văn phòng",
            "Dịch vụ in ấn danh thiếp 500 tờ",
            "Phí sang tên xe máy",
        ],
    },
}


# ── Gemini client ─────────────────────────────────────────────────────────────

def get_gemini_client():
    api_key = os.getenv("Gemini_API_Key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found. Set Gemini_API_Key env var.")
    from google import genai
    return genai.Client(api_key=api_key)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(category: str, count: int) -> str:
    info = CATEGORY_PROMPTS[category]
    merchant_examples = ", ".join(info["merchants"][:10])
    item_examples = ", ".join(info["items"][:12])

    return f"""Bạn là chuyên gia tạo dữ liệu training cho bài toán phân loại chi tiêu Việt Nam.

## Nhiệm vụ
Sinh ra CHÍNH XÁC {count} dòng văn bản mô tả hoá đơn thuộc danh mục **{category}**.

## Định nghĩa danh mục
{info["definition"]}

## Format mỗi dòng
{{tên merchant}} | {{sản phẩm 1}}, {{sản phẩm 2}}, {{sản phẩm 3}}

HOẶC (khi không có merchant):
{{sản phẩm 1}}, {{sản phẩm 2}}, {{sản phẩm 3}}

HOẶC (khi chỉ có merchant, không có item list):
{{tên merchant}}

## Quy tắc bắt buộc
1. Mỗi dòng = 1 mẫu. Mỗi mẫu PHẢI đúng danh mục **{category}**.
2. Dùng tên merchant thực tế tại Việt Nam. Gợi ý: {merchant_examples}
3. Dùng sản phẩm/dịch vụ thực tế tại Việt Nam. Gợi ý: {item_examples}
4. ĐA DẠNG: khác nhau về merchant, sản phẩm, phong cách viết (đầy đủ / viết tắt / sai chính tả nhẹ).
5. Giữ nguyên tiếng Việt có dấu.
6. KHÔNG giải thích, KHÔNG đánh số, KHÔNG thêm header — chỉ trả về {count} dòng.
7. Mỗi dòng trên 1 hàng riêng.
8. 20% dòng không có merchant (chỉ có items).
9. 10% dòng chỉ có merchant (không có items) — tên merchant đủ rõ danh mục.
10. Tên sản phẩm giống thực tế trên hoá đơn VN: có thể viết tắt, có số lượng/trọng lượng.

## Ví dụ mẫu đúng format:
{_get_examples(category)}

Bắt đầu sinh {count} dòng:"""


def _get_examples(category: str) -> str:
    examples = {
        "Ăn uống": (
            "Highlands Coffee | Cà phê sữa đá, Bánh mì que\n"
            "Phở Hùng | Phở tái chín, Nước ngọt\n"
            "Cà phê Cộng | Cà phê cốt dừa, Trà đào cam sả"
        ),
        "Đi lại": (
            "Petrolimex | Xăng RON95 8.5L\n"
            "Grab | Chuyến xe từ Q1 đến Q7\n"
            "Bãi xe BigC Hoàng Mai | Phí giữ xe máy 4h"
        ),
        "Mua sắm": (
            "Uniqlo | Áo thun Heattech size M, Quần jogger nam\n"
            "Thế Giới Di Động | Samsung Galaxy A55 5G\n"
            "Guardian | Kem dưỡng ẩm Neutrogena, Son dưỡng Vaseline"
        ),
        "Giải trí": (
            "CGV Vincom | Vé xem phim Dune 2 - 2 vé, Bắp rang bơ lớn\n"
            "Karaoke Luxury VIP | Phòng đôi 3 giờ, Heineken 6 chai, Đĩa trái cây\n"
            "Billiards New Vip | Giờ chơi bi-a 2h30 bàn số 5, Nước ngọt 2 ly"
        ),
        "Hoá đơn tiện ích": (
            "EVN TP.HCM | Tiền điện tháng 3/2024 - 215 kWh\n"
            "FPT Telecom | Cước internet gói 100Mbps tháng 5\n"
            "Viettel | Cước điện thoại trả sau tháng 4"
        ),
        "Sức khoẻ": (
            "Pharmacity | Paracetamol 500mg, Vitamin C 1000mg, Oresol\n"
            "Nha Khoa Kim | Trám răng số 7, Lấy cao răng\n"
            "California Fitness | Gói gym tháng + yoga 8 buổi"
        ),
        "Giáo dục": (
            "ILA | Học phí tiếng Anh IELTS tháng 4\n"
            "Nhà Sách Fahasa | Sách giáo khoa Toán 10, Vở ô ly 200 trang x5\n"
            "Udemy | Python for Data Science bootcamp"
        ),
        "Du lịch": (
            "Vietnam Airlines | Vé HAN-SGN khứ hồi ngày 20/4\n"
            "Vinpearl Resort Nha Trang | Phòng Sea View 2 đêm\n"
            "Vietravel | Tour Đà Lạt 3N2Đ - 2 người lớn"
        ),
        "Nhà cửa": (
            "IKEA Việt Nam | Bàn ăn EKEDALEN 6 ghế, Đèn trần HEKTAR\n"
            "Điện Máy Xanh | Tủ lạnh Samsung RT43K6631BS 426L\n"
            "Dịch vụ vệ sinh Sạch Sẽ | Vệ sinh máy lạnh 3 cục, Dọn nhà 3 tầng"
        ),
        "Khác": (
            "Cục Thuế Quận 1 | Thuế TNCN tháng 3/2024\n"
            "Văn phòng Công chứng số 3 | Phí công chứng hợp đồng mua bán nhà\n"
            "Bảo hiểm Prudential | Phí bảo hiểm nhân thọ năm 2024"
        ),
    }
    return examples.get(category, "")


# ── Parse response ────────────────────────────────────────────────────────────

def parse_response(text: str, category: str) -> list[dict]:
    """Parse Gemini's line-by-line response into list of training samples."""
    lines = [ln.strip() for ln in text.strip().splitlines()]
    samples = []
    for line in lines:
        if not line:
            continue
        # Strip markdown numbering (1. / 1) etc.)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Strip leading/trailing quotes
        line = line.strip('"\'')
        if not line:
            continue
        # Basic sanity: must have at least a few chars
        if len(line) < 3:
            continue
        # Must not look like a header/comment
        if line.startswith("#") or line.startswith("//"):
            continue
        samples.append({
            "text": line,
            "label": category,
            "source": "gemini_synthetic",
            "confidence": 0.90,
        })
    return samples


# ── Generate one batch ────────────────────────────────────────────────────────

def generate_batch(
    client,
    category: str,
    count: int,
    max_retries: int = 6,
) -> list[dict]:
    from google.genai import types

    prompt = build_prompt(category, count)
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.9,          # high diversity
                    max_output_tokens=4096,
                ),
            )
            text = response.text or ""
            samples = parse_response(text, category)
            log.info(
                "  [%s] attempt %d — requested %d, parsed %d",
                category, attempt, count, len(samples),
            )
            if len(samples) >= max(1, count * 0.4):   # accept if got ≥40% of requested
                return samples[:count]
            log.warning("  Too few samples parsed (%d/%d), retrying...", len(samples), count)
        except Exception as exc:
            wait = min(2 ** attempt, 30)  # cap wait at 30s
            log.warning("  Attempt %d failed: %s (wait %ds)", attempt, exc, wait)
            time.sleep(wait)
    return []


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target",     type=int, default=250, help="Target samples per category")
    p.add_argument("--batch-size", type=int, default=40,  help="Samples per Gemini call")
    p.add_argument("--category",   type=str, default=None, help="Only generate for this category")
    p.add_argument("--count",      type=int, default=None, help="Override count for --category")
    p.add_argument("--output",     type=str, default=str(OUTPUT_CSV))
    p.add_argument("--reference",  type=str, default=None,
                   help="CSV to read existing counts from (default: phobert_training.csv)")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def load_existing_counts(reference_path: Path | None = None) -> Counter:
    path = reference_path or TRAIN_CSV
    if not path.exists():
        return Counter()
    import csv as _csv
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            counts[row["label"]] += 1
    return counts


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_gemini_client()

    # ── Decide how many to generate per category ──────────────────────────────
    ref_path = Path(args.reference) if args.reference else None
    existing = load_existing_counts(ref_path)
    log.info("Existing distribution:")
    for cat in CATEGORIES:
        log.info("  %-25s %d", cat, existing.get(cat, 0))

    if args.category:
        todo = {args.category: args.count or args.target}
    else:
        todo = {}
        for cat in CATEGORIES:
            need = max(0, args.target - existing.get(cat, 0))
            if need > 0:
                todo[cat] = need

    total_needed = sum(todo.values())
    log.info("=" * 50)
    log.info("Will generate %d samples across %d categories", total_needed, len(todo))
    log.info("=" * 50)

    # ── Open CSV for appending ────────────────────────────────────────────────
    file_exists = output_path.exists()
    all_generated = []
    total_generated = 0

    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source", "confidence"])
        if not file_exists:
            writer.writeheader()

        for cat, need in todo.items():
            log.info("\nGenerating %d samples for [%s] ...", need, cat)
            cat_samples = []
            remaining = need

            while remaining > 0:
                batch_size = min(args.batch_size, remaining)
                batch = generate_batch(client, cat, batch_size)
                if not batch:
                    log.warning("  No samples returned for [%s], skipping remaining.", cat)
                    break
                # Deduplicate within this run
                new_texts = {s["text"] for s in all_generated}
                unique = [s for s in batch if s["text"] not in new_texts]
                writer.writerows(unique)
                f.flush()
                cat_samples.extend(unique)
                all_generated.extend(unique)
                total_generated += len(unique)
                remaining -= len(unique)
                log.info("  [%s] +%d (total this category: %d, remaining: %d)",
                         cat, len(unique), len(cat_samples), max(0, remaining))
                # Rate limit buffer
                time.sleep(1.0)

            log.info("[%s] DONE: generated %d / %d", cat, len(cat_samples), need)

    log.info("=" * 50)
    log.info("Total generated: %d samples → %s", total_generated, output_path)
    log.info("=" * 50)

    # ── Final distribution report ─────────────────────────────────────────────
    import csv as _csv
    final: Counter = Counter(existing)
    for s in all_generated:
        final[s["label"]] += 1

    log.info("Final distribution (existing + new):")
    max_n = max(final.values()) if final else 1
    for cat in CATEGORIES:
        n = final.get(cat, 0)
        bar = "█" * int(n / max_n * 25)
        log.info("  %-25s %4d  %s", cat, n, bar)


if __name__ == "__main__":
    main()
