# Edge Cases — SmartReceipt VN

Tài liệu này ghi lại các trường hợp đặc biệt mà hệ thống gặp phải trong quá trình phát triển, cách xử lý, và lý do quyết định. Mỗi case được phát hiện qua quá trình review `pre_golden.json` hoặc chạy pipeline thực tế.

---

## EC-001 — Phí dịch vụ / tiền giờ nằm ngoài bảng items

**Phát hiện:** `pos_clean_0030.jpg` — KARAOKE DORAEMON  
**Ngày ghi nhận:** 2026-04-09

### Mô tả

Một số loại hóa đơn (karaoke, bi-a, phòng hát, nhà hàng có phòng VIP, khách sạn) tách cấu trúc thành **2 block riêng biệt**:

```
Block 1 — Phí dịch vụ (ngoài bảng)
  Tiền giờ VIP 2 (1 giờ 1 phút): 610,000

Block 2 — Bảng items (đồ uống/đồ ăn)
  Sài gòn lager lon x24 = 336,000
  Sài gòn đỏ      x5  =  60,000
  ...
  T.CỘNG items:          624,000

TỔNG CỘNG:             1,234,000
```

### Lỗi AI hay mắc

AI chỉ đọc bảng items dạng bảng (table), bỏ sót block phí dịch vụ phía trên → `sum(items) = 624,000 ≠ total_amount = 1,234,000`. Validator toán học sẽ bắt được sai lệch này.

### Cách xử lý đúng

Đưa **tất cả** các khoản phí vào `items`, kể cả phí dịch vụ/giờ/phòng không nằm trong bảng chính:

```json
{
  "name": "Tiền giờ VIP 2 (1 giờ 1 phút)",
  "quantity": 1,
  "unit_price": 610000,
  "total": 610000
}
```

**Bất biến cần giữ:** `sum(item.total for item in items) == total_amount` (sai lệch ≤ 1 VND do làm tròn).

### Các merchant hay gặp pattern này

| Loại | Ví dụ khoản phí ngoài bảng |
|------|---------------------------|
| Karaoke | Tiền giờ, tiền phòng VIP |
| Bi-a / Snooker | Tiền bàn, tiền giờ |
| Nhà hàng phòng riêng | Phí đặt phòng, phí phục vụ |
| Khách sạn | Tiền phòng, phí dịch vụ |
| Bãi đỗ xe | Phí giữ xe (đôi khi in trước bảng) |

### Phần mềm tạo ra pattern này

**Vietbill.vn** — phần mềm tính tiền phổ biến cho bi-a, karaoke, cà phê bàn tại Việt Nam. Nhận dạng qua dòng chữ nhỏ góc trên phải: *"In bởi Vietbill.vn"*. Layout chuẩn của Vietbill luôn tách:

```
[Bảng items]  →  Tổng dịch vụ: X
               + Tiền giờ:     Y
               = Thanh toán:   X + Y
```

Các ảnh từ bi-a, karaoke dùng Vietbill sẽ **luôn** có sai lệch này nếu AI không đọc phần dưới bảng. Ước tính ảnh hưởng: toàn bộ nhóm `01_pos_clean` có nguồn gốc bi-a/karaoke/cà phê bàn.

**Ví dụ đã xác nhận:**
- `pos_clean_0030.jpg` — KARAOKE DORAEMON, tiền giờ 610,000
- `pos_clean_0034.jpg` — BILLIARDS CLUP NEW VIP, tiền giờ 25,000

### Ảnh hưởng đến category

Khi có cả phí dịch vụ lẫn đồ uống/ăn, áp dụng quy tắc §6.3 trong proposal:
- Lấy nhóm chiếm tỷ trọng **lớn nhất theo tổng tiền**.
- `pos_clean_0030`: 610k (giờ phòng - Giải trí) vs 624k (đồ uống - Ăn uống) → gần bằng nhau → ưu tiên **dịch vụ chính** là karaoke → **Giải trí**.

### Rule đã thêm vào prompt

`prelabel_golden.py` — `USER_PROMPT`, bước 4:
> "Nếu có tiền giờ, tiền phòng, phí dịch vụ nằm ngoài bảng items — đưa vào items như một dòng riêng."

---

## EC-002 — Siêu thị / minimart: Ăn uống vs Mua sắm

**Phát hiện:** `pos_clean_0000.jpg`, `pos_clean_0002.jpg`, `pos_clean_0003.jpg`  
**Ngày ghi nhận:** 2026-04-09

### Mô tả

AI nhìn thấy merchant là VinMart / MINIMART → tự gán **Mua sắm**, bỏ qua nội dung items thực tế là thực phẩm.

### Cách xử lý đúng

Phân loại dựa trên **sản phẩm**, không phải tên merchant. Xem §6.3 trong `docs/proposal.md`:
- ≥70% giá trị là thực phẩm/đồ uống → **Ăn uống**
- Items là đồ dùng, vật dụng → **Mua sắm**

---

## EC-003 — Vé điện tử / xác nhận đặt vé bị nhầm là "not_a_receipt"

**Phát hiện:** `pos_clean_0045.jpg` — CineStar Bình Dương  
**Ngày ghi nhận:** 2026-04-09

### Mô tả

Ảnh chụp màn hình xác nhận đặt vé điện tử (e-ticket, booking confirmation) gửi qua app/email. Giao diện khác hoàn toàn so với POS receipt truyền thống — không có logo cửa hàng in đậm, không có bảng items dạng bảng, có QR code lớn — khiến AI nhận dạng nhầm là "not_a_receipt".

### Tại sao đây vẫn là receipt hợp lệ

Có đủ các trường giao dịch:
- Merchant: rạp chiếu phim (CineStar Bình Dương)
- Ngày giờ chiếu = ngày giờ giao dịch
- Dịch vụ: số vé, tên phim, số ghế
- Tổng tiền: 110,000 VND

### Cách xử lý đúng

Đọc như một receipt bình thường. Mỗi loại vé là một line item:

```json
{
  "name": "Vé xem phim Tee Yod: Quỷ Ăn Tạng 3 (Ghế E04, E03)",
  "quantity": 2,
  "unit_price": 55000,
  "total": 110000
}
```

`unit_price = total_amount / quantity` khi không hiển thị đơn giá riêng.

### Các dạng vé điện tử hay gặp trong dataset

| Loại | Merchant ví dụ | Trường đặc trưng |
|------|---------------|-----------------|
| Vé rạp phim | CGV, Lotte Cinema, Galaxy, CineStar | Mã đặt vé, tên phim, số ghế |
| Vé xe | Futa Bus, Phương Trang | Mã vé, tuyến đường, giờ khởi hành |
| Vé sự kiện | Ticketbox, Zalo | Mã QR, tên sự kiện |

### Category

Rạp chiếu phim → **Giải trí** (rule 2a trong §6.3).

### Rule đã thêm vào prompt

`prelabel_golden.py` — `USER_PROMPT`, Hard constraints:
> "Digital booking confirmations and e-tickets (xác nhận đặt vé, vé điện tử) with a total amount ARE valid receipts — do not mark as not_a_receipt."

---

## EC-004 — Biên lai chuyển tiền ngân hàng (bank transfer confirmation)

**Phát hiện:** `pos_clean_0046.jpg` — Techcombank  
**Ngày ghi nhận:** 2026-04-09

### Mô tả

Ảnh chụp màn hình xác nhận chuyển tiền thành công từ app ngân hàng (Techcombank, VietcomBank, MB Bank, v.v.). Phổ biến ở Việt Nam khi thanh toán qua chuyển khoản thay cho POS.

Cấu trúc điển hình:
```
Chuyển thành công  VND 350,000
Người nhận:        PHAM THANH HUYEN  /  0886644228
Ngân hàng nhận:    Cake by VPBank
Lời nhắn:          mua claude pro
Ngày chuyển:       7 Th4, 2026 lúc 21:40
Mã giao dịch:      FT26098851056619
```

### Cách xử lý từng trường

| Trường | Cách lấy | Ghi chú |
|--------|----------|---------|
| `merchant_name` | `null` | Nếu người nhận là cá nhân (họ tên viết hoa, không có MST). Nếu là tên doanh nghiệp rõ ràng thì dùng tên đó |
| `date` / `time` | Trường "Ngày chuyển" | Đây là thời điểm giao dịch thực tế |
| `items[0].name` | Trường "Lời nhắn" | Dùng nội dung lời nhắn làm mô tả item nếu có |
| `total_amount` | Số tiền chuyển | Luôn hiển thị rõ |
| `payment_method` | `"transfer"` | Luôn là transfer với loại biên lai này |
| `category` | Suy từ lời nhắn nếu rõ ràng, còn lại → `"Khác"` | Lời nhắn là do người dùng tự ghi, không được xác minh bởi merchant |

### Quy tắc phân category từ lời nhắn

Chỉ dùng lời nhắn để suy category khi nội dung **rõ ràng và không mơ hồ**:

| Lời nhắn | Category |
|----------|----------|
| "tiền phòng tháng 4", "cọc nhà" | Nhà cửa |
| "học phí", "tiền khóa học" | Giáo dục |
| "tiền xăng", "đổ xăng" | Đi lại |
| "mua thuốc", "viện phí" | Sức khoẻ |
| Không có lời nhắn / mơ hồ ("chuyển tiền", tên người) | Khác |
| "mua claude pro", tên phần mềm/app | Mua sắm (nếu rõ là mua hàng số) |

### Ví dụ đã xác nhận
- `pos_clean_0046.jpg`: lời nhắn "mua claude pro" → giữ `"Khác"` vì không xác minh được merchant

---

## EC-005 — Biên lai thanh toán qua ví điện tử (MoMo, ZaloPay, VNPay)

**Phát hiện:** `pos_clean_0047.jpg` — CTĐL TP Cần Thơ thanh toán qua MoMo  
**Ngày ghi nhận:** 2026-04-09

### Mô tả

Ảnh chụp màn hình "Chi Tiết Giao Dịch" từ app ví điện tử (MoMo, ZaloPay, VNPay, ViettelPay). Phổ biến khi thanh toán hoá đơn tiện ích (điện, nước, internet) hoặc mua hàng online.

Cấu trúc MoMo điển hình:
```
[Logo nhà cung cấp]  CTĐL TP CẦN THƠ
                     -202,338đ
Trạng thái:          Thành công
Thời gian:           04:34 - 02/04/2026
Tài khoản/thẻ:       Ví MoMo          ← nhận dạng e_wallet
Nhà cung cấp:        CTĐL TP Cần Thơ  ← merchant_name
Kỳ thanh toán:       Tiền điện tháng 3/2026  ← item name
```

### Cách xử lý

| Trường | Lấy từ | Ghi chú |
|--------|--------|---------|
| `merchant_name` | "Nhà cung cấp" | Tên tổ chức, không phải tên app ví |
| `items[0].name` | "Kỳ thanh toán" | Mô tả dịch vụ được thanh toán |
| `payment_method` | `"e_wallet"` | Khi thấy "Ví MoMo / ZaloPay / VNPay / ViettelPay" |
| `category` | Suy từ nhà cung cấp | CTĐL/EVN → Hoá đơn tiện ích; trường học → Giáo dục |

### Lưu ý về privacy

Màn hình MoMo/ZaloPay thường hiển thị **thông tin cá nhân** của chủ tài khoản:
- Tên khách hàng
- Mã khách hàng
- Địa chỉ

**Bắt buộc blur** các trường này trước khi commit ảnh vào repo (theo CLAUDE.md). Không push raw ảnh có PII lên GitHub.

### Ví dụ đã xác nhận
- `pos_clean_0047.jpg`: MoMo thanh toán tiền điện CTĐL TP Cần Thơ 202,338 VND → `payment_method: "e_wallet"`, `category: "Hoá đơn tiện ích"`

---

## EC-006 — Hóa đơn VAT chụp thiếu header (partial invoice)

**Phát hiện:** `pos_clean_0048.jpg` — hóa đơn GTGT sửa chữa bếp từ  
**Ngày ghi nhận:** 2026-04-09

### Mô tả

Ảnh chỉ chụp **phần bảng items** của hóa đơn GTGT, phần header chứa tên công ty, địa chỉ, MST, số hóa đơn, ngày tháng bị cắt ra ngoài khung.

```
[PHẦN BỊ CẮT: tên merchant, ngày, số HĐ]
─────────────────────────────────────────
STT | Tên hàng hóa | ĐVT | SL | Đơn giá | Thành tiền | Thuế suất | Tiền thuế
 1  | Sửa chữa bếp từ...                                              ← CÒN LẠI
 2  | Chi phí nhân công...
─────────────────────────────────────────
Tổng hợp | Trước thuế | Tiền thuế | Cộng thanh toán
```

### Cách xử lý

- `merchant_name`: `null` — không đọc từ watermark (in chìm, không phải thông tin chính thức)
- `date`: `null` — không có trong ảnh
- `items`, `subtotal`, `vat`, `total_amount`: trích xuất bình thường từ bảng
- `confidence`: `"medium"` — có items đầy đủ nhưng thiếu metadata quan trọng

### Category khi thiếu merchant

Phân loại dựa vào **nội dung items**, không cần merchant:
- "Sửa chữa bếp từ", "nhân công sửa chữa" → **Nhà cửa**
- Nếu items mô tả đủ rõ mục đích → dùng category đó với confidence "medium"
- Nếu items quá chung chung ("hàng hóa", "dịch vụ") → **Khác**

### Hướng xử lý trong pipeline thực tế

Validator sẽ flag các entry có `merchant_name: null` + `date: null` là `needs_human_review: true`. Không reject — vẫn trả về items và total hợp lệ.

---

---

## EC-007 — Hóa đơn viết tay: date accuracy thấp (63.6%)

**Phát hiện:** exp_009 full pipeline eval — nhóm `04_handwritten`  
**Ngày ghi nhận:** 2026-04-15

### Mô tả

Hóa đơn viết tay (quán cơm, chợ, dịch vụ cá nhân) có ngày tháng viết không theo chuẩn, nét chữ không đều. Gemini gặp khó với:
- Ngày viết dạng "16/8/18", "16-8-2018", "ngày 16 tháng 8" (không chuẩn ISO)
- Chữ số dễ nhầm: "1" vs "7", "6" vs "0", "3" vs "8"
- Ngày không có → AI đoán ngày hiện tại (sai)

### Ảnh hưởng đến pipeline

Date accuracy: 63.6% (so với 88–96% ở các nhóm khác). E2E nhóm này: 36.4%.

### Cách xử lý hiện tại

- Validator không reject date sai format — trả về `null` thay vì crash
- Trường hợp date parse fail: `date = null`, `validation_warnings` ghi nhận

### Accepted limitation

Cải thiện cần: preprocessing binarization (tăng độ tương phản chữ tay) hoặc prompt cụ thể hơn cho handwritten. Chưa implement — documented là known limitation.

---

## EC-008 — VAT invoice: tên merchant dài và đa dạng (70.6% accuracy)

**Phát hiện:** exp_009 full pipeline eval — nhóm `02_vat_invoice`  
**Ngày ghi nhận:** 2026-04-15

### Mô tả

Hóa đơn GTGT có tên merchant là tên pháp lý đầy đủ của doanh nghiệp (thường rất dài):
```
CÔNG TY CỔ PHẦN THƯƠNG MẠI DỊCH VỤ XĂNG DẦU ĐỒNG THÁP
→ AI trích xuất: "Cty CP TM DV Xăng Dầu Đồng Tháp"  (viết tắt)
→ Ground truth:  "CÔNG TY CỔ PHẦN THƯƠNG MẠI..."       (đầy đủ)
→ Fuzzy match fails → merchant = wrong
```

### Ảnh hưởng

Merchant accuracy 70.6% → kéo E2E xuống 58.8% cho nhóm này.

### Cách xử lý hiện tại

Validator fuzzy-match vs `data/merchants.json` nhưng database chỉ có ~65 entries — tên công ty GTGT không có trong database. Correction không xảy ra.

### Accepted limitation

Cần mở rộng `merchants.json` với tên pháp lý doanh nghiệp phổ biến. Future work.

---

## EC-009 — File không phải JPEG/PNG bị ghi nhầm extension trong golden set

**Phát hiện:** exp_009 — `low_quality_0003.gif` SKIP  
**Ngày ghi nhận:** 2026-04-15

### Mô tả

`data/golden_test_set.json` ghi `"image": "low_quality_0003.gif"` nhưng file thực là `.jpg`. Pipeline bỏ qua (SKIP) vì không tìm thấy file.

### Fix đã áp dụng

Sửa trực tiếp trong `golden_test_set.json`: `.gif` → `.jpg` (commit riêng với note "fix typo").

### Bài học

Khi thêm entry vào golden set, verify filename khớp với file thực bằng:
```bash
ls data/raw/05_low_quality/ | grep <basename>
```

---

_Cập nhật file này mỗi khi phát hiện pattern mới trong quá trình review golden test set._
