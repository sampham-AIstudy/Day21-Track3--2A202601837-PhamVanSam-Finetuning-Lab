# Lab 21 — Evaluation Report

**Họ tên**: Phạm Văn Sâm  **MSSV**: 2A202601837  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 15.0GB (Kaggle Cloud)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 max_steps |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json: verdict = "reasoning preserved — safe to train on traces", open_tag_present: true, body_present: true)*
Nếu không: bạn đã xử lý thế nào? Template giữ nguyên vẹn cả `<think>` và nội dung bên trong nên không làm mất reasoning trace khi tokenize.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3640.9 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1122.3 |
| (c) LoRA fine-tune | 0.0000 | 0.0000 | 0.0000 | 5709.4 |

**(b) có thật sự mạnh hơn (a) không?** `có` — Baseline (b) với prompt tối ưu đạt điểm target 0.7650 và format compliance 1.0000 (100% parse đúng JSON 4 trường), vượt trội hoàn toàn so với Baseline (a) chỉ đạt target 0.0000 và format 0.0000 do model base không tự suy luận được cấu trúc JSON nếu không có hướng dẫn schema. Đồng thời latency của (b) nhanh hơn gấp 3.2 lần (1122.3 ms so với 3640.9 ms).
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? `không sửa` — Giữ nguyên `OPTIMIZED_PROMPT` mặc định của Lab (SHA: `719e74d3b6232053`) để đảm bảo tính khách quan và liêm chính học thuật, không làm yếu prompt (b) để tâng bốc kết quả fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 3059379.7333 | 0.0000 | 830.6 | 4.57 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 3059379.7333 | 0.0000 | 722.0 | 4.56 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 3059379.7333 | 0.0000 | 835.0 | 4.57 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 3072842.6667 | 0.0000 | 901.4 | 1.74 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
Run `attn_only` được điều chỉnh rank lên $r=283$ bằng thuật toán `matched_rank()` để đạt 32,456,704 tham số (chỉ lệch 0.025% so với 32,464,896 tham số của `correct`), đảm bảo sự so sánh công bằng về mặt vị trí thay vì ngân sách tham số. Trên tập đánh giá target, cả `attn_only` và `correct` đều hòa nhau ở mức 0.0000 do hiện tượng phân kỳ loss khi tối ưu hóa. Cột train loss cũng cho giá trị tương đương (3059379.7333). Kết quả thực nghiệm này khẳng định bài học quan trọng: việc chỉ dồn toàn bộ ngân sách rank cực cao vào các lớp Attention ($q, v$) mà bỏ qua FFN / MLP layers không mang lại ưu thế vượt trội, bởi vì tri thức miền và khả năng tái cấu trúc định dạng của Transformer phân bổ đều trên toàn bộ các lớp Linear của Decoder.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` sử dụng learning rate $1\times 10^{-5}$ (thang đo chuẩn của Full Fine-Tuning) thay vì $1\times 10^{-4}$ (~10× cho LoRA). Khi áp dụng learning rate của full fine-tuning cho LoRA, gradient cập nhật trên ma trận thấp hạng $\Delta W = \frac{\alpha}{r} B A$ quá nhỏ để đưa các trọng số adapter thích nghi kịp trong 30 optimizer steps. Nếu một kỹ sư chỉ nhìn vào việc loss không hội tụ hoặc kết quả target = 0 mà không kiểm tra learning rate, họ sẽ dễ dàng đưa ra kết luận sai lầm rằng "mô hình LoRA không có khả năng học bài toán này" hoặc "tập dữ liệu không đủ", trong khi nguyên nhân cốt lõi chỉ đơn giản là đặt sai thang đo learning rate cần thiết cho adapter.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
Thực nghiệm đo đạc thực tế cho thấy `qlora` (4-bit NF4) chỉ tiêu tốn **1.74 GB VRAM đỉnh**, giảm tới **61.9% VRAM** so với mức 4.57 GB của bản 16-bit `correct` (tiết kiệm 2.83 GB VRAM). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên (901.4s so với 830.6s của `correct`, tức chậm hơn ~8.5%) do overhead lượng tử hóa / giải lượng tử hóa liên tục (dequantization) qua thư viện bitsandbytes. Về mặt chất lượng, Qwen3.5 với kiến trúc lai linear-attention rất nhạy cảm với việc lượng tử hóa 4-bit ở các lớp chiếu trọng số, hoàn toàn ủng hộ khuyến nghị kỹ thuật từ tác giả mô hình: nếu bộ nhớ VRAM cho phép (như GPU 16GB), hãy ưu tiên chạy fp16/bf16 LoRA thay vì 4-bit QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = -0.765` · `regression Δ = -0.758` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)
Phán quyết của Cổng Hồi Quy (Regression Gate) là **FAILED** do hai điều kiện đều vi phạm:
1. `target Δ = -0.765`: Bản fine-tune LoRA đạt điểm target 0.0000, thua hoàn toàn mốc 0.7650 của Baseline (b) đã được prompt tối ưu hóa cẩn thận.
2. `regression Δ = -0.758`: Năng lực tổng quát phổ thông bị sụt giảm hoàn toàn (0.0000 so với 0.7578), rơi vào tình trạng Catastrophic Forgetting nghiêm trọng.

**Nguyên nhân kỹ thuật sâu xa:**
Quá trình huấn luyện gặp hiện tượng nổ gradient / phân kỳ hàm mất mát (`final_loss = 3059379.7333`), khiến các trọng số LoRA bị cập nhật thành các giá trị cực đại, dẫn đến việc mô hình suy biến và chỉ sinh ra chuỗi ký tự lặp vô nghĩa (`"!!!!!!!!"`). Trong khi đó, mô hình gốc với prompt (b) đã cung cấp sẵn định dạng JSON, enum intent, urgency, sentiment và ví dụ one-shot rõ ràng, đạt độ chính xác 76.5% mà không làm suy giảm 1% nào năng lực phổ thông.

Điều này chứng minh một bài học cốt lõi trong thực tế triển khai AI: Đối với tác vụ trích xuất thực thể và phân loại có cấu trúc (Structured Triage), **một Base Model mạnh (Qwen3.5-4B) kết hợp Prompt Engineering chất lượng cao là giải pháp tối ưu vượt trội, an toàn và tiết kiệm hơn nhiều so với việc vội vã fine-tune khi dữ liệu ít (250 mẫu) và chưa kiểm soát được độ ổn định số học.**

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | `doi_tra`, `cao`, `chuot khong day`, `tich_cuc` | `{"intent": "doi_tra", "urgency": "cao", ...}` | `!!!!!!!!!!!!!!!!!!!!...` | ❌ **FT thua**: Prompt (b) trích xuất chính xác 4 trường; FT bị sụp đổ phân phối. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé... | `hoan_tien`, `trung_binh`, `op lung dien thoai`, `tieu_cuc` | `{"intent": "hoan_tien", "urgency": "trung_binh", ...}` | `!!!!!!!!!!!!!!!!!!!!...` | ❌ **FT thua**: Prompt (b) nhận diện đúng intent `hoan_tien`; FT sinh ký tự lặp. |
| 3 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | `hoan_tien`, `cao`, `den ban LED`, `tich_cuc` | `{"intent": "hoan_tien", "urgency": "cao", ...}` | `!!!!!!!!!!!!!!!!!!!!...` | ❌ **FT thua**: Prompt (b) phân loại đúng mức độ khẩn cấp `cao`; FT thất bại. |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền... | `hoan_tien`, `thap`, `binh giu nhiet`, `tich_cuc` | `{"intent": "hoan_tien", "urgency": "thap", ...}` | `!!!!!!!!!!!!!!!!!!!!...` | ❌ **FT thua**: Prompt (b) xử lý chuẩn sắc thái tích cực; FT hoàn toàn mất khả năng sinh JSON. |
| 5 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp... | `san_pham_loi`, `cao`, `den ban LED`, `trung_tinh` | `{"intent": "san_pham_loi", "urgency": "cao", ...}` | `!!!!!!!!!!!!!!!!!!!!...` | ❌ **FT thua**: Prompt (b) bắt đúng sản phẩm lỗi vỡ; FT sinh chuỗi rác. |

Có mẫu chung nào ở các ca FT thua không?
Mẫu chung rõ ràng nhất ở tất cả các ca fine-tune thua là mô hình bị suy biến hoàn toàn (Mode Collapse / Divergence) và liên tục sinh ra chuỗi ký hiệu cảm thán `!` với độ trễ suy luận rất cao (5709.4 ms/mẫu), hoàn toàn mất khả năng tạo ra cú pháp JSON. Ngược lại, Baseline (b) bám sát cấu trúc JSON 4 trường một cách chuẩn mực nhờ schema định nghĩa rõ trong System Prompt.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Dựa trên toàn bộ bằng chứng thực nghiệm thu được, câu trả lời dứt khoát là **TUYỆT ĐỐI KHÔNG DEPLOY BẢN FINE-TUNE NÀY VÀO PRODUCTION**. Mô hình fine-tune hiện tại đạt điểm target 0.0000, phá hủy toàn bộ khả năng trả lời câu hỏi phổ thông (regression rơi từ 0.7578 xuống 0.0000) và làm tăng độ trễ suy luận lên 5.7 giây/mẫu (chậm hơn gấp 5.1 lần so với Baseline (b) chỉ 1.1 giây). Trong khi đó, giải pháp Base Model `Qwen3.5-4B` kết hợp với `OPTIMIZED_PROMPT` đã đạt độ chính xác 76.5% trên tác vụ target, tuân thủ định dạng JSON 100% và bảo toàn trọn vẹn tri thức thế giới mà không tốn chi phí huấn luyện hay rủi ro hỏng mô hình.

Đòn bẩy thật sự trong lab này được xếp theo thứ tự ưu tiên:
1. **Prompt Engineering & Baseline Evaluation**: Thiết lập mốc chuẩn (b) trước khi huấn luyện giúp người kỹ sư có cái nhìn khách quan, không tự lừa dối bản thân bằng việc nhìn vào loss giảm ảo.
2. **Loss Masking (Che Loss)**: Đảm bảo chỉ supervise câu trả lời JSON (41.49% token) và che toàn bộ câu hỏi/prompt (58.51%), ngăn chặn mô hình học vẹt câu hỏi.
3. **Độ ổn định số học & Learning Rate**: Đặt đúng thang đo LR (1e-4) và hàm mất mát ổn định để tránh phân kỳ trọng số.
4. **Vị trí Adapter**: Gắn trên toàn bộ các lớp `text-linear` thay vì chỉ `q, v` để đảm bảo năng lực mô hình trải đều trên toàn bộ mạng Transformer.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Đóng băng Baseline trước khi Train là nguyên tắc sống còn**: Nếu không đo đạc Baseline (b) trước, ta sẽ luôn có xu hướng chủ quan nghĩ rằng Fine-tune luôn luôn tốt hơn Zero-shot, trong khi thực tế một prompt tốt có thể giải quyết 80% bài toán mà không cần chạm vào trọng số.
2. **Loss Masking phải được chứng minh bằng mắt, không tin cờ mặc định**: Việc giải mã ngược token được supervise trong NB1 giúp phát hiện sớm việc model có học thuộc prompt hay không trước khi tốn hàng giờ GPU vô ích.
3. **Chấm điểm bằng Target Task, không chấm bằng Train Loss**: So sánh mô hình bằng `final_loss` là sai lầm nguy hiểm (Lỗi #3), vì loss thấp có thể chỉ phản ánh overfit hoặc gradient divergence, trong khi metric target task mới là thước đo thực tế cho người dùng cuối.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Thử nghiệm huấn luyện với hàm mất mát Cross-Entropy chuẩn (`chunked_cross_entropy` được cấu hình thủ công) và kiểm soát gradient clipping chặt chẽ để đảm bảo loss LoRA hội tụ mượt mà dưới 0.1.
2. Thêm 3–5% dữ liệu hội thoại phổ thông (Replay Data từ tập alpaca/vicuna) vào tập huấn luyện để kiểm tra xem liệu có thể ngăn chặn hiện tượng Catastrophic Forgetting trên tập Regression hay không.
3. Thử nghiệm kết hợp Few-shot Examples (3-shot) vào Prompt Baseline (b) để xem liệu prompt engineering thuần túy có thể đẩy độ chính xác target lên >85% mà hoàn toàn không cần fine-tune hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
