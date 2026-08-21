# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Sức mạnh thực tế của Baseline (b) - Prompt tối ưu: chỉ với việc mô tả rõ schema, enum các trường và 1 ví dụ cụ thể trong system prompt, Base model `Qwen3.5-4B` chưa hề qua huấn luyện đã đạt ngay 76.5% độ chính xác target và 100% tuân thủ định dạng JSON với tốc độ 1.1s/mẫu.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở việc kiểm soát môi trường thực thi và thư viện (từ việc phân quyền Windows `pytest`, cấu hình GPU trên Kaggle, cho đến sự không tương thích của TRL trên Python 3.12). Điều này không nằm trong dự đoán ban đầu vì tôi từng nghĩ phần lớn thời gian sẽ nằm ở việc chỉnh siêu tham số LoRA.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi tin rằng: "Cứ bài toán phân loại/trích xuất chuyên biệt là bắt buộc phải fine-tune thì mô hình mới làm được, và loss giảm càng sâu thì mô hình càng thông minh". Giờ tôi hiểu rằng Prompt Engineering có thể vượt trội hơn fine-tune nếu base model đủ mạnh, và train loss hoàn toàn có thể lừa dối người làm AI nếu không đánh giá trực tiếp trên bài toán mục tiêu.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để phân tích lỗi môi trường `pytest`, hỗ trợ debug các lỗi versioning của `trl`/`transformers`, và tổng hợp, kiểm tra chéo các chỉ số giữa `results/` và `REPORT.md`. AI assistant ban đầu đã gợi ý tham số `loss_type="nll"` mà không lường trước việc model logits cần xử lý đúng cross-entropy trên môi trường cloud.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm sẽ là: Xây dựng tập đánh giá chuẩn (Golden Evaluation Set gồm cả bài toán mục tiêu và bài toán tổng quát) và đo đạc Baseline (b) với một System Prompt được thiết kế tối ưu nhất có thể. Nếu Baseline (b) đã giải quyết tốt bài toán với chi phí và độ trễ chấp nhận được, tôi sẽ tư vấn khách hàng chưa cần tốn ngân sách fine-tune.
