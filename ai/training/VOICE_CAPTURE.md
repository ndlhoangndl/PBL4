# Thu giọng đọc OTP tiếng Việt

Mục tiêu là nhận dạng nội dung: người nói đọc sáu chữ số bằng tiếng Việt, Whisper phiên âm lời đọc rồi chuẩn hóa thành chuỗi sáu số. Không phân loại danh tính người nói. Số 0 đầu phải được giữ nguyên, ví dụ `không năm bảy tám bốn ba` → `057843`.

## Thu âm

Mở `SmartDoor/scripts/Collect-Voice.cmd`. Nhập một mã giả danh cố định cho người nói (ví dụ `speaker-001`), chọn micro, chọn tốc độ đọc `slow`, `normal` hoặc `fast`, đọc OTP ngẫu nhiên đang hiển thị, nghe lại và chỉ lưu nếu transcript khớp hoàn toàn. Công cụ ghi tốc độ đọc vào manifest. Mỗi buổi dùng mã người/micro nhất quán; mở buổi mới để đổi người hoặc micro.

Thu nhiều người nói, nhiều phiên và đủ các tốc độ đọc. Mục tiêu ban đầu cho pilot lớp học: tối thiểu 6 người nói, 3 buổi/người, 10 OTP khác nhau cho mỗi tốc độ trong mỗi buổi. Như vậy có khoảng 540 mẫu; nếu thiếu người hoặc mẫu thì ghi rõ giới hạn thay vì tuyên bố hoạt động với mọi giọng. Chọn cả giọng nam/nữ, âm sắc và tốc độ tự nhiên khác nhau; thêm điều kiện micro/phòng khác nhau ở các buổi.

Các số được sinh ngẫu nhiên và cần bao phủ đủ chữ số 0–9 ở nhiều vị trí, gồm số 0 đầu. Không lặp lại cùng mã OTP trong một buổi. Không dùng mật khẩu mở cửa thật trong dữ liệu thu.

## Train và đánh giá

Dùng `pbl4_training/train_whisper.py` cho manifest nhiều buổi. Trainer áp dụng speed perturbation ngẫu nhiên chỉ trên train: 0.85×, 0.9×, 1.0×, 1.1×, 1.15×; validation/test vẫn là audio thật chưa biến đổi. Phép biến đổi đổi cả tốc độ và cao độ một chút, nên phải thu thêm giọng nói chậm/nhanh thật. Bản tăng cường sinh trong lúc train và không được đưa sang validation/test.

Tách theo `speaker_id` để kiểm tra người nói chưa thấy khi mục tiêu là hỗ trợ giọng bất kỳ. Với ít dữ liệu hơn, có thể tách theo phiên và chỉ kết luận về những người đã thu. Báo cáo OTP exact accuracy cùng WER/CER trên test; không dùng accuracy trên chính file train làm kết quả tổng quát. Kiểm tra riêng các tốc độ chậm/vừa/nhanh và trường hợp số 0 đầu.

Bản Whisper Tiny pilot trước đây có 20 mẫu một người/một buổi, 5 epoch; 10/20 khớp OTP trên chính tập train. Đây là bằng chứng pipeline đã cập nhật decoder, chưa phải đánh giá người nói mới. Xem [WHISPER_PILOT_DEMO.md](WHISPER_PILOT_DEMO.md).
