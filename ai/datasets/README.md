# Thu thập dữ liệu cho SmartDoor

## Ảnh khuôn mặt

Chụp bằng `scripts/Collect-Faces.cmd`; dùng `scripts/Collect-Faces-Person-002.cmd` cho người thứ hai. Ảnh lưu cục bộ trong `face-capture/<mã-người>/<buổi>/`. Mỗi ảnh cần được xem lại; dữ liệu nhận diện/chống giả mạo còn phải duyệt vùng mặt từ trán đến cằm. Xem [FACE_CAPTURE.md](../training/FACE_CAPTURE.md).

## Giọng nói đọc OTP

Mở `scripts/Collect-Voice.cmd`. Bản ghi lưu dưới `audio/<speaker>/<buổi>/`, còn `whisper.jsonl` chứa transcript tiếng Việt, OTP sáu chữ số và metadata. Nghe lại từng mẫu, giữ nguyên số 0 đầu và chỉ dùng câu đọc dành cho thu dữ liệu, không dùng OTP đang hoạt động. Xem [VOICE_CAPTURE.md](../training/VOICE_CAPTURE.md).

Hiện có một buổi với 20 câu đã được người thu xác nhận. Whisper Tiny đã được fine-tune thử trên buổi này; metric pilot được tính trên chính dữ liệu train. Thu thêm buổi khác để đánh giá độc lập trước khi kết luận hiệu quả.

## Quy tắc lưu trữ

Dữ liệu sinh trắc học giữ trong `ai/datasets` trên máy cục bộ. Không commit ảnh, audio, embedding hay dữ liệu cá nhân. Dùng mã giả danh thay tên thật trong manifest. Audio chỉ phục vụ nhận dạng nội dung chữ số, không dùng để xác minh danh tính người nói.
