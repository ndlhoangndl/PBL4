# Biên bản kiểm tra bản bàn giao

Ngày theo phiên làm việc: **11/10/2026, Asia/Saigon**. Các kết quả dưới chỉ là kiểm tra phần mềm.

## Môi trường đã khảo sát

- Windows x64; .NET SDK 8.0.419, Microsoft.WindowsDesktop.App 8.0.25, dotnet tại `%USERPROFILE%/.dotnet` và chưa có trong PATH.
- SQL Server Express 2022, instance mặc định `.`; đã kết nối bằng Windows Integrated Security. Không xác nhận có LocalDB.
- Python 3.12.10 x64. Đã tạo venv riêng và cài requirements gốc của bộ AI; phiên bản đầy đủ trong `ai-requirements.lock.txt`.
- Không có AGENTS.md trong workspace/thư mục cha đã khảo sát. Không dùng subagent.

## Đã chạy

- Build toàn solution .NET: **0 lỗi, 0 cảnh báo** ở bản build sạch.
- **27 tests passed, 0 failed, 0 skipped** với SQL integration và live AI đều được bật. Kết quả có trong `TestResults/final.trx`; các lần trước nằm cạnh file này. Kiểm thử nghiệp vụ, hợp đồng HTTP, SQL và model thật được tách rõ trong source.
- OTP giữ số 0 đầu; đúng/sai/hết hạn/đã dùng/thu hồi; giới hạn thử; không ghi mã vào sự kiện/database; cửa lỗi vẫn giữ OTP Consumed.
- Hai consumer trong test double và **8 request đồng thời trên SQL Server thật** dùng cùng OTP: đúng một Accepted, các request còn lại Consumed.
- Người bị vô hiệu hóa; danh tính hoặc mẫu bị thay trong lúc suy luận; đăng ký/xóa một phía thất bại; slot thứ 5 bị từ chối.
- Kết quả AI đến muộn sau hủy/hết hạn bị bỏ; hủy khi cửa đang đợi ngăn mở muộn; 10 lệnh đồng thời trùng ID chỉ tăng một lượt mở.
- Lệnh không phản hồi/mất kết nối được hoàn tất có thời hạn, không mở muộn. Đây là MockDoorController, chưa chứng minh khả năng đáp ứng UI hoặc thiết bị thật.
- Client multipart voice/enroll đúng field gốc, giữ session ID, không giữ transcript trong DTO voice. AI thật không cấp quyền mặt kể cả response hợp đồng giả lập nói liveness passed.
- SQL EF schema được tạo thật; luồng người dùng Mock → quyền bị chặn → OTP với SQL thật; phục hồi phiên dở dang khi khởi động lại. Database test tên GUID được dọn sau test; không seed người giả vào database ứng dụng.
- Kiểm tra tích hợp trước khi sắp xếp: model gốc **PASS SHA256** theo manifest. Hiện SmartDoor dùng `ai/models-manifest.json`; `scripts/start_ai.py` xác minh model bundle đó trước khi nạp.
- FastAPI đã khởi động với YuNet/SFace/Whisper gốc bằng launcher cache ASCII. `GET /health` ready, `GET /users` rỗng trên database AI thử riêng. RealAiClient gửi ảnh trống và nhận lỗi đúng một mặt; gửi âm thanh im lặng 1 giây và nhận retry, digits=null.
- Python launcher/input-test scripts đã qua kiểm tra cú pháp.

## Giới hạn và phần chưa kiểm tra

- Đã thử khởi chạy tiến trình WinForms với runtime hiện có và database SQL smoke riêng. **Chưa xác nhận hoàn chỉnh bố cục hoặc thao tác bằng UI**: Computer Use bị người dùng dừng bằng Escape. Tiến trình ẩn/database smoke đã được dọn; không tính việc khởi chạy là kiểm thử UI đạt.
- Một lần chạy VSTest trong sandbox bị testhost connection timeout; các lần chạy cho phép kết nối SQL/localhost đã thực thi thành công đầy đủ. Không coi lần abort là pass.
- Chưa dùng ảnh người thật để đăng ký/nhận diện qua desktop, chưa phiên âm bản đọc OTP tiếng Việt thực tế, chưa kiểm tra positive enrollment/voice với model thật.
- Ảnh trống và silence là đầu vào tổng hợp cho kiểm tra từ chối; không phải dữ liệu sinh trắc học và không phản ánh độ chính xác AI.
- Chưa có webcam/microphone capture, board Arduino, firmware, serial, cảm biến chốt hoặc khóa thật; chưa kiểm thử phần cứng, chống giả mạo, độ chính xác/độ trễ trên thiết bị ngoài.
- Chưa đo/hiệu chỉnh threshold và margin. Real face chỉ trả ứng viên và yêu cầu OTP.
- Chưa kiểm thử di chuyển DPAPI key giữa tài khoản/máy, cấu hình SQL khác, package distribution/self-contained installer hoặc quản trị nhiều desktop.
- API Python chưa có template revision: thay mẫu trực tiếp bên ngoài ứng dụng với cùng tên/số lượng không phát hiện đầy đủ. Chỉ dùng một desktop và không ghi SQLite/SQL bằng công cụ khác khi ứng dụng đang xác thực.

## Dữ liệu kiểm thử và trạng thái bàn giao

Không lưu model mới, embedding giả, ảnh khuôn mặt hay âm thanh người thật. Giữ nguyên bộ AI, giấy phép và manifest. `TestResults` chỉ có TRX, input trống/im lặng và SQLite AI thử rỗng, được gitignore. Database SQL thử GUID và database UI smoke đã được dọn. Dịch vụ AI dùng để test được dừng sau kiểm tra; dùng `scripts/Start-AI.ps1` để khởi động dịch vụ sản phẩm.

README có hướng dẫn chạy, tạo schema thủ công, quản lý secret, kịch bản UI và công việc cần bổ sung khi có phần cứng.
