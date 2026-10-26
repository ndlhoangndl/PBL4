# PBL4 — Bộ AI nhận dữ liệu từ thiết bị ngoài

Bộ API Python tích hợp YuNet, SFace và Whisper base từ thư mục model chung `../models`; giấy phép nằm ở `../licenses`. Đây là bộ xử lý AI của SmartDoor: không thu webcam/micro laptop, không có lệnh điều khiển khóa. Chế độ chạy độc lập có thể tải bản model riêng vào `runtime/models`.

## 1. Chạy trên Windows

1. Cài Python 3.11 hoặc 3.12 64-bit.
2. Với ứng dụng SmartDoor, chạy `SmartDoor/scripts/Setup-AI.ps1` một lần để tạo venv và cài thư viện.
3. Chạy `SmartDoor/scripts/Start-AI.ps1` để nạp model từ `SmartDoor/ai/models` và mở API ở `http://127.0.0.1:8000`.
4. Giữ cửa sổ PowerShell đang chạy; nhấn Ctrl+C để dừng dịch vụ.

Các file `setup.cmd` và `start.cmd` ở đây dành cho chế độ chạy runtime độc lập. Chế độ đó tạo venv riêng và tải model vào thư mục runtime nếu thiếu; không dùng các lệnh này khi chạy ứng dụng SmartDoor.

Nếu Python Launcher không tìm thấy 3.11, cài đúng Python 3.11 64-bit trước. Nếu cổng 8000 đang dùng, dừng tiến trình cũ hoặc đổi cổng trong start.cmd và client. Python 3.12 trên Linux đã được kiểm thử, còn file cài Windows chưa được chạy trên máy người dùng.

Sau khi cài đủ thư viện và tải/kiểm tra model qua `setup.cmd`, suy luận chạy cục bộ. Không có lời gọi dịch vụ AI đám mây. Mặc định chỉ lắng nghe localhost; chưa có đăng nhập quản trị hoặc bảo vệ truy cập qua LAN. Không đổi sang 0.0.0.0 trước khi bổ sung xác thực và giới hạn upload ở tầng HTTP.

## 2. Các model đã thu thập

| Model | Vai trò | File |
|---|---|---|
| YuNet 2023mar | Phát hiện mặt và 5 điểm mốc | `../models/face_detection_yunet_2023mar.onnx` |
| SFace 2021dec | Trích đặc trưng và so khớp danh tính | `../models/face_recognition_sface_2021dec.onnx` |
| Whisper base multilingual | Phiên âm tiếng Việt | `../models/whisper-base/` |
| Silero VAD | Phát hiện đoạn tiếng nói | Tài nguyên đi kèm thư viện faster-whisper sau khi cài |

Giữ nguyên trọng số gốc; không tự huấn luyện hay sửa model. Phần xây lại cho dự án gồm dịch vụ, giao diện, đăng ký nhiều mẫu, lưu dữ liệu, chuẩn hóa OTP và API C#.

Nguồn gốc, revision, kích thước và SHA256 của bộ model tích hợp nằm trong `../models-manifest.json`; manifest riêng cạnh runtime chỉ phục vụ chế độ độc lập. Không xóa thông tin giấy phép khi phân phối lại. YuNet dùng MIT, SFace kèm Apache 2.0, Whisper/faster-whisper dùng MIT theo các tài liệu nguồn đã thu.

Chạy kiểm tra toàn bộ file:

```powershell
.\.venv\Scripts\python.exe verify_models.py
```

Nếu một file bị mất hoặc checksum sai, chạy download_models.py. Khi manifest đã có, chương trình dùng URL/revision ghi trong manifest, tải lại và đối chiếu checksum. Bộ tải chưa tự retry lỗi mạng; chạy lại nếu mạng bị gián đoạn.

## 3. Đăng ký 4 người

Trên giao diện, chọn vị trí 1–4, nhập tên và chọn 3–30 ảnh của cùng người đó. Khuyến nghị bắt đầu với 10–20 ảnh khác nhau, từ camera ngoài. Bao gồm chính diện, hơi quay trái/phải, hơi cúi/ngẩng và điều kiện sáng thực tế. Mỗi ảnh đúng một mặt rõ nét. Không dùng nhiều frame giống hệt nhau làm bằng chứng đa dạng dữ liệu.

Mỗi ảnh được: phát hiện mặt → căn chỉnh → tạo vector SFace → lưu một mẫu riêng. Bản này không gộp toàn bộ ảnh thành một vector trung bình như bản trước. Khi nhận diện, lấy điểm tốt nhất trong các mẫu của mỗi người; sau đó so người đứng đầu với ngưỡng và chênh lệch so người thứ hai. Có nhiều mẫu giúp bao phủ góc nhìn nhưng không bảo đảm nhận mọi góc; cần hiệu chỉnh vì số lượng mẫu có thể làm tăng nguy cơ nhận nhầm.

Người quản trị phải xác nhận mọi ảnh trong một bộ cùng danh tính. Phần mềm chưa tự kiểm chứng điều đó. Không tạo sẵn người giả hay dùng khuôn mặt công khai để đăng ký trong sản phẩm.

Muốn thay mẫu cũ, bật lựa chọn thay bộ mẫu. Nếu ảnh mới có lỗi, dữ liệu đăng ký cũ được giữ nguyên. Xóa người trong danh sách sẽ xóa các vector liên quan.

Database phát sinh tại data/faces-v2.sqlite3. Nó chứa tên và vector; ảnh upload và âm thanh chỉ lưu tạm trong lúc xử lý rồi xóa. File nguồn trên thiết bị/máy người dùng vẫn còn ở vị trí cũ. Database bản cũ faces.sqlite3 có schema khác; bản này dùng database mới, cần đăng ký lại. Sao lưu database khi dịch vụ đã dừng. Giữ dữ liệu sinh trắc học trong phạm vi nhóm và có sự đồng ý của người cung cấp mẫu.

## 4. Thử nhận diện

Chọn ảnh mới không nằm trong bộ đăng ký. Hai giá trị mặc định trên giao diện:

- threshold=0.363: ngưỡng tham khảo trong ví dụ OpenCV.
- margin=0.05: chênh lệch thử nghiệm giữa điểm người thứ nhất và thứ hai.

Cả hai chưa hiệu chỉnh cho camera của nhóm. Nếu không đủ ngưỡng hoặc quá gần người thứ hai, trả unknown. Điểm cosine không phải xác suất phần trăm. Không dùng chính ảnh đăng ký để báo cáo độ chính xác.

Kết quả candidate chỉ có nghĩa là ứng viên khớp sinh trắc học. liveness luôn not_implemented và unlock_allowed luôn false. Chưa được nối kết quả này trực tiếp với relay.

## 5. Thử âm thanh

Upload bản ghi của module ngoài. Khuyến nghị WAV PCM mono 16 kHz, 16 bit, dài 0.3–20 giây. Module phải xuất được âm thanh thật; tín hiệu có/không tiếng động hoặc mã lệnh nhận sẵn không thay thế cho WAV.

Đọc từng chữ số, ví dụ “không tám ba bảy hai năm”. Bộ chuẩn hóa chỉ chấp nhận đúng sáu chữ số, giữ số 0 đầu. Có chữ lạ, thiếu/thừa số hoặc im lặng sẽ yêu cầu thử lại. Không dùng mã bí mật làm prompt và không sửa mã gần đúng theo database.

Whisper phiên âm nội dung, không xác minh ai đang nói. VAD không chống phát lại bản ghi. Giao diện hiển thị transcript/digits để debug cục bộ; không đưa các trường này vào log OTP thực tế. Chưa có backend xác thực OTP hết hạn/dùng một lần: otp_verified luôn false.

## 6. API để tích hợp C#

Tài liệu tương tác: http://127.0.0.1:8000/docs. Trường file dùng multipart/form-data.

| Method/path | Đầu vào |
|---|---|
| GET /health | Không cần dữ liệu |
| GET /users | Không cần dữ liệu |
| POST /face/enroll | slot, name, replace, nhiều trường images, session_id |
| POST /face/recognize | image, threshold, margin, session_id |
| POST /voice/transcribe | audio, session_id |
| DELETE /users/{slot} | Vị trí 1–4 |

Các kết quả AI có session_id, request_id, elapsed_ms. C# phải kiểm tra session_id thuộc phiên đang hoạt động và thời hạn kết quả; không coi một JSON trả về là quyền mở cửa.

Ví dụ client tại examples/CSharpClient.cs, chưa được biên dịch trong môi trường tạo bộ mã. Có thể thêm vào dự án WinForms/.NET. Dịch vụ giữ model trong RAM và tuần tự hóa các công việc AI để tránh dùng lẫn trạng thái model. Timeout HTTP không tự hủy một suy luận đã bắt đầu; backend phải bỏ kết quả đến muộn.

## 7. Chạy dòng lệnh nếu chưa cần giao diện

```powershell
.\.venv\Scripts\python.exe ai_files.py normalize --text "không tám ba bảy hai năm"
.\.venv\Scripts\python.exe ai_files.py list
.\.venv\Scripts\python.exe ai_files.py enroll --slot 1 --name "Hoang" --images samples/01.jpg samples/02.jpg samples/03.jpg
.\.venv\Scripts\python.exe ai_files.py recognize --image samples/query.jpg --threshold 0.363 --margin 0.05
.\.venv\Scripts\python.exe ai_files.py voice --audio samples/otp.wav
```

Các lệnh CLI tải model khi cần; dịch vụ là lựa chọn cho tích hợp chạy liên tục. Không chạy ghi database bằng CLI đồng thời với dịch vụ trong bản thử.

## 8. Phần tiếp theo để hoàn thiện dự án

1. Thu ảnh và âm thanh từ đúng camera/module, đăng ký lại 4 người.
2. Xây adapter nhận dữ liệu HTTP/UART/stream theo model thiết bị thật. Bản này chỉ nhận upload file.
3. Chỉnh threshold/margin bằng tập validation riêng; chốt rồi đánh giá trên tập test ở buổi khác, có người lạ.
4. Thêm chống giả mạo. Silent-Face là một ứng viên riêng; chưa được thu thập/tích hợp vào bộ này. Phải dùng crop/preprocessing đúng model và đánh giá ảnh in/màn hình qua camera thật.
5. Backend C# quản lý quyền, phiên, OTP dùng một lần, lịch sử và lệnh Arduino. Python không điều khiển COM.
6. Bổ sung quản trị truy cập và bảo vệ dữ liệu nếu mở dịch vụ ra ngoài máy.

## 9. Kiểm thử đã thực hiện

Chi tiết trong verification.json. Đã nạp model thật, chạy YuNet/SFace trên ảnh mẫu OpenCV, đăng ký nhiều vector và so khớp, kiểm tra rollback khi thay mẫu lỗi, từ chối ảnh không có mặt/ảnh hỏng, chạy VAD trên âm thanh im lặng, chạy Whisper với bản ghi tiếng Anh mẫu của OpenAI, và kiểm tra chuẩn hóa mã tiếng Việt.

Đây là kiểm tra tích hợp phần mềm. Chưa đo độ chính xác nhiều góc của 4 người, chất lượng đọc OTP tiếng Việt, độ trễ trên laptop của nhóm hoặc hoạt động với phần cứng. Không kèm ảnh/bản ghi mẫu công khai vào ZIP và không giữ database thử nghiệm. Windows setup/start và client C# chưa được chạy trên Windows trong môi trường này.
