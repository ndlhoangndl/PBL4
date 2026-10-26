# PBL4 — Hệ thống khóa cửa thông minh

Ứng dụng C# WinForms/.NET 8 quản lý tối đa 4 người, OTP và phiên xác thực, tích hợp API Python trong `ai/runtime`. **Mọi lệnh cửa chỉ chạy trên MockDoorController.** Luồng xác thực nhận file do người dùng chọn; chưa có COM/UART, relay hay firmware. Công cụ riêng `scripts/Collect-Faces.cmd` chụp webcam và `scripts/Collect-Voice.cmd` ghi micro để thu dữ liệu train; xem `ai/training/FACE_CAPTURE.md` và `ai/training/VOICE_CAPTURE.md`. Các công cụ thu dữ liệu không nối trực tiếp vào xác thực/mở cửa.

## Chạy nhanh trên máy hiện tại

Trong PowerShell, tại thư mục `SmartDoor`:

```powershell
.\scripts\Start-Desktop.ps1
```

Script tìm `dotnet` trong PATH hoặc `%USERPROFILE%\.dotnet\dotnet.exe`, build solution rồi chạy DLL bằng runtime đó. Máy hiện tại có SDK **8.0.419** và WindowsDesktop runtime **8.0.25** tại vị trí thứ hai. Chạy trực tiếp `.exe` khi runtime chưa được đăng ký có thể hiện thông báo thiếu .NET; dùng script trên.

Nếu execution policy chặn script, mở PowerShell riêng để chạy lệnh:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Start-Desktop.ps1
```

Mặc định **Mock AI**; dữ liệu nghiệp vụ vẫn dùng **SQL Server thật**, không có database giả dự phòng. Cấu hình mẫu dùng `Server=.\SQLEXPRESS;Database=Pbl4SmartDoor;Integrated Security=true;TrustServerCertificate=true;Connect Timeout=3`. Đã xác nhận SQL Server Express 2022 instance `SQLEXPRESS` trên máy này. Nếu máy khác có LocalDB, đổi Server thành `(localdb)\MSSQLLocalDB`; nếu dùng instance khác, nhập đúng tên instance đó.

Lần chạy đầu, EF Core tạo database/schema rỗng nếu tài khoản Windows có quyền. Không tạo người dùng mặc định. Khi database lỗi, giao diện hiển thị hướng dẫn và chặn nghiệp vụ; sửa connection string trong tab **Cấu hình và kiểm thử**, lưu và khởi động lại. Có thể đặt biến môi trường `PBL4_SQL_CONNECTION` để ghi đè connection string. Biến này được ưu tiên hơn cấu hình đã lưu.

## Luồng thử OTP đầy đủ khi chưa có thiết bị

1. Tab **OTP**: chọn thời hạn, bấm **Sinh OTP**, sao chép mã mới. Mã giữ số 0 đầu, chỉ hiển thị trong 60 giây hoặc đến khi rời tab. Lịch sử không có mã rõ.
2. Tab **Tổng quan**: **Bắt đầu phiên**, **Chuyển sang OTP**.
3. Tab **Cấu hình và kiểm thử**: nhập mã vào **ĐẦU VÀO OTP KIỂM THỬ**, bấm **Gửi OTP kiểm thử**.
4. Kết quả `Granted`, lệnh `Acknowledged`, cửa mô phỏng mở. Tab **Lịch sử**: lọc phương thức `TestOtp`, xem session ID, nguồn đầu vào kiểm thử và trạng thái lệnh.
5. Bắt đầu phiên mới, thử lại cùng mã: `Consumed`, không tạo lượt mở mới.

Đây là kiểm thử nghiệp vụ OTP bằng chuỗi nhập tay, **không phải nhận diện giọng nói**. OTP là quyền truy cập dạng mã cấp cho người giữ mã; không gắn với danh tính người nói hoặc bắt buộc liên kết một trong 4 người đăng ký.

Để thử đường đi qua `MockAiClient.TranscribeAsync`: nhập mã vào ô kiểm thử, bấm **Dùng mã làm kết quả Mock voice**, sau đó gửi một file âm thanh trong phiên `AwaitingOtp`. Mock không đọc nội dung file và chỉ trả kết quả đã đặt; giao diện/lịch sử ghi Mock AI.

## Quản lý 4 người và thử mặt Mock

1. Tab **Người dùng**: chọn vị trí 1–4, tên, quyền truy cập, bấm **Lưu tên / quyền**.
2. **Đăng ký 3–30 ảnh**: chọn ảnh do thiết bị ngoài cung cấp của đúng một người, xác nhận danh tính. Trong Mock, chỉ lưu metadata vị trí/tên/số mẫu trong bộ nhớ Mock; không tạo embedding hay khuôn mặt giả.
3. Chọn kịch bản **Người hợp lệ** trong tab kiểm thử. Vị trí ứng viên lấy từ vị trí đã chọn ở tab Người dùng.
4. **Bắt đầu phiên**, **Chọn ảnh…**. Mock trả danh tính hợp lệ + chống giả mạo đạt; backend kiểm tra quyền/đồng bộ/phiên bản trước khi gửi lệnh cửa mô phỏng.
5. Vô hiệu hóa người dùng rồi thử lại: `Denied`, không mở.

Mock metadata không được giữ sau khi đóng ứng dụng. Lần chạy lại, kiểm tra đồng bộ sẽ đánh dấu mẫu Mock mất đồng bộ; đăng ký lại metadata để thử mặt Mock. SQL vẫn giữ người dùng và lịch sử.

Đổi tên làm mất đồng bộ vì API Python không có endpoint đổi tên riêng; **Thay bộ mẫu…** để cập nhật tên ở Python. Thay mẫu và xóa đều có xác nhận. SQL đánh dấu chưa đồng bộ trước khi gọi AI; lỗi bất kỳ phía nào không được báo hoàn tất và không cho xác thực mặt. Muốn xóa mẫu phải chọn đúng chế độ đã đăng ký mẫu; xóa trong chế độ khác sẽ chặn quyền và yêu cầu chuyển về backend tương ứng.

## Chạy AI thật

Bộ mã API Python nằm ở `ai/runtime`; database mẫu đã đăng ký nằm tại `ai/runtime/data/faces-v2.sqlite3`. Các trọng số dùng lúc chạy và giấy phép tương ứng nằm trong `ai/models` và `ai/licenses`. Launcher dùng `ai/models-manifest.json` để kiểm tra SHA256 trước khi nạp. Nếu đặt runtime ở vị trí khác, truyền `-AiDirectory` cho các script dưới.

```powershell
.\scripts\Setup-AI.ps1
.\scripts\Start-AI.ps1
```

`Setup-AI.ps1` tạo `.venv-ai` riêng, cài `requirements.txt` của bộ AI. Đã chạy với Python **3.12.10 x64** trên máy hiện tại. `ai-requirements.lock.txt` ghi lại các phiên bản dependency thực tế đã cài để tham khảo/tái lập môi trường. Cài đặt lần đầu cần Internet.

`Start-AI.ps1` chỉ bind `127.0.0.1:8000`. Giữ terminal đó mở; Ctrl+C để dừng. Mở `http://127.0.0.1:8000/docs` nếu cần xem API.

OpenCV Windows có thể không đọc đường dẫn ONNX chứa dấu như `D:\Học tập`. Launcher `scripts/start_ai.py` kiểm tra SHA256 từng file trong `SmartDoor\ai`, sao chép nguyên byte sang `%TEMP%\Pbl4SmartDoor-model-cache` (đường dẫn ASCII), rồi truyền vị trí đó cho engine. Database Python nằm tại `ai/runtime/data/faces-v2.sqlite3`, không đổi schema khi sắp xếp thư mục. Các mô hình đang dùng là YuNet, SFace và Whisper base; chưa có mô hình chống giả mạo/liveness và chưa huấn luyện lại các mô hình này. Xem `ai/README.md` để biết cấu trúc, nguồn và bước tiếp theo cho phần train.

Trong desktop chọn **Real**, đặt URL và timeout, lưu, khởi động lại. Không tự chuyển sang Mock khi Python lỗi. **Kiểm tra dịch vụ** sẽ gọi `/health`, `/users` và kiểm tra liên kết metadata.

Các endpoint dùng đúng multipart field:

| Endpoint | Field |
|---|---|
| `POST /face/enroll` | `slot`, `name`, `replace`, nhiều `images`, `session_id` |
| `POST /face/recognize` | `image`, `threshold`, `margin`, `session_id` |
| `POST /voice/transcribe` | `audio`, `session_id` |
| `GET /health`, `GET /users` | Không có multipart |
| `DELETE /users/{slot}` | Vị trí 1–4 |

`session_id` được kiểm tra trước khi dùng kết quả. DTO giữ `request_id`, `elapsed_ms`; client không coi `unlock_allowed`, `otp_verified` hay transcript là quyền mở. Voice client chỉ giữ `digits` tạm để kiểm tra HMAC, không giữ transcript trong lịch sử.

**AI thật chỉ hiển thị ứng viên mặt và chuyển sang OTP**, kể cả nếu response bị thay đổi để nói chống giả mạo đạt. YuNet/SFace hiện chưa chống giả mạo. Ngưỡng `0.363` và margin `0.05` chỉ là thử nghiệm, chưa hiệu chỉnh trên camera thật; cosine không phải phần trăm xác suất.

Giới hạn file giống Python: ảnh 10 MiB, âm thanh 20 MiB; đăng ký 3–30 ảnh. WAV mono PCM 16 kHz, 0,3–20 giây là định dạng gợi ý từ bộ AI. Real AI hiện trả cùng lỗi “Exactly one face required” cho không mặt/nhiều mặt nên client hiển thị nhóm lỗi đó; kịch bản Mock phân biệt hai trường hợp. Không suy đoán một response có thông tin mà API chưa cung cấp.

## OTP, secret và dữ liệu

- Mã sinh bằng `RandomNumberGenerator`, định dạng `D6`; không lưu mã rõ.
- HMAC-SHA256, unique digest. Sinh lại khi trùng digest; không tái cấp mã còn nằm trong sổ OTP (kể cả đã dùng/hết hạn). Chưa tự xóa lịch sử OTP; nếu triển khai lâu dài cần chính sách lưu/thu hồi, vì không gian mã có 1 triệu giá trị.
- Khóa mặc định sinh riêng cho tài khoản Windows, bảo vệ bằng **DPAPI CurrentUser** ở `%LOCALAPPDATA%\Pbl4SmartDoor\otp-hmac.dpapi`. Khóa không nằm trong mã nguồn, cấu hình hoặc SQL. Sao lưu khóa phù hợp với tài khoản Windows trước khi chuyển máy; mất/đổi khóa làm các OTP cũ không kiểm tra được.
- Có thể dùng biến `PBL4_OTP_HMAC_KEY` (Base64, ít nhất 32 byte) thay DPAPI. Cấp qua môi trường/secret manager của người vận hành, không commit hoặc in khóa ra log. Không đổi khóa khi còn OTP đang hoạt động.
- Mỗi phiên tối đa 3 lần nhập/đọc sai; thiếu số tính là một lần gửi OTP. Mỗi lần chọn ảnh là một yêu cầu, không đếm frame nội bộ.
- SQL transaction `Serializable` + `UPDLOCK,HOLDLOCK` khóa phiên rồi bản ghi OTP; kiểm tra và consume trong một transaction. Kiểm tra lại expiry sau khi đợi khóa. Hai yêu cầu cùng mã chỉ một yêu cầu được chấp nhận.
- Sau consume, mất kết nối/không phản hồi/hủy phiên **không khôi phục OTP**.
- Thời gian lưu UTC, hiển thị giờ địa phương.
- `.gitignore` loại ảnh, âm thanh, database, secret, venv và kết quả thử. C# không chứa kho embedding; Python quản lý mẫu.

Cấu hình người dùng lưu ở `%LOCALAPPDATA%\Pbl4SmartDoor\appsettings.local.json`, được ưu tiên so với cấu hình mẫu cạnh DLL. Nên dùng Integrated Security; nếu dùng password trong connection string, người vận hành phải bảo vệ file cấu hình hoặc cấp connection string qua môi trường.

## Phiên và cửa mô phỏng

Luồng: `Idle → AwaitingFace → ProcessingFace → AwaitingOtp → ProcessingOtp → Granted/Denied`; hủy/hết hạn dẫn tới `Cancelled/TimedOut`. Mock mặt hợp lệ có thể đi trực tiếp từ `ProcessingFace` đến `Granted`.

Một mutex ngăn hai tiến trình desktop trong cùng phiên Windows; engine cho phép một phiên/yêu cầu tại một thời điểm. Token hủy và kiểm tra thời hạn/session ID bỏ kết quả cũ, kể cả AI không tuân thủ hủy. Thay người dùng/bộ mẫu trong lúc suy luận làm thay revision và vô hiệu kết quả cũ. Gate chung giữ ổn định quyền/mẫu từ kiểm tra cuối đến khi cửa mô phỏng xử lý lệnh; việc hủy và hành động mô phỏng được kiểm tra dưới cùng khóa.

Không thao tác trực tiếp SQLite Python/SQL hoặc dùng CLI đăng ký đồng thời với desktop. API `/users` chỉ có tên và số mẫu, **không có version/checksum mẫu**; thay vector ngoài ứng dụng mà giữ nguyên tên/số lượng không thể được phát hiện đầy đủ bởi C#. Cần bổ sung revision API trước khi hỗ trợ nhiều công cụ quản trị đồng thời. Bản hiện tại dành cho một desktop trên một máy.

Lệnh cửa có GUID `command_id`; gọi trùng ID trong cùng vòng đời simulator không tăng lượt mở. Không tự gửi lại lệnh sau khởi động lại. Phiên chưa kết thúc được đánh dấu Cancelled; lệnh Sent còn dang dở chuyển `UnknownAfterRestart`.

Giao diện tách **Sent**, **Acknowledged**, **NoResponse**, **Disconnected**, **CancelledOrTimedOut** khỏi trạng thái cửa mở/đóng. Vị trí chốt luôn “chưa xác định” vì chưa có cảm biến. Có các nút có/không người, đóng cửa, mất/kết nối lại, không phản hồi và EXIT. EXIT hủy phiên đang hoạt động rồi tạo lệnh riêng, không gọi AI; vẫn cần SQL để lưu audit trong bản mô phỏng này.

## Database và cấu trúc

```text
SmartDoor.sln
src/SmartDoor.Core/            Entities, DTO, interface, OTP, phiên, quản lý người dùng
src/SmartDoor.Infrastructure/  EF Core SQL Server, HttpClient AI, Mock AI, Mock door
src/SmartDoor.Desktop/         WinForms + AppController (không query SQL trong button handler)
tests/SmartDoor.Tests/         Nghiệp vụ, hợp đồng HTTP, SQL/live AI integration
database/                     Script schema chính xác từ EF model
scripts/                      Khởi động, setup, kiểm thử
```

Schema có `Users`, `OtpCodes`, `AuthenticationSessions`, `AccessEvents`, `DoorCommands`, khóa ngoại audit và check slot 1–4. Không seed dữ liệu cá nhân. Bản đầu dùng `EnsureCreatedAsync`, **không dùng EF migrations trộn với EnsureCreated**. Nếu nâng schema về sau, cần thiết kế migration riêng.

Tạo thủ công nếu ứng dụng không có quyền tạo database, chạy từ thư mục `database` bằng SSMS SQLCMD mode hoặc:

```powershell
sqlcmd -S . -E -C -i Create-Database.sql
```

Script dành cho database mới/rỗng, không chạy lặp lại trên schema đã có. `schema.sql` được xuất từ model EF đã chạy integration test, không gồm secret/dữ liệu.

## Build và kiểm thử

```powershell
.\scripts\Test.ps1
.\scripts\Test.ps1 -SqlIntegration
```

Không đặt `PBL4_TEST_SQL_BASE` thì SQL integration được **Skipped**, không tính là đã thử SQL. Mặc định script dùng `Server=.\SQLEXPRESS`. Dùng SQL khác: đặt biến này tới server có quyền tạo/xóa database kiểm thử. Mỗi test tạo database tên ngẫu nhiên `Pbl4SmartDoor_Test_<guid>` và chỉ xóa đúng database vừa tạo, không đụng database ứng dụng.

Live AI test cũng chỉ chạy khi được bật. Dùng server với database thử riêng, tuyệt đối không chỉ tới database đã đăng ký người:

```powershell
# Terminal 1
.\.venv-ai\Scripts\python.exe .\scripts\start_ai.py --test-db .\TestResults\ai-smoke.sqlite3
# Terminal 2
.\.venv-ai\Scripts\python.exe .\scripts\create_smoke_inputs.py
$env:PBL4_TEST_AI_URL='http://127.0.0.1:8000'
$env:PBL4_TEST_BLANK_IMAGE=(Resolve-Path .\TestResults\blank.png).Path
$env:PBL4_TEST_SILENT_AUDIO=(Resolve-Path .\TestResults\silence.wav).Path
.\scripts\Test.ps1 -SqlIntegration
```

Ảnh trống và âm thanh im lặng là đầu vào tổng hợp để thử **từ chối**, không phải dữ liệu khuôn mặt/giọng nói của người thật. Kết quả TRX nằm trong `TestResults`. Xem `VERIFICATION.md` để biết chính xác phần đã/chưa chạy.

## Kịch bản kiểm thử thủ công

| Tình huống | Thao tác | Kết quả mong đợi |
|---|---|---|
| Người hợp lệ | Đăng ký metadata Mock, bật quyền, chọn Người hợp lệ, gửi ảnh | Granted; cửa mô phỏng Acknowledged |
| Người lạ | Chọn Người lạ, bắt đầu phiên, gửi ảnh | Chuyển AwaitingOtp; chưa mở |
| Không mặt / nhiều mặt / ảnh lỗi | Chọn từng kịch bản Mock và gửi ảnh | Lý do lỗi tương ứng, không mở |
| Người bị vô hiệu hóa | Tắt quyền rồi thử Mock mặt hợp lệ | Denied |
| OTP thiếu số | Chọn OTP thiếu số, gửi file audio trong AwaitingOtp | Một lần sai, không mở |
| OTP đúng/sai/hết hạn/đã dùng/thu hồi | Sinh mã, thử theo trạng thái; thu hồi trong tab OTP | Chỉ Active+đúng được consume/mở |
| AI không sẵn sàng | Chọn kịch bản Mock hoặc dừng Python khi ở Real | Báo AI lỗi, không đổi chế độ |
| DB không sẵn sàng | Bấm DB không sẵn sàng hoặc cấu hình instance sai | Báo lỗi, chặn nghiệp vụ, không lưu giả |
| Cửa mất kết nối | Bấm Cửa mất kết nối rồi gửi OTP hợp lệ | OTP Consumed; lệnh Disconnected, cửa không mở |
| Lệnh không phản hồi | Bật Lệnh không phản hồi rồi gửi OTP đúng | NoResponse sau khoảng 3 giây, không mở muộn |
| EXIT khi AI lỗi | Bật AI không sẵn sàng rồi bấm Nút EXIT | Có lệnh mô phỏng riêng, không phụ thuộc AI |
| Hủy/AI đến muộn | Chọn AI đến muộn, gửi ảnh rồi hủy | Cancelled, không mở muộn |
| Thay mẫu trong lúc xử lý | AI đến muộn, đồng thời thay mẫu | Kết quả cũ bị từ chối |
| Đăng ký thất bại | Chọn Đăng ký lỗi rồi đăng ký/thay mẫu | Synchronized=false, không báo hoàn tất |

Các kịch bản trên được cung cấp cho người vận hành kiểm thử UI. Không mặc định coi mọi dòng đã được thực hiện bằng giao diện.

## Khi có phần cứng

Thêm adapter nhận file từ camera/module, implementation Serial/UART của `IDoorController` sau khi biết board/giao thức, command ACK và cơ chế dedup bền vững trên thiết bị. Thu dữ liệu có đồng ý, hiệu chỉnh threshold/margin bằng tập riêng, đánh giá người lạ/ảnh in/màn hình, bổ sung chống giả mạo trước khi cấp quyền bằng mặt thật. Đánh giá an toàn EXIT và cảm biến chốt theo sơ đồ thực tế. Chưa có firmware, quản trị nhiều máy hoặc đăng nhập quản trị trong bản này.

Không gọi kết quả Mock là độ chính xác model hoặc kiểm thử phần cứng. Không dùng nhận diện ứng viên mặt thật để điều khiển khóa.
