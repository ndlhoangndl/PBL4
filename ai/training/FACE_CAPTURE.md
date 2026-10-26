# Chụp bộ ảnh đầu tiên bằng webcam hoặc ESP32-CAM

## Mở công cụ

Nhấp đúp `SmartDoor/scripts/Collect-Faces.cmd`, hoặc từ PowerShell tại `SmartDoor`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Collect-Faces.ps1
```

Dùng `.venv-ai` đã có OpenCV và Tkinter của Python Windows; không cần cài Pillow hay thay OpenCV headless bằng bản GUI. Công cụ này riêng cho thu dữ liệu, không gọi API đăng ký người hay gửi lệnh cửa.

## Chụp 50 ảnh mỗi người

1. Mỗi người dùng mã riêng: `person-001` cho bạn, `person-002`, `person-003`, `person-004` cho ba bạn còn lại. Chọn **USB** và camera 0 (thử 1 nếu cần), hoặc **ESP32** (đang được chọn mặc định).
2. Với ESP32-CAM, bật board và kết nối máy tính cùng mạng Wi-Fi. Mở trang camera để chắc chắn video đang chạy. Trong ô URL của công cụ thu ảnh, nhập URL luồng MJPEG, thường là `http://10.120.253.192:81/stream` theo IP gần nhất in trong Serial Monitor. Nếu IP của board đổi, thay địa chỉ IP trong URL. URL này lấy khung hình qua mạng; ảnh vẫn được lưu trên máy tính.
3. Vén tóc mái để lộ trán, rồi hạ tay khỏi khung hình. Ngồi cách camera khoảng 0,5–1 m, ánh sáng từ phía trước, mặt rõ và chỉ một người trong khung hình. Với USB, đóng ứng dụng khác đang giữ webcam.
4. Bấm **Bật camera**. Chỉ thao tác này mới kết nối tới nguồn ảnh; mở cửa sổ công cụ không tự bật camera.
5. Chụp theo hướng dẫn hiện trong cửa sổ: **20 chính diện, 10 hơi quay trái, 10 hơi quay phải, 5 hơi cúi, 5 hơi ngẩng**. Mỗi lần đổi nhẹ khoảng cách/biểu cảm/tư thế; không lưu nhiều bản giống nhau.
6. Bấm **Chụp ảnh**, xem ảnh đứng yên. Nếu mờ/che mặt/nhiều người, chọn **Chụp lại**. Nếu ảnh đạt, tích xác nhận rồi bấm **Lưu ảnh này**.
7. Sau 50 ảnh, bấm **Dừng camera**. **Mở thư mục ảnh** để xem bộ vừa lưu. Đóng cửa sổ cũng giải phóng kết nối.

Nếu ESP32 không kết nối được, mở thử chính URL stream trong trình duyệt. CameraWebServer chuẩn thường dùng cổng `81` và đường dẫn `/stream`; nếu firmware đang chạy dùng đường dẫn khác, nhập đúng URL MJPEG do firmware cung cấp. Trang điều khiển ở địa chỉ gốc `/` chỉ là giao diện web, không phải lúc nào cũng là URL luồng video.

## Giảm giật/độ trễ khi dùng ESP32

Trong trang điều khiển camera, thử **Resolution = SVGA (800×600)** và **Quality = 12**. Nếu vẫn trễ, dùng VGA (640×480) hoặc tăng Quality lên 15; số Quality cao hơn nén ảnh mạnh hơn nên truyền nhẹ hơn. Trong công cụ thu ảnh, luồng được đọc nền liên tục và chỉ khung mới nhất được đưa lên preview để tránh xem ảnh cũ bị dồn hàng. Ảnh lưu vẫn giữ nguyên độ phân giải do ESP32 gửi.

Các cảnh báo sáng/tối/nhòe chỉ là gợi ý thống kê ảnh; công cụ không tự xác minh có mặt, số mặt hay đúng tư thế. Ảnh xem trước và ảnh lưu đều không lật gương. Chỉ preview được thu nhỏ; ảnh PNG giữ độ phân giải camera thực tế. Không có chức năng thu âm hoặc tự chụp nền.

## File được tạo

```text
SmartDoor/ai/datasets/face-capture/
  person-001/
    capture-<thời gian UTC>-<mã ngẫu nhiên>/
      images/             Ảnh PNG đã được bạn chấp nhận
      captures.jsonl      Đường dẫn tương đối, người, buổi, tư thế hướng dẫn, hash
      session.json        Mô tả bộ dữ liệu và kế hoạch 50 ảnh
```

Mỗi buổi thật dùng **Buổi mới** sau khi dừng camera; thư mục mới không ghi đè ảnh cũ. Nếu chỉ tạm dừng và tiếp tục ngay, giữ nguyên cửa sổ rồi bấm Bật camera để tiếp tục cùng buổi. Sau khi đóng cửa sổ, lần mở tiếp theo sẽ tạo buổi mới; không đổi tên buổi để biến ảnh cũ thành buổi độc lập. Nếu cần loại ảnh đã lưu, đánh dấu loại ở bước rà soát/gán nhãn; giữ sample/source ID để truy vết.

Ảnh và metadata ở đây nằm trong thư mục dataset được Git bỏ qua. Công cụ đọc video ESP32 qua mạng nội bộ nhưng chỉ lưu ảnh và metadata trên máy; không tải dữ liệu lên dịch vụ ngoài. Metadata ghi nguồn camera, không ghi mật khẩu Wi-Fi. `device_id` cho biết USB index hoặc địa chỉ ESP32 để truy vết nguồn ảnh.

**ESP32 chỉ chụp/truyền ảnh, không huấn luyện model.** Huấn luyện chạy trên máy tính bằng PyTorch sau khi ảnh được duyệt và gán nhãn. Code model tự tạo nằm trong `ai/training/pbl4_training/door_models.py`; vòng lặp huấn luyện ở `ai/training/pbl4_training/train_door_vision.py`. Bộ 50 ảnh là dữ liệu thu thô của một buổi; chưa đủ nhãn và phiên độc lập để đánh giá khả năng tổng quát.

## Bộ đầu tiên dùng để làm gì?

Mỗi bộ 50 ảnh là **dữ liệu thô của một buổi thu**, có thể bắt đầu rà soát và gán nhãn để chuẩn bị train. `captures.jsonl` là manifest thu thập, **chưa phải manifest huấn luyện**: chưa có box, landmark, crop/căn chỉnh hoặc nhãn chống giả mạo được xác minh.

- Phát hiện mặt: đánh dấu box và 5 landmark trên ảnh gốc; thêm ảnh nền không có mặt. Sau đó tạo manifest theo `manifests/face_detection.jsonl.example`.
- Nhận diện: tạo ảnh crop/căn chỉnh nhất quán, gán `identity_id`; trainer hiện cần tối thiểu hai danh tính và các buổi độc lập. Một người trong một buổi chưa thể train/đánh giá nhận diện theo pipeline này.
- Chống giả mạo: ảnh webcam là dữ liệu mặt thật; còn cần các clip ảnh in/màn hình, đúng metadata và cách crop nhất quán. Không tự gán tất cả 50 ảnh là một dataset chống giả mạo hoàn chỉnh.

Không chia ngẫu nhiên 50 ảnh trong cùng một buổi thành train/validation/test. Thu ảnh của bốn người với mã riêng; để đánh giá train/validation/test cần quay lại chụp các buổi độc lập khác. Xem [DOOR_VISION.md](DOOR_VISION.md). Bộ ảnh này phục vụ thử nghiệm, không cam kết model tổng quát hoặc an toàn cho khóa thật.

## Chỉ lấy da mặt từ trán đến cằm, không lấy tóc/nền

Nhấp đúp `SmartDoor/scripts/Review-Face-Regions.cmd`. Công cụ kiểm tra buổi chụp mới nhất, sau đó mở trình chỉnh viền da mặt. Muốn chọn buổi cụ thể: chạy `Review-Face-Regions.ps1 -SessionDirectory <đường dẫn buổi>`.

1. Kéo các điểm đỏ theo đường chân tóc/da trán, hai má và cằm. Bỏ tóc mái, tai, cổ, áo và nền khỏi viền. Mắt/mũi/miệng vẫn phải nằm trong viền.
2. Nếu viền ban đầu không hợp, bấm **Vẽ lại viền**, bấm lần lượt quanh vùng da mặt theo một chiều. Bấm **Đặt mắt bên trái ảnh**, chọn tâm mắt; làm tương tự cho mắt còn lại. Sau đó có thể kéo tiếp điểm viền; chuột phải bỏ một điểm.
3. Xem hình bên phải: đó là vùng đã che ngoài viền, căn theo mắt và chuẩn hóa mà model tự train sẽ nhận. Nếu còn tóc thì chỉnh viền; không thể khôi phục phần trán đang bị tóc che, cần chụp lại để có phần da trán đó.
4. Tích xác nhận rồi **Lưu vùng mặt và tiếp**. Ảnh không phù hợp thì ghi lý do và **Bỏ ảnh**; ảnh gốc vẫn được giữ.

Mỗi buổi có thêm `face-review.json` (kiểm tra và gợi ý), `face-regions/annotations.json` (viền đã duyệt), `face-regions/previews/` và `face-regions/recognition.jsonl`. Manifest recognition chỉ chứa ảnh đã duyệt, dùng `identity_id` là mã người; các trường pool/split cần được rà soát theo kế hoạch train khi tổng hợp nhiều người/buổi.

Gợi ý vị trí mặt/mắt ban đầu đến từ YuNet có sẵn, không phải nhãn đúng sẵn hay model tự train đã hoàn thành. Tóc/da được phân biệt bằng việc bạn chỉnh và duyệt viền, không phải thuật toán tự phân đoạn tóc. Mã `face_region.py` thay pixel ngoài viền bằng màu xám trước khi nội suy/căn mắt để ảnh nền hoặc tóc ngoài viền không đi vào tensor nhận diện/chống giả mạo. Trainer hai tác vụ này từ chối ảnh thiếu `face_region` đã duyệt hoặc annotation không khớp hash ảnh gốc.

Detector vẫn cần ảnh gốc để học tìm vị trí mặt. Runtime SFace cũ chưa được đổi cách tiền xử lý; khi tích hợp `DoorFaceEmbedder`/`DoorAntiSpoof` phải dùng cùng chính sách ROI và đánh giá lại. Không tự áp mask mới lên SFace cũ hoặc mẫu đăng ký cũ.

Kiểm thử phần mềm ngày 18/10/2026: 23 bài kiểm thử đạt, gồm xác nhận thay toàn bộ pixel ngoài viền không thay đổi đầu vào model (kể cả khi ảnh bị xoay), kiểm tra hash annotation và xuất manifest chỉ từ ảnh đã duyệt. Giao diện chỉnh viền đã được kiểm tra dựng cửa sổ/preview trên bộ ảnh hiện có; chưa thay việc bạn duyệt đường chân tóc của từng ảnh.

## Kế hoạch train bốn người

Mỗi người dùng mã cố định riêng (`person-001` đến `person-004`). Buổi thu hiện tại đặt 50 ảnh/người; kế hoạch đánh giá đầy đủ vẫn cần tối thiểu ba buổi độc lập/người, duyệt và gán nhãn từng buổi. Sau đó ghép dữ liệu và train một model nhận diện chung có bốn nhãn danh tính; không khởi tạo model mới cho từng người. Detector cũng học chung trên ảnh của cả bốn cùng ảnh nền. Giữ train/validation/test tách theo phiên; một buổi/người chỉ đủ làm demo nhỏ. Chống giả mạo cần thêm video mặt thật và video phát lại từ ảnh in/màn hình, lưu nhãn attack type và thiết bị.
