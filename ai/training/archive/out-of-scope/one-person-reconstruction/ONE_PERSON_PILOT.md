# Nhánh thử một người: CNN tái tạo vùng mặt

`pbl4_training/train_one_person.py` định nghĩa `DoorFaceAutoencoder` và vòng lặp train từ trọng số ngẫu nhiên. Đây là nghiên cứu tái tạo ảnh của một người qua bottleneck 32 chiều, **không phải DoorFaceEmbedder/ArcFace đã train**, và chưa phải bộ xác minh danh tính.

Ảnh phải có vùng da mặt và mắt đã duyệt, hash khớp nguồn. Tiền xử lý dùng `face_region.py` để che ngoài viền/căn mắt; loss chỉ tính trên mask da mặt. CNN dùng ảnh 64×64 RGB [0,1], ba lớp convolution encoder, bottleneck, ba lớp transposed-convolution decoder. Không có trọng số pretrained.

Trainer yêu cầu train và validation khác buổi, khác source và không trùng byte ảnh; chỉ có một identity. Checkpoint chọn bằng validation masked MSE. Không có accuracy nhận diện, FAR người lạ hay threshold cho phép mở cửa khi dữ liệu chỉ có một người. Lỗi tái tạo thấp cũng không đủ chứng minh một ảnh là đúng người.

Lần chạy thử thực tế ngày 19/10/2026 dùng 26 ảnh buổi mới để train và 9 ảnh buổi cũ để validation, chọn bảo thủ từ 60 ảnh đã duyệt. Lý do loại các ảnh còn tóc/mép nền ghi trong `data/one-person-pilot-20261019/selection.json`; giữ nguyên toàn bộ ảnh/annotation gốc. Việc chọn trực quan không thay thế phân đoạn tóc tự động, vẫn có thể còn sai số ở viền. Validation chỉ có ít ảnh và chủ yếu góc cúi/ngẩng; không đại diện đầy đủ mọi góc hay điều kiện.

```powershell
.\.venv-training\Scripts\python.exe -B -m pbl4_training.train_one_person --train-manifest .\data\one-person-pilot-20261019\train.jsonl --validation-manifest .\data\one-person-pilot-20261019\validation.jsonl --run-dir .\runs\person001-face-pilot-20261019-v1 --epochs 30
```

Lệnh yêu cầu thư mục output mới, không chạy đè bằng chứng. Output gồm `best.pt`, `last.pt`, loss theo epoch, manifest snapshot, hash dữ liệu/mã và `summary.json`. Báo cáo còn so với CNN ngẫu nhiên và baseline ảnh trung bình từ train. Kết quả có thể không tốt hơn baseline đơn giản; phải đọc đúng số liệu. Không có test độc lập và không tích hợp checkpoint vào khóa cửa. Phiên sau cần cải thiện dữ liệu, dùng buổi test riêng và có mẫu người lạ để đo từ chối trước khi gọi đây là nhận diện một người đã được xác minh.

## Kết quả lần chạy thực tế

`runs/person001-face-pilot-20261019-v1/`: đã hoàn tất 30 epoch trên CPU, model 331.635 tham số, 16 tensor trọng số thay đổi. Chọn checkpoint epoch 2. Validation masked MSE: lúc khởi tạo 0.0152071, checkpoint tốt nhất 0.0145121; baseline ảnh trung bình train là 0.0207910. Đây là lỗi tái tạo RGB chuẩn hóa [0,1], không phải tỷ lệ nhận diện đúng. Train loss tiếp tục giảm nhưng validation tăng sau các epoch đầu, cho thấy rủi ro overfit/chênh điều kiện thu. Chưa có threshold và chưa đo chấp nhận người lạ.
