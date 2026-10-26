# Ba mô hình thị giác riêng của SmartDoor

## Phạm vi chính xác

`pbl4_training/door_models.py` chứa mã kiến trúc CNN và loss. `pbl4_training/train_door_vision.py` chứa đọc dữ liệu, forward, backward, AdamW, chọn checkpoint và đánh giá. Mặc định khởi tạo ngẫu nhiên, không tải trọng số của YuNet, SFace hay model chống giả mạo bên thứ ba.

Đây là ba baseline nghiên cứu nhỏ do dự án định nghĩa, chưa được train bằng dữ liệu mặt thật. Hiện có 60 ảnh của người thứ nhất qua hai buổi, với vùng da mặt đã duyệt. Cần thu thêm ba người; box và năm landmark của detector cũng cần gán nhãn riêng. Chúng **không phải phiên bản YuNet/SFace được fine-tune** và không tương thích trực tiếp với OpenCV `FaceDetectorYN`/`FaceRecognizerSF`. API hiện tại vẫn dùng model gốc; chưa có adapter hay thao tác thay model trong xác thực. Chưa thể báo cáo độ chính xác hoặc dùng baseline mới làm điều kiện mở cửa.

| Mô hình | Đầu vào và đầu ra | Loss |
|---|---|---|
| `DoorFaceDetector` | RGB 160×160; một điểm hiện diện, một box, năm điểm mốc | BCE hiện diện + 5×SmoothL1 box + 5×SmoothL1 landmark; mask nền/điểm không có nhãn |
| `DoorFaceEmbedder` | Da mặt đã duyệt viền, che ngoài viền/căn mắt 112×112; vector 128 chiều chuẩn hóa L2 | ArcFace head + cross entropy; head chỉ dùng khi train |
| `DoorAntiSpoof` | Vùng da mặt cùng chính sách 112×112; hai logits spoof/live | Cross entropy |

Backbone gồm bốn khối Conv–GroupNorm–SiLU. Đây là cách ghép các thành phần chuẩn bằng PyTorch, không tuyên bố phát minh CNN hay ArcFace. Công thức ArcFace được ghi nguồn theo [Deng và cộng sự](https://arxiv.org/abs/1801.07698). Detector giữ pooling không gian 4×4; embedder/classifier dùng pooling toàn cục. Tất cả input đổi RGB và chuẩn hóa `pixel/127.5 - 1`. Detector resize bilinear toàn ảnh; nhận diện và chống giả mạo bắt buộc `face_region` đã duyệt: che vùng ngoài bằng 127 trước khi rotate/nội suy, căn ngang mắt, crop và letterbox vuông. Xem [FACE_CAPTURE.md](FACE_CAPTURE.md) để gán viền da mặt và loại tóc.

## Dữ liệu cần cung cấp

Dùng schema và CLI hiện có trong `DATASET_SCHEMA.md` để validate rồi split; các file split phải là `train.jsonl`, `validation.jsonl`, `test.jsonl`. Trainer kiểm tra lại toàn bộ ảnh, trùng bytes, sample ID, session/video/source không được rò qua các tập.

- **Detector:** ảnh có đúng một mặt với box và năm landmark được kiểm tra thủ công; thêm ảnh không mặt với `no_face:true`. Mỗi split cần cả ảnh mặt và ảnh nền. Baseline chỉ dự đoán tối đa một mặt, không phát hiện/đếm nhiều mặt nên chưa đáp ứng cơ chế từ chối ảnh có nhiều mặt của ứng dụng.
- **Embedder:** ảnh gốc kèm `face_region` đã duyệt viền da mặt và hai mắt, hash khớp nguồn. Không đưa nguyên cảnh camera vào model. Mục tiêu dự án là bốn danh tính có nhiều phiên độc lập. Thu từng người riêng, nhưng gộp nhãn của cả bốn vào cùng manifest và train một embedder chung; train lại từng người từ trọng số ngẫu nhiên sẽ làm mất biểu diễn đã học trước. Người lạ chỉ đưa vào test. Bốn người là pilot hẹp, chưa chứng minh tổng quát cho người bất kỳ. `--aligned-faces` xác nhận người chuẩn bị đã kiểm tra viền/mắt; trainer tự áp mask và căn mắt theo annotation.
- **Anti-spoof:** ảnh mặt thật và ảnh mặt bị phát lại qua ảnh in/màn hình, có metadata attack type, video, buổi thu, camera và thiết bị trình chiếu, kèm `face_region` đã duyệt như nhận diện. Mỗi split cần cả `live` và `spoof`. Chính sách da mặt loại nền và viền màn hình; chưa có bằng chứng hiệu quả trên tấn công thực tế. Dùng danh tính/thiết bị giữ riêng khi đánh giá khả năng tổng quát.

## Cài môi trường

Từ `SmartDoor/ai/training`, dùng `.venv-training` riêng. PyTorch CPU được cài trong phiên phát triển để chạy kiểm thử; bản này không dùng RTX 4050. Trước khi train GPU, cài build CUDA phù hợp từ [bộ chọn PyTorch](https://pytorch.org/get-started/locally/) và kiểm tra `torch.cuda.is_available()`.

```powershell
py -3.12 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-training\Scripts\python.exe -m pip install -r requirements-door.txt
```

## Train từ đầu

Ví dụ sau giả định dữ liệu thật đã được kiểm tra/chia đúng. `data/splits/*` chưa có sẵn; không dùng file `.example` làm dữ liệu train.

```powershell
.\.venv-training\Scripts\python.exe -m pbl4_training.train_door_vision face_detection --splits .\data\splits\detection-v1 --run-dir .\runs\door-detector-v1 --epochs 20 --device cpu
.\.venv-training\Scripts\python.exe -m pbl4_training.train_door_vision face_recognition --splits .\data\splits\recognition-v1 --run-dir .\runs\door-embedder-v1 --epochs 20 --aligned-faces --device cpu
.\.venv-training\Scripts\python.exe -m pbl4_training.train_door_vision antispoof --splits .\data\splits\antispoof-v1 --run-dir .\runs\door-antispoof-v1 --epochs 20 --device cpu
```

Đổi `--device cuda` sau khi đã cài PyTorch CUDA. Batch mặc định 16, learning rate 0.001 là cấu hình ban đầu, chưa tối ưu trên dữ liệu dự án. Có thể thay qua CLI và chỉ chọn cấu hình bằng validation.

`--init-checkpoint <best.pt>` fine-tune model SmartDoor cùng task/kiến trúc/mapping danh tính, khởi tạo optimizer mới. Không nhận ONNX hay `model.bin` của Whisper. Đây không phải resume trạng thái optimizer; muốn tiếp tục một lần train bị ngắt cần bổ sung resume riêng. Mỗi lần chạy phải dùng `--run-dir` mới.

## Bằng chứng sau mỗi lần chạy

- `best.pt`: trọng số tốt nhất theo **validation loss**, kèm head, optimizer, epoch, cấu hình và mapping lớp.
- `last.pt`, `epochs.jsonl`: trạng thái cuối và loss từng epoch.
- `run.json`, ba manifest snapshot, `media-sha256.json`: nguồn mã, cấu hình, seed và hash dữ liệu.
- `initial-evaluation.json`, `trained-evaluation.json`: so sánh model lúc khởi tạo với model tốt nhất; test chỉ được đánh giá sau khi chọn checkpoint. Đây không phải so sánh với YuNet/SFace gốc.
- `comparison.json`: số tensor thật sự thay đổi và hash checkpoint.
- Các file prediction validation/test để đối chiếu metric.

Detector đánh giá precision/recall/F1 ở IoU 0.5 và landmark NME, với ngưỡng hiện diện cố định 0.5. Recognition tạo gallery centroid từ train, dùng validation chọn ngưỡng cosine theo balanced error trên genuine/wrong-identity scores; test giữ nguyên ngưỡng và báo riêng false accept người lạ khi có người lạ. Đây là phép đo gallery/probe, không được gộp thành một con số “accuracy” chung. Anti-spoof chọn ngưỡng live trên validation theo ACER rồi báo APCER/BPCER/ACER và nhóm attack trên test. Không dùng test để chỉnh threshold, epoch hay learning rate.

## Kiểm thử phần mềm

```powershell
.\.venv-training\Scripts\python.exe -B -m unittest discover -s tests -v
```

Ảnh tổng hợp ngẫu nhiên chỉ tồn tại tạm trong các bài test, nhằm chứng minh forward/backward, cập nhật trọng số, ghi/đọc checkpoint và chống rò dữ liệu hoạt động. Chúng không phải dữ liệu train của dự án, không chứng minh chất lượng khuôn mặt hay liveness. Không đưa checkpoint từ smoke test vào `ai/models`.

Kết quả kiểm thử ngày 18/10/2026: **12 tests passed**, gồm một epoch đầy đủ cho mỗi tác vụ trên dữ liệu tổng hợp tạm. Môi trường: Python 3.12.10, PyTorch 2.14.1+cpu, Pillow 12.3.0, NumPy 2.5.4; CUDA không bật trong venv này. Các phiên bản đã cài ghi trong `requirements-door.cpu.lock.txt`. Không có checkpoint dữ liệu thật được tạo trong lần kiểm thử này.

## Kế hoạch bốn người

Thu tối thiểu ba buổi độc lập cho mỗi người, mỗi buổi 50 ảnh với tư thế/ánh sáng/khoảng cách đa dạng. Duyệt vùng da mặt cho nhận diện và chống giả mạo; gán box cùng thứ tự năm điểm trên ảnh gốc cho detector. Có thể thu và rà từng người một, sau đó ghép manifest rồi train detector/embedder chung. Giữ một người ngoài nhóm train để đánh giá từ chối người lạ nếu có người tình nguyện.
