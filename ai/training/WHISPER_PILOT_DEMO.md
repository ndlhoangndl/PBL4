# Fine-tune giọng đọc OTP — bản trình bày

## Phạm vi và dữ liệu

Đây là fine-tune **decoder của Whisper Tiny đa ngôn ngữ pretrained**, không phải tự phát minh kiến trúc nhận dạng tiếng nói. Encoder âm thanh được giữ nguyên. Mã dự án thực hiện đọc dữ liệu, tạo feature/label, tính loss, backward, AdamW, lưu model và đo trước/sau. Nguồn checkpoint là [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny), được tải ghim revision/SHA256, có model card kèm theo trong cache.

Dùng 20 file WAV mono 16 kHz, tổng 129,64 giây, một người/một buổi. Người dùng đã nghe và xác nhận tất cả câu đọc đúng. `human-label-confirmation.json` lưu xác nhận cùng hash manifest. Giữ nguyên audio/nhãn, không thay nhãn bằng kết quả ASR gốc. Hai câu 9/20 có một ít chạm đỉnh và nhiều file có khoảng lặng dài; giữ làm dữ liệu pilot theo xác nhận của người thu.

Toàn bộ 20 bản ghi dùng để train. **Không tạo validation/test giả từ cùng buổi.** Số đúng OTP/WER/CER trước và sau được đo lại trên các file đã train, không phải độ chính xác trên người/buổi chưa thấy. Chưa kiểm chứng khả năng tổng quát hoặc hoạt động với micro khác; Whisper đọc nội dung, không xác minh ai đang nói.

## Mở code nào cho thầy xem?

1. `pbl4_training/train_whisper_pilot.py`: nạp checkpoint, khóa encoder, cache đặc trưng; `loss_for()` gọi model với label; vòng epoch có `loss.backward()`, clip gradient và `optimizer.step()`.
2. `pbl4_training/predict_whisper_pilot.py`: nạp **model sau fine-tune** từ thư mục local rồi nhận dạng WAV; không đưa chuỗi OTP đích vào prompt.
3. `scripts/download_training_whisper.py`: ghim revision và hash nguồn; file `.bin` CTranslate2 inference cũ không dùng làm checkpoint train.
4. `runs/whisper-tiny-otp-pilot-20261019-v1/metrics.jsonl`, `run.json`, `summary.json`: loss, số bước optimizer, số tensor thay đổi, encoder hash trước/sau và kết quả thực tế.
5. `runs/whisper-tiny-otp-pilot-20261019-v1/final-model/`: model Safetensors, config và processor/tokenizer sau train. File trọng số là nhị phân; mã `.py` mới là phần đọc trong editor.

Model nhận log-Mel từ audio → encoder giữ nguyên → decoder được fine-tune dự đoán token tiếng Việt. Cross-entropy đo sai khác với transcript. Nhãn bỏ token bắt đầu decoder bị lặp, nhưng giữ EOS để model học kết thúc câu. Batch size 1, gradient accumulation 4, AdamW lr 0.00002, 5 epoch; chọn model ở epoch cuối theo cấu hình định sẵn, không gọi đó là checkpoint tốt nhất validation.

## Kết quả lần train ngày 19/10/2026

- 5 epoch, 25 bước optimizer; 29.552.256 tham số decoder được phép train, 100 tensor thay đổi. Hash encoder trước/sau giống nhau.
- Loss đánh giá lại trên tập train giảm từ 6,8804 xuống 1,2923.
- Cùng checkpoint Whisper Tiny: trước fine-tune đúng 0/20 chuỗi OTP, sau fine-tune đúng 10/20. Đây là kết quả trên **20 file đã dùng train**, không phải độ chính xác kiểm thử độc lập.
- Kết quả 6/20 từ lần kiểm tra cũ dùng Whisper Base, không dùng làm baseline cho lần train Tiny này.
- Đã nạp lại model lưu trên đĩa: file đầu được đọc thành `năm không bảy tám bốn ba` → `507843`. Bộ kiểm tra phần mềm: 33 test đạt.

## Lệnh đã dùng

Từ `SmartDoor/ai/training`, sau khi cài `requirements-whisper-pilot.txt` trong `.venv-training`:

```powershell
.\.venv-training\Scripts\python.exe -B -m pbl4_training.train_whisper_pilot `
  --manifest ..\datasets\audio\speaker-001\audio-20261019T024336Z-83797213\whisper.jsonl `
  --checkpoint .\cache\checkpoints\whisper-tiny `
  --run-dir .\runs\whisper-tiny-otp-pilot-20261019-v1 `
  --epochs 5 --learning-rate 0.00002 --grad-accum 4 --labels-confirmed
```

Không chạy đè thư mục kết quả này. Nếu cần demo train lại, dùng tên run mới; có thể chọn `--epochs 1` để rút ngắn bản trình diễn, không thay bằng chứng lần đã chạy.

## Demo đọc file

Nhấp đúp `SmartDoor/scripts/Demo-Voice-Model.cmd`, chọn WAV. Cửa sổ trả transcript và chuỗi 6 số nếu đọc được. Nếu WAV thuộc tập train, kết quả ghi `reused_training_recording: true`; không giới thiệu đó là test độc lập.

Ứng dụng khóa cửa vẫn dùng Whisper base cũ; bản fine-tune nằm riêng, chưa tự đưa vào xác thực. Sau này cần thêm các buổi thu, giữ validation/test độc lập, đánh giá và xuất CTranslate2 trước khi xem xét thay model runtime.
