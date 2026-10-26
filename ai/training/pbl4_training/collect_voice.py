"""Record local labeled OTP speech. Microphone opens only after Record is clicked."""
from __future__ import annotations
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import secrets
import threading
import uuid
import wave

from .collect_faces import atomic_json, validate_subject
from .metrics import normalize_otp

WORDS = "không một hai ba bốn năm sáu bảy tám chín".split()
SPEAKING_RATES = ("slow", "normal", "fast")


def prompt_for(digits):
    if len(digits) != 6 or any(c not in "0123456789" for c in digits):
        raise ValueError("OTP label must contain exactly six ASCII digits")
    return " ".join(WORDS[int(c)] for c in digits)


def wav_bytes(pcm, rate):
    memory = BytesIO()
    with wave.open(memory, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)
    return memory.getvalue()


class AudioSession:
    def __init__(self, root, speaker, device):
        self.speaker = validate_subject(speaker)
        self.device = device
        self.session_id = "audio-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        self.directory = root.resolve() / self.speaker / self.session_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.hashes = set()
        atomic_json(self.directory / "session.json", {"speaker_id": self.speaker, "session_id": self.session_id,
            "device": str(device), "purpose": "local labeled OTP speech; not speaker identity verification"})

    def save(self, pcm, rate, digits, transcript, speaking_rate="normal"):
        if not isinstance(rate, int) or not 8000 <= rate <= 192000:
            raise ValueError("Invalid sample rate")
        duration = len(pcm) / (2 * rate)
        if len(pcm) % 2 or not .3 <= duration <= 20:
            raise ValueError("Bản ghi phải dài 0,3–20 giây, PCM mono 16-bit.")
        prompt_for(digits)
        if speaking_rate not in SPEAKING_RATES:
            raise ValueError("Speaking rate must be slow, normal, or fast")
        if normalize_otp(transcript) != digits:
            raise ValueError("Nhãn câu đọc không khớp chuỗi 6 số. Hãy thu lại hoặc sửa transcript đúng.")
        sample = f"{self.session_id}-{len(self.rows)+1:03d}-{uuid.uuid4().hex[:6]}"
        path = self.directory / f"{sample}.wav"
        row = {"sample_id": sample, "speaker_id": self.speaker, "session_id": self.session_id,
            "device_id": f"local-microphone-{self.device}", "source_id": sample, "audio_path": path.name,
            "transcript": transcript.strip(), "otp_digits": digits, "sample_rate": rate,
            "channels": 1, "duration_sec": duration, "speaking_rate": speaking_rate,
            "human_review": "operator listened and confirmed transcript"}
        data = wav_bytes(pcm, rate)
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.hashes:
            raise ValueError("Âm thanh trùng hoàn toàn một bản đã lưu; hãy ghi câu mới thật sự.")
        row["sha256"] = digest
        if any(r["otp_digits"] == digits for r in self.rows):
            raise ValueError("Chuỗi số này đã được lưu trong buổi; chọn Câu mới.")
        temporary = self.directory / "whisper.jsonl.tmp"
        try:
            with path.open("xb") as stream:
                stream.write(data)
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows + [row]), encoding="utf-8")
            os.replace(temporary, self.directory / "whisper.jsonl")
        except Exception:
            path.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            raise
        self.rows.append(row)
        self.hashes.add(digest)
        return path


class VoiceApp:
    def __init__(self, root, output):
        import tkinter as tk
        from tkinter import ttk
        import sounddevice as sd
        self.root, self.output, self.tk, self.sd = root, output, tk, sd
        self.stream = self.session = self.pcm = None
        self.frames = []
        self.rate = 16000
        self.recorded_frames = 0
        self.record_error = ""
        self.recorded_device = self.recorded_speaker = None
        self.recorded_speaking_rate = None
        self.playback_error = ""
        self.playing = False
        self.playback_done = threading.Event()
        root.title("SmartDoor — Thu giọng đọc OTP")
        root.geometry("860x560")
        root.minsize(760, 520)
        root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Thu âm đọc sáu chữ số", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Đây là câu thu dữ liệu, không phải mã mở cửa. Micro chỉ bật khi bấm Ghi âm.", wraplength=800).pack(anchor="w", pady=8)
        settings = ttk.Frame(frame)
        settings.pack(fill="x", pady=5)
        ttk.Label(settings, text="Mã người nói:").pack(side="left")
        self.speaker = tk.StringVar(value="speaker-001")
        ttk.Entry(settings, textvariable=self.speaker, width=20).pack(side="left", padx=8)
        ttk.Label(settings, text="Tốc độ đọc:").pack(side="left")
        self.speaking_rate = tk.StringVar(value="normal")
        ttk.Combobox(settings, state="readonly", textvariable=self.speaking_rate,
                     values=SPEAKING_RATES, width=10).pack(side="left", padx=8)
        self.devices = []
        try:
            self.devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
        except Exception:
            pass
        self.device = ttk.Combobox(frame, state="readonly", values=[f"{i}: {d['name']}" for i, d in self.devices], width=85)
        self.device.pack(fill="x", pady=8)
        if self.devices:
            preferred = next((j for j, (i, _) in enumerate(self.devices) if i == sd.default.device[0]), 0)
            self.device.current(preferred)
        self.digits = tk.StringVar()
        self.transcript = tk.StringVar()
        ttk.Label(frame, textvariable=self.digits, font=("Consolas", 28, "bold")).pack(pady=(14, 5))
        ttk.Label(frame, text="Chọn chậm / bình thường / nhanh, đọc từng chữ số; sửa transcript nếu cần:").pack(anchor="w")
        ttk.Entry(frame, textvariable=self.transcript, font=("Segoe UI", 15)).pack(fill="x", pady=8)
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=10)
        self.record_button = ttk.Button(controls, text="Ghi âm", command=self.record)
        self.record_button.pack(side="left", padx=5)
        ttk.Button(controls, text="Dừng", command=self.stop).pack(side="left", padx=5)
        ttk.Button(controls, text="Nghe lại", command=self.play).pack(side="left", padx=5)
        self.save_button = ttk.Button(controls, text="Lưu bản ghi", command=self.save)
        self.save_button.pack(side="left", padx=5)
        ttk.Button(controls, text="Câu mới / bỏ bản này", command=self.next_prompt).pack(side="left", padx=5)
        self.review = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Tôi đã nghe lại: bản ghi rõ, câu chữ và chuỗi số đúng nội dung đọc.", variable=self.review).pack(anchor="w", pady=8)
        self.status = tk.StringVar(value="Đặt micro cách miệng khoảng 15–25 cm; thu 20 câu cho buổi thử đầu tiên.")
        ttk.Label(frame, textvariable=self.status, wraplength=800).pack(anchor="w", pady=8)
        ttk.Button(frame, text="Mở thư mục bản ghi", command=self.open_folder).pack(anchor="w")
        self.next_prompt()
        root.after(100, self.tick)

    def error(self, message):
        from tkinter import messagebox
        messagebox.showerror("SmartDoor", str(message))

    def next_prompt(self):
        if self.stream is not None or self.playing:
            return
        used = {r["otp_digits"] for r in self.session.rows} if self.session else set()
        digits = "".join(secrets.choice("0123456789") for _ in range(6))
        while digits in used:
            digits = "".join(secrets.choice("0123456789") for _ in range(6))
        self.digits.set(digits)
        self.transcript.set(prompt_for(digits))
        self.review.set(False)
        self.pcm = None

    def record(self):
        if self.stream is not None or self.playing:
            return
        try:
            self.speaker.set(validate_subject(self.speaker.get()))
            if not self.devices or self.device.current() < 0:
                raise ValueError("Không tìm thấy micro. Kiểm tra quyền Microphone của Windows và thiết bị đầu vào.")
            index, information = self.devices[self.device.current()]
            if self.session and (self.session.speaker != self.speaker.get() or self.session.device != index):
                raise ValueError("Giữ nguyên mã người/micro trong buổi. Đóng và mở lại công cụ để bắt đầu buổi mới.")
            self.rate = 16000
            try:
                self.sd.check_input_settings(device=index, channels=1, dtype="int16", samplerate=self.rate)
            except Exception:
                self.rate = int(information["default_samplerate"])
            self.frames, self.recorded_frames, self.record_error = [], 0, ""
            self.recorded_device, self.recorded_speaker = index, self.speaker.get()
            self.recorded_speaking_rate = self.speaking_rate.get()
            self.pcm = None
            self.review.set(False)
            def callback(data, frames, time_info, status):
                self.frames.append(bytes(data))
                self.recorded_frames += frames
                if status:
                    self.record_error = str(status)
            self.stream = self.sd.RawInputStream(device=index, channels=1, dtype="int16", samplerate=self.rate, callback=callback)
            self.stream.start()
            self.status.set("Đang ghi. Đọc câu một lần rồi bấm Dừng; tự dừng sau khoảng 10 giây.")
        except Exception as error:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            self.error(error)

    def stop(self):
        if self.stream is None:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.pcm = b"".join(self.frames)
        import numpy as np
        audio = np.frombuffer(self.pcm, dtype="<i2").astype(float) / 32768
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0
        warning = " Có thể quá nhỏ/im lặng; kiểm tra lại micro." if rms < .005 else ""
        if self.record_error:
            warning += " Có lỗi luồng âm thanh; hãy ghi lại."
        self.status.set(f"Đã thu {len(audio)/self.rate:.1f} giây, {self.rate} Hz. Nghe lại trước khi lưu." + warning)

    def play(self):
        if self.stream is not None or self.playing or not self.pcm:
            return
        import winsound
        payload = wav_bytes(self.pcm, self.rate)
        self.playing = True
        self.playback_error = ""
        self.playback_done.clear()
        def worker():
            try:
                winsound.PlaySound(payload, winsound.SND_MEMORY)
            except Exception as error:
                self.playback_error = str(error)
            finally:
                self.playback_done.set()
        threading.Thread(target=worker, daemon=True).start()

    def save(self):
        if self.stream is not None or self.playing:
            return
        if not self.pcm or not self.review.get():
            self.error("Hãy ghi, nghe lại và tích xác nhận trước khi lưu.")
            return
        if self.record_error:
            self.error("Luồng âm thanh có lỗi. Thu lại để tránh mẫu bị mất đoạn.")
            return
        try:
            index = self.devices[self.device.current()][0]
            if (index != self.recorded_device or self.speaker.get() != self.recorded_speaker
                    or self.speaking_rate.get() != self.recorded_speaking_rate):
                raise ValueError("Mã người, micro hoặc tốc độ đọc đã đổi sau khi ghi. Hãy thu lại với đúng thông tin.")
            if self.session is None:
                self.session = AudioSession(self.output, self.speaker.get(), index)
            self.session.save(self.pcm, self.rate, self.digits.get(), self.transcript.get(), self.recorded_speaking_rate)
            count = len(self.session.rows)
            self.next_prompt()
            self.status.set(f"Đã lưu {count} câu. " + ("Đã đủ 20 câu thử; có thể đóng cửa sổ." if count >= 20 else "Tiếp tục đọc câu mới."))
        except Exception as error:
            self.error(error)

    def tick(self):
        if self.stream is not None and self.recorded_frames / self.rate >= 10:
            self.stop()
        if self.playing and self.playback_done.is_set():
            self.playing = False
            if self.playback_error:
                self.review.set(False)
                self.error(self.playback_error)
        self.root.after(100, self.tick)

    def open_folder(self):
        path = self.session.directory if self.session else self.output
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.playing:
            import winsound
            winsound.PlaySound(None, 0)
        self.root.destroy()


def main():
    import argparse
    import tkinter as tk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-ui", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    app = VoiceApp(root, Path(__file__).resolve().parents[2] / "datasets" / "audio")
    if args.check_ui:
        root.update_idletasks()
        if app.stream is not None:
            raise RuntimeError("Microphone opened during UI initialization")
        app.close()
        print("Voice collector UI: PASS; microphone was not opened")
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
