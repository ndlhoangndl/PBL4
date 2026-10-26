"""Local, user-triggered webcam collection. Captures are NOT annotated training data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Event, Lock, Thread
import uuid
from urllib.parse import urlsplit


POSES = (("Chính diện", 20, "Nhìn thẳng; thay đổi nhẹ khoảng cách và biểu cảm giữa các ảnh."),
         ("Hơi quay trái", 10, "Quay đầu nhẹ về bên trái của bạn, vẫn nhìn thấy rõ mặt."),
         ("Hơi quay phải", 10, "Quay đầu nhẹ về bên phải của bạn, vẫn nhìn thấy rõ mặt."),
         ("Hơi cúi", 5, "Cúi đầu nhẹ; tránh để tóc che mắt và mặt."),
         ("Hơi ngẩng", 5, "Ngẩng đầu nhẹ, giữ khuôn mặt rõ trong khung hình."))
TARGET = sum(count for _, count, _ in POSES)


def pose_at(index: int):
    if not 0 <= index < TARGET:
        raise ValueError("Photo index is outside the 50-photo guide")
    offset = 0
    for label, count, hint in POSES:
        if index < offset + count:
            return label, index - offset + 1, count, hint
        offset += count


def validate_subject(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", value):
        raise ValueError("Mã người dùng 1–40 ký tự: chữ không dấu, số, gạch ngang hoặc gạch dưới. Ví dụ person-001.")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class CaptureSession:
    """One subject, one real collection session, no artificial train/test split."""
    def __init__(self, root: Path, subject: str, camera_source):
        self.subject = validate_subject(subject)
        self.camera_source = str(camera_source)
        self.camera_index = int(camera_source) if isinstance(camera_source, int) else None
        if self.camera_index is not None and self.camera_index < 0:
            raise ValueError("Camera index must be non-negative")
        if self.camera_index is None:
            parsed = urlsplit(self.camera_source)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("URL camera ESP32 phải bắt đầu bằng http:// hoặc https:// và có địa chỉ máy chủ.")
            # Keep useful provenance but never write possible credentials or query tokens into the dataset.
            self.camera_source = f"{parsed.scheme}://{parsed.hostname}{':' + str(parsed.port) if parsed.port else ''}{parsed.path}"
            self.device_id = f"esp32-camera-{parsed.hostname}"
        else:
            self.device_id = f"local-camera-{self.camera_index}"
        self.session_id = "capture-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        self.directory = root.resolve() / self.subject / self.session_id
        (self.directory / "images").mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.hashes = set()
        atomic_json(self.directory / "session.json", {
            "schema": "smartdoor-face-capture-v1", "subject_id": self.subject,
            "session_id": self.session_id, "created_utc": utc_now(),
            "camera_source": self.camera_source,
            "camera_index": self.camera_index, "target_photos": TARGET,
            "purpose": "pilot collection; requires labels and independent sessions before training",
            "guide": [{"pose_target": label, "photos": count} for label, count, _ in POSES],
            "image_policy": "Original unmirrored camera frames; preview only is resized",
            "annotation_status": "pending; no box, landmark or liveness labels inferred",
        })

    def save(self, png: bytes, width: int, height: int, quality: dict):
        if len(self.rows) >= TARGET:
            raise ValueError("Buổi này đã đủ 50 ảnh. Dừng camera rồi chọn Buổi mới để thu tiếp.")
        if not png.startswith(b"\x89PNG\r\n\x1a\n") or width <= 0 or height <= 0:
            raise ValueError("Ảnh PNG hoặc kích thước không hợp lệ.")
        digest = hashlib.sha256(png).hexdigest()
        if digest in self.hashes:
            raise ValueError("Ảnh trùng hoàn toàn ảnh đã lưu. Hãy đổi tư thế và chụp lại.")
        ordinal = len(self.rows) + 1
        sample = f"{self.session_id}-{ordinal:02d}-{uuid.uuid4().hex[:8]}"
        relative = f"images/{sample}.png"
        destination = self.directory / relative
        row = {"sample_id": sample, "subject_id": self.subject,
               "session_id": self.session_id, "device_id": self.device_id,
               "source_id": sample, "image_path": relative, "image_width": width, "image_height": height,
               "captured_utc": utc_now(), "pose_target": pose_at(ordinal - 1)[0],
               "sha256": digest, "quality_indicators": quality,
               "annotation_status": "pending", "human_review": "one person; sharp enough; accepted by operator"}
        with destination.open("xb") as stream:
            stream.write(png)
        manifest = self.directory / "captures.jsonl"
        temporary = manifest.with_suffix(".jsonl.tmp")
        try:
            temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.rows + [row]), encoding="utf-8")
            os.replace(temporary, manifest)
        except Exception:
            destination.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            raise
        self.rows.append(row)
        self.hashes.add(digest)
        return destination


class CollectorApp:
    def __init__(self, window, output_root: Path, subject: str = "person-001"):
        import tkinter as tk
        from tkinter import ttk
        import cv2
        self.tk, self.ttk, self.cv2 = tk, ttk, cv2
        self.window, self.output_root = window, output_root
        self.camera = self.frame = self.pending = self.session = None
        self.frame_lock = Lock()
        self.reader_thread = None
        self.reader_stop = None
        self.reader_error = None
        self.frame_sequence = 0
        self.last_displayed_sequence = -1
        self.pending_quality = None
        self.preview_image = None
        self.colors = {"page": "#f3f6fb", "card": "#ffffff", "border": "#e1e8f0",
                       "text": "#17243a", "muted": "#64748b", "accent": "#2563eb",
                       "accent_dark": "#1d4ed8", "green": "#0f9f76", "preview": "#111b2b"}
        self.configure_style()
        window.title(f"SmartDoor — Thu ảnh — {subject}")
        window.geometry(f"1120x{min(900, max(760, window.winfo_screenheight() - 80))}")
        window.minsize(960, 720)
        window.configure(background=self.colors["page"])
        window.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(window, padding=(22, 18), style="App.TFrame")
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        header = ttk.Frame(outer, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.tk.Label(header, text="SD", font=("Segoe UI", 15, "bold"), bg=self.colors["accent"],
                      fg="white", width=4, height=2).pack(side="left", padx=(0, 12))
        heading = ttk.Frame(header, style="App.TFrame")
        heading.pack(side="left", fill="y")
        ttk.Label(heading, text="Thu dữ liệu khuôn mặt", style="Title.TLabel").pack(anchor="w")
        ttk.Label(heading, text="Chụp • xem lại • xác nhận • lưu trên máy", style="Subtitle.TLabel").pack(anchor="w", pady=(3, 0))
        self.subject = tk.StringVar(value=subject)
        self.subject_badge = self.tk.Label(header, text=subject, font=("Segoe UI", 10, "bold"),
                                           bg="#e8f0ff", fg="#1d4ed8", padx=12, pady=7)
        self.subject_badge.pack(side="right", padx=(10, 0))
        self.subject.trace_add("write", self.update_subject_badge)

        _, settings = self.make_card(outer, "THIẾT LẬP BUỔI CHỤP", row=1)
        settings.columnconfigure(7, weight=1)
        ttk.Label(settings, text="Mã người", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.subject_entry = ttk.Entry(settings, textvariable=self.subject, width=22)
        self.subject_entry.grid(row=0, column=1, sticky="ew", padx=(0, 16))
        ttk.Label(settings, text="Nguồn camera", style="CardMuted.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.source_kind = tk.StringVar(value="ESP32")
        self.source_selector = ttk.Combobox(settings, textvariable=self.source_kind, values=("USB", "ESP32"), state="readonly", width=8)
        self.source_selector.grid(row=0, column=3, sticky="w", padx=(0, 14))
        self.source_selector.bind("<<ComboboxSelected>>", self.source_changed)
        ttk.Label(settings, text="USB #", style="CardMuted.TLabel").grid(row=0, column=4, sticky="w", padx=(0, 5))
        self.camera_index = tk.IntVar(value=0)
        self.camera_selector = ttk.Spinbox(settings, from_=0, to=9, textvariable=self.camera_index, width=4)
        self.camera_selector.grid(row=0, column=5, sticky="w", padx=(0, 18))
        self.start_button = ttk.Button(settings, text="▶  Kết nối camera", command=self.start, style="Primary.TButton")
        self.start_button.grid(row=0, column=6, padx=(0, 7), sticky="ew")
        self.stop_button = ttk.Button(settings, text="Dừng camera", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=7, padx=(0, 7), sticky="ew")
        self.new_button = ttk.Button(settings, text="Buổi mới", command=self.new_session)
        self.new_button.grid(row=0, column=8, sticky="ew")
        ttk.Label(settings, text="URL stream ESP32", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.source_url = tk.StringVar(value="http://10.120.253.192:81/stream")
        self.source_url_entry = ttk.Entry(settings, textvariable=self.source_url)
        self.source_url_entry.grid(row=1, column=1, columnspan=8, sticky="ew", pady=(12, 0))
        ttk.Label(settings, text="Nếu preview trễ: chọn SVGA hoặc VGA, đặt Quality khoảng 12 trên trang ESP32.",
                  style="CardMuted.TLabel").grid(row=2, column=1, columnspan=8, sticky="w", pady=(7, 0))

        self.progress = tk.StringVar()
        self.guide = tk.StringVar()
        content = ttk.Frame(outer, style="App.TFrame")
        content.grid(row=2, column=0, sticky="nsew", pady=(14, 12))
        content.columnconfigure(0, weight=7, minsize=560)
        content.columnconfigure(1, weight=4, minsize=300)
        content.rowconfigure(0, weight=1)

        _, preview_body = self.make_card(content, "KHUNG HÌNH CAMERA", row=None, column=0)
        preview_body.master.grid_configure(sticky="nsew", padx=(0, 8))
        preview_body.columnconfigure(0, weight=1)
        preview_body.rowconfigure(1, weight=1)
        self.preview = tk.Label(preview_body, text="Camera đang tắt\n\nBấm ‘Kết nối camera’ để bắt đầu",
                                bg=self.colors["preview"], fg="#d8e1ee", font=("Segoe UI", 13),
                                height=18, bd=0)
        self.preview.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        ttk.Label(preview_body, text="Ảnh gốc không lật gương · ảnh xem trước được thu nhỏ",
                  style="CardMuted.TLabel").grid(row=2, column=0, sticky="w")

        right = ttk.Frame(content, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        _, progress_body = self.make_card(right, "TIẾN ĐỘ BUỔI CHỤP", row=0)
        ttk.Label(progress_body, textvariable=self.progress, style="Progress.TLabel").pack(anchor="w")
        self.progress_bar = ttk.Progressbar(progress_body, mode="determinate", maximum=TARGET)
        self.progress_bar.pack(fill="x", pady=(10, 15))
        ttk.Label(progress_body, text="TƯ THẾ TIẾP THEO", style="CardMuted.TLabel").pack(anchor="w")
        ttk.Label(progress_body, textvariable=self.guide, style="Guide.TLabel", wraplength=330,
                  justify="left").pack(anchor="w", pady=(5, 0))

        _, review_body = self.make_card(right, "XEM LẠI TRƯỚC KHI LƯU", row=1)
        ttk.Label(review_body, text="Ảnh sẽ dừng lại sau khi chụp để bạn kiểm tra.", style="CardMuted.TLabel",
                  wraplength=320, justify="left").pack(anchor="w", pady=(0, 9))
        self.review = tk.BooleanVar(value=False)
        ttk.Checkbutton(review_body, text="Đã kiểm tra: một người, mặt rõ.", variable=self.review,
                        style="Card.TCheckbutton").pack(anchor="w")

        controls = ttk.Frame(right, style="App.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.capture_button = ttk.Button(controls, text="Chụp ảnh", command=self.capture,
                                         state="disabled", style="Primary.TButton")
        self.capture_button.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=(0, 7))
        self.save_button = ttk.Button(controls, text="Lưu ảnh", command=self.save,
                                      state="disabled", style="Save.TButton")
        self.save_button.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 7))
        self.retake_button = ttk.Button(controls, text="Chụp lại", command=self.retake, state="disabled")
        self.retake_button.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(controls, text="Mở thư mục ảnh", command=self.open_folder).grid(row=1, column=1, sticky="ew", padx=(5, 0))

        status_bar = ttk.Frame(outer, style="Status.TFrame", padding=(12, 9))
        status_bar.grid(row=3, column=0, sticky="ew")
        status_bar.columnconfigure(1, weight=1)
        ttk.Label(status_bar, text="●", style="StatusDot.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.status = tk.StringVar(value="Vén tóc để lộ trán · ngồi cách camera 0,5–1 m · ánh sáng chiếu từ phía trước.")
        ttk.Label(status_bar, textvariable=self.status, style="StatusText.TLabel", wraplength=940,
                  justify="left").grid(row=0, column=1, sticky="w")
        ttk.Label(outer, text="50 ảnh là dữ liệu thô của một buổi. Box và landmark cần được gán nhãn trước khi train.",
                  style="Footer.TLabel").grid(row=4, column=0, sticky="w", pady=(9, 0))
        self.update_guide()
        self.source_changed()
        window.after(60, self.tick)

    def configure_style(self):
        style = self.ttk.Style(self.window)
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        page, card = self.colors["page"], self.colors["card"]
        style.configure("App.TFrame", background=page)
        style.configure("Card.TFrame", background=card)
        style.configure("CardHeading.TLabel", background=card, foreground=self.colors["muted"], font=("Segoe UI", 9, "bold"))
        style.configure("CardMuted.TLabel", background=card, foreground=self.colors["muted"], font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=page, foreground=self.colors["text"], font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=page, foreground=self.colors["muted"], font=("Segoe UI", 10))
        style.configure("Progress.TLabel", background=card, foreground=self.colors["text"], font=("Segoe UI", 17, "bold"))
        style.configure("Guide.TLabel", background=card, foreground=self.colors["text"], font=("Segoe UI", 11))
        style.configure("TEntry", padding=(8, 7), fieldbackground="#ffffff", foreground=self.colors["text"])
        style.configure("TCombobox", padding=(7, 6), fieldbackground="#ffffff", foreground=self.colors["text"])
        style.configure("TSpinbox", padding=(6, 6), fieldbackground="#ffffff", foreground=self.colors["text"])
        style.configure("TButton", padding=(10, 8), font=("Segoe UI", 9, "bold"), background="#e8edf5", foreground=self.colors["text"], borderwidth=0)
        style.map("TButton", background=[("active", "#dbe4f0"), ("disabled", "#edf1f6")], foreground=[("disabled", "#98a3b3")])
        style.configure("Primary.TButton", background=self.colors["accent"], foreground="white")
        style.map("Primary.TButton", background=[("active", self.colors["accent_dark"]), ("disabled", "#aabce0")], foreground=[("disabled", "#f2f5fa")])
        style.configure("Save.TButton", background=self.colors["green"], foreground="white")
        style.map("Save.TButton", background=[("active", "#0b8060"), ("disabled", "#a9cfc3")], foreground=[("disabled", "#f2f5fa")])
        style.configure("Horizontal.TProgressbar", troughcolor="#e9eef5", background=self.colors["accent"], thickness=12)
        style.configure("Card.TCheckbutton", background=card, foreground=self.colors["text"], font=("Segoe UI", 10))
        style.map("Card.TCheckbutton", background=[("active", card)])
        style.configure("Status.TFrame", background="#eaf1fb")
        style.configure("StatusDot.TLabel", background="#eaf1fb", foreground=self.colors["accent"], font=("Segoe UI", 12, "bold"))
        style.configure("StatusText.TLabel", background="#eaf1fb", foreground=self.colors["text"], font=("Segoe UI", 9))
        style.configure("Footer.TLabel", background=page, foreground=self.colors["muted"], font=("Segoe UI", 9))

    def make_card(self, parent, title, row, column=0):
        card = self.tk.Frame(parent, bg=self.colors["card"], highlightbackground=self.colors["border"],
                             highlightthickness=1, padx=15, pady=13)
        if row is not None:
            card.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        else:
            card.grid(row=0, column=column, sticky="nsew")
        card.columnconfigure(0, weight=1)
        self.ttk.Label(card, text=title, style="CardHeading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 11))
        body = self.ttk.Frame(card, style="Card.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        card.rowconfigure(1, weight=1)
        return card, body

    def update_subject_badge(self, *_args):
        value = self.subject.get().strip() or "chưa đặt mã"
        self.subject_badge.configure(text=value)
        self.window.title(f"SmartDoor — Thu ảnh — {value}")

    def source_changed(self, _event=None):
        is_usb = self.source_kind.get() == "USB"
        self.camera_selector.configure(state="normal" if is_usb else "disabled")
        self.source_url_entry.configure(state="disabled" if is_usb else "normal")

    def inform_error(self, error):
        from tkinter import messagebox
        messagebox.showerror("SmartDoor", str(error), parent=self.window)

    def update_guide(self):
        count = len(self.session.rows) if self.session else 0
        self.progress.set(f"Đã lưu {count}/{TARGET} ảnh")
        if hasattr(self, "progress_bar"):
            self.progress_bar.configure(value=count)
        if count >= TARGET:
            self.guide.set("Đã đủ 50 ảnh. Dừng camera. Dùng Buổi mới cho một lần thu riêng; không tách ảnh cùng buổi thành train/test.")
        else:
            label, ordinal, total, hint = pose_at(count)
            self.guide.set(f"Ảnh tiếp theo: {label} ({ordinal}/{total}). {hint}")

    def read_camera_loop(self, camera, stop_event):
        consecutive_failures = 0
        while not stop_event.is_set():
            try:
                success, frame = camera.read()
            except Exception as error:
                if stop_event.is_set():
                    return
                with self.frame_lock:
                    self.reader_error = f"Lỗi đọc luồng camera: {error}"
                return
            if stop_event.is_set():
                return
            if success and frame is not None:
                consecutive_failures = 0
                with self.frame_lock:
                    self.frame = frame
                    self.frame_sequence += 1
            else:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    with self.frame_lock:
                        self.reader_error = "Nguồn camera không gửi được hình. Hãy kiểm tra kết nối, URL stream hoặc camera USB."
                    return
                stop_event.wait(0.03)

    def start(self):
        try:
            subject = validate_subject(self.subject.get())
            if self.source_kind.get() == "ESP32":
                camera_source = self.source_url.get().strip()
                parsed = urlsplit(camera_source)
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    raise ValueError("Nhập URL stream ESP32, ví dụ http://10.120.253.192:81/stream.")
            else:
                camera_source = self.camera_index.get()
                if camera_source < 0:
                    raise ValueError("Chỉ số camera phải từ 0 trở lên.")
            if self.session and (subject != self.session.subject or camera_source != self.session.camera_source):
                raise ValueError("Để đổi người/camera, hãy bấm Buổi mới trước.")
            cv2 = self.cv2
            if isinstance(camera_source, str):
                camera = cv2.VideoCapture(camera_source)
                if camera.isOpened():
                    self.camera = camera
                else:
                    camera.release()
            else:
                backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if os.name == "nt" else (cv2.CAP_ANY,)
                for backend in backends:
                    camera = cv2.VideoCapture(camera_source, backend)
                    if camera.isOpened():
                        self.camera = camera
                        break
                    camera.release()
            if self.camera is None:
                if isinstance(camera_source, str):
                    raise ValueError("Không mở được stream ESP32. Hãy kiểm tra ESP32 và máy tính cùng Wi‑Fi, mở URL trên trình duyệt, rồi thử http://<IP-ESP32>:81/stream. Một số firmware chỉ hỗ trợ /stream trên cổng 81.")
                raise ValueError("Không mở được webcam. Đóng ứng dụng Camera/Meet đang giữ thiết bị; kiểm tra quyền Camera của Windows hoặc thử chỉ số 1.")
            if isinstance(camera_source, int):
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            else:
                # Drain MJPEG continuously; the reader thread keeps only the newest frame
                # so network/decoder queues do not make the preview show stale frames.
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.session is None:
                self.session = CaptureSession(self.output_root, subject, camera_source)
            self.reader_stop = Event()
            with self.frame_lock:
                self.frame = None
                self.reader_error = None
                self.frame_sequence = 0
                self.last_displayed_sequence = -1
            self.reader_thread = Thread(target=self.read_camera_loop, args=(self.camera, self.reader_stop), daemon=True)
            self.reader_thread.start()
            self.subject_entry.configure(state="disabled")
            self.camera_selector.configure(state="disabled")
            self.source_selector.configure(state="disabled")
            self.source_url_entry.configure(state="disabled")
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.new_button.configure(state="disabled")
            self.capture_button.configure(state="normal" if len(self.session.rows) < TARGET else "disabled")
            self.status.set("Đang xem camera. Đổi tư thế theo hướng dẫn rồi bấm Chụp ảnh.")
        except Exception as error:
            self.stop()
            self.inform_error(error)

    def stop(self):
        if self.reader_stop is not None:
            self.reader_stop.set()
        if self.camera is not None:
            self.camera.release()
        self.camera = self.pending = None
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=0.25)
        self.reader_thread = None
        self.reader_stop = None
        with self.frame_lock:
            self.frame = None
            self.reader_error = None
        self.review.set(False)
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.new_button.configure(state="normal")
        self.subject_entry.configure(state="normal")
        self.camera_selector.configure(state="normal")
        self.source_selector.configure(state="readonly")
        self.source_changed()
        for button in (self.capture_button, self.save_button, self.retake_button):
            button.configure(state="disabled")
        self.preview.configure(image="", text="Camera đã dừng", height=18)
        self.preview_image = None
        self.status.set("Camera đã dừng. Ảnh đã lưu vẫn ở trong thư mục bộ dữ liệu.")

    def new_session(self):
        if self.camera is not None:
            return
        self.session = None
        self.review.set(False)
        self.update_guide()
        self.status.set("Buổi mới sẽ tạo thư mục riêng khi bật camera. Dùng cùng mã người nếu vẫn là bạn.")

    def show_frame(self, frame):
        height, width = frame.shape[:2]
        available_height = max(150, min(405, self.window.winfo_height() - 390))
        scale = min(720 / width, available_height / height, 1)
        preview = self.cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
        success, ppm = self.cv2.imencode(".ppm", preview)
        if success:
            self.preview_image = self.tk.PhotoImage(data=ppm.tobytes(), format="PPM")
            self.preview.configure(image=self.preview_image, text="", height=0)

    def tick(self):
        if self.camera is not None and self.pending is None:
            try:
                with self.frame_lock:
                    frame = self.frame
                    sequence = self.frame_sequence
                    reader_error = self.reader_error
                if reader_error:
                    self.stop()
                    self.inform_error(reader_error)
                elif frame is not None and sequence != self.last_displayed_sequence:
                    self.last_displayed_sequence = sequence
                    self.show_frame(frame)
            except Exception as error:
                self.stop()
                self.inform_error(error)
        self.window.after(40, self.tick)

    def capture(self):
        with self.frame_lock:
            frame = self.frame
        if frame is None or self.session is None or len(self.session.rows) >= TARGET:
            return
        self.pending = frame.copy()
        gray = self.cv2.cvtColor(self.pending, self.cv2.COLOR_BGR2GRAY)
        blur = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        brightness = float(gray.mean())
        self.pending_quality = {"laplacian_variance": round(blur, 2), "mean_brightness": round(brightness, 2)}
        warnings = []
        if blur < 60:
            warnings.append("có thể bị nhòe")
        if brightness < 45:
            warnings.append("có thể quá tối")
        if brightness > 215:
            warnings.append("có thể quá sáng")
        self.show_frame(self.pending)
        self.review.set(False)
        self.capture_button.configure(state="disabled")
        self.save_button.configure(state="normal")
        self.retake_button.configure(state="normal")
        self.status.set("Ảnh đang chờ xem lại. " + (("Gợi ý: " + ", ".join(warnings) + ". ") if warnings else "") + "Kiểm tra mặt rõ và chỉ có một người; chưa có tự động kiểm tra mặt/tư thế.")

    def retake(self):
        self.pending = None
        with self.frame_lock:
            self.frame = None
            self.last_displayed_sequence = self.frame_sequence
        self.review.set(False)
        self.save_button.configure(state="disabled")
        self.retake_button.configure(state="disabled")
        self.capture_button.configure(state="normal" if self.camera is not None and self.session and len(self.session.rows) < TARGET else "disabled")
        self.status.set("Đang xem camera. Đổi tư thế nhẹ rồi chụp; ảnh chưa lưu vừa rồi đã được bỏ.")

    def save(self):
        if self.pending is None:
            return
        if not self.review.get():
            self.inform_error("Hãy xem lại ảnh và tích xác nhận trước khi lưu, hoặc chọn Chụp lại.")
            return
        try:
            success, png = self.cv2.imencode(".png", self.pending)
            if not success:
                raise ValueError("Không mã hóa được ảnh PNG.")
            height, width = self.pending.shape[:2]
            path = self.session.save(png.tobytes(), width, height, self.pending_quality)
            self.retake()
            self.update_guide()
            self.status.set(f"Đã lưu ảnh {len(self.session.rows)}/{TARGET}. " + ("Đã đủ 50 ảnh; có thể dừng camera." if len(self.session.rows) == TARGET else "Đổi tư thế rồi chụp tiếp."))
        except Exception as error:
            self.inform_error(error)

    def open_folder(self):
        directory = self.session.directory if self.session else self.output_root
        directory.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(directory))

    def close(self):
        if self.reader_stop is not None:
            self.reader_stop.set()
        if self.camera is not None:
            self.camera.release()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=0.25)
        self.window.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets" / "face-capture")
    parser.add_argument("--check-ui", action="store_true", help="Build and close UI using a synthetic preview; never open the camera")
    parser.add_argument("--subject", default="person-001", help="Identity code, e.g. person-002")
    args = parser.parse_args(argv)
    import tkinter as tk
    root = tk.Tk()
    app = CollectorApp(root, args.output_root, args.subject)
    if args.check_ui:
        import numpy as np
        import tempfile
        root.update_idletasks()
        app.show_frame(np.zeros((360, 640, 3), dtype=np.uint8))
        root.update_idletasks()
        if app.save_button.winfo_rooty() + app.save_button.winfo_height() > root.winfo_rooty() + root.winfo_height():
            raise RuntimeError("Capture controls extend below the window")
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tests") as temporary:
            app.session = CaptureSession(Path(temporary), "synthetic-ui-test", 0)
            frame = np.random.default_rng(42).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
            app.frame = frame
            app.capture()
            errors = []
            app.inform_error = lambda error: errors.append(str(error))
            app.save()
            if len(errors) != 1 or app.session.rows:
                raise RuntimeError("Review confirmation gate failed")
            app.review.set(True)
            app.save()
            if len(app.session.rows) != 1 or len(errors) != 1:
                raise RuntimeError("Synthetic capture/save failed")
            path = app.session.directory / app.session.rows[0]["image_path"]
            decoded = app.cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), app.cv2.IMREAD_COLOR)
            if not np.array_equal(decoded, frame) or app.camera is not None:
                raise RuntimeError("Saved image changed or a real camera was opened")
        app.close()
        print("Tk layout, preview, review gate, and original-resolution PNG save: PASS; camera was not opened")
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
