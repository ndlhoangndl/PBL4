"""Review visible face-skin polygons; remove hair/background before identity training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageTk

from .collect_faces import atomic_json
from .dataset import read_jsonl, write_jsonl
from .face_region import POLICY, prepare_face, validate_region_file


class RegionEditor:
    def __init__(self, root, session):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.tk, self.session = root, tk, session.resolve()
        self.rows = read_jsonl(self.session / "captures.jsonl")
        review = json.loads((self.session / "face-review.json").read_text(encoding="utf-8"))
        self.proposals = {r["sample_id"]: r for r in review["images"]}
        self.state_path = self.session / "face-regions" / "annotations.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.index = 0
        self.mode = "drag"
        self.dragging = None
        root.title("SmartDoor — Chỉ lấy da mặt, bỏ tóc và nền")
        root.geometry(f"1160x{min(850, max(640, root.winfo_screenheight() - 85))}")
        root.minsize(960, 640)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        self.title = tk.StringVar()
        ttk.Label(frame, textvariable=self.title, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Kéo các chấm đỏ theo đường da trán → má → cằm. Đưa viền xuống dưới tóc mái; không lấy tai, cổ, áo hoặc nền.", wraplength=1100).pack(anchor="w", pady=4)
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, width=780, height=420, bg="#17212b", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(body, padding=(10, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="Ảnh model sẽ nhận:").pack(anchor="w")
        self.preview_label = ttk.Label(side)
        self.preview_label.pack(pady=5)
        ttk.Label(side, text="Ngoài viền được thay bằng màu xám.\nHai chấm xanh là tâm mắt, dùng để\ncăn thẳng trước khi resize.", wraplength=250).pack(anchor="w")
        tools = ttk.Frame(frame)
        tools.pack(fill="x", pady=6)
        ttk.Button(tools, text="Vẽ lại viền", command=self.reset_polygon).pack(side="left", padx=3)
        ttk.Button(tools, text="Đặt mắt bên trái ảnh", command=lambda: self.set_mode("eye0")).pack(side="left", padx=3)
        ttk.Button(tools, text="Đặt mắt bên phải ảnh", command=lambda: self.set_mode("eye1")).pack(side="left", padx=3)
        ttk.Label(tools, text="Chuột phải: bỏ một điểm viền").pack(side="left", padx=8)
        self.status = tk.StringVar()
        ttk.Label(frame, textvariable=self.status, wraplength=1100).pack(anchor="w")
        self.confirm = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Đã kiểm tra: viền chỉ chứa da mặt; đã loại tóc/nền/cổ và chỉnh đúng hai tâm mắt.", variable=self.confirm).pack(anchor="w", pady=6)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Ảnh trước", command=lambda: self.move(-1)).pack(side="left", padx=3)
        ttk.Button(buttons, text="Ảnh tiếp (chưa lưu)", command=lambda: self.move(1)).pack(side="left", padx=3)
        self.save_button = ttk.Button(buttons, text="Lưu vùng mặt và tiếp", command=self.save)
        self.save_button.pack(side="left", padx=8)
        ttk.Label(buttons, text="Lý do bỏ:").pack(side="left")
        self.reason = tk.StringVar(value="cần chụp lại")
        ttk.Entry(buttons, textvariable=self.reason, width=18).pack(side="left", padx=4)
        ttk.Button(buttons, text="Bỏ ảnh", command=self.exclude).pack(side="left", padx=3)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _: setattr(self, "dragging", None))
        self.canvas.bind("<Button-3>", self.remove_point)
        root.update_idletasks()
        self.load()

    def set_mode(self, mode):
        self.mode = mode
        self.status.set("Bấm chính xác vào tâm mắt trên ảnh; sau đó tiếp tục kéo viền mặt.")

    def load(self):
        row = self.rows[self.index]
        self.image = Image.open(self.session / row["image_path"]).convert("RGB")
        if hashlib.sha256((self.session / row["image_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Ảnh gốc đã đổi so với manifest; cần kiểm tra trước khi gán nhãn")
        proposal = self.proposals.get(row["sample_id"], {})
        existing = self.state.get(row["sample_id"], {})
        region = existing.get("region", proposal.get("suggested_region", {}))
        self.points = [list(p) for p in region.get("polygon_xy", [])]
        w, h = self.image.size
        self.eyes = [list(p) for p in region.get("eyes_xy", [[w * .4, h * .4], [w * .6, h * .4]])]
        if "bbox_xywh" in proposal:
            x, y, bw, bh = proposal["bbox_xywh"]
            self.view = (max(0, int(x - .3 * bw)), max(0, int(y - .25 * bh)),
                         min(w, int(x + 1.3 * bw)), min(h, int(y + 1.2 * bh)))
        else:
            self.view = (0, 0, w, h)
        visible = self.image.crop(self.view)
        self.scale = min(max(300, self.canvas.winfo_width()) / visible.width,
                         max(220, self.canvas.winfo_height()) / visible.height)
        visible = visible.resize((int(visible.width * self.scale), int(visible.height * self.scale)), Image.Resampling.BILINEAR)
        self.display_image = ImageTk.PhotoImage(visible)
        self.confirm.set(False)
        self.mode = "drag" if self.points else "draw"
        accepted = sum(item.get("status") == "approved" for item in self.state.values())
        self.title.set(f"Ảnh {self.index + 1}/{len(self.rows)} — {row['subject_id']} — đã duyệt {accepted} ảnh")
        self.status.set("Viền gợi ý chưa xác định đường chân tóc. Kéo chỉnh và xem preview; tóc che trán thì loại vùng tóc hoặc bỏ ảnh.")
        self.redraw()

    def source_point(self, event):
        return [round(min(self.image.width - 1, max(0, event.x / self.scale + self.view[0])), 2),
                round(min(self.image.height - 1, max(0, event.y / self.scale + self.view[1])), 2)]

    def screen_point(self, point):
        return ((point[0] - self.view[0]) * self.scale, (point[1] - self.view[1]) * self.scale)

    def click(self, event):
        point = self.source_point(event)
        self.confirm.set(False)
        if self.mode.startswith("eye"):
            self.eyes[int(self.mode[-1])] = point
            self.mode = "drag"
        elif self.mode == "draw":
            self.points.append(point)
        else:
            distances = [((self.screen_point(p)[0] - event.x) ** 2 + (self.screen_point(p)[1] - event.y) ** 2, i) for i, p in enumerate(self.points)]
            if distances and min(distances)[0] <= 18 ** 2:
                self.dragging = min(distances)[1]
        self.redraw()

    def drag(self, event):
        if self.dragging is not None:
            self.points[self.dragging] = self.source_point(event)
            self.confirm.set(False)
            self.redraw()

    def remove_point(self, event):
        if len(self.points) > 3:
            index = min(range(len(self.points)), key=lambda i: sum((a - b) ** 2 for a, b in zip(self.points[i], self.source_point(event))))
            self.points.pop(index)
            self.confirm.set(False)
            self.redraw()

    def reset_polygon(self):
        self.points = []
        self.mode = "draw"
        self.confirm.set(False)
        self.status.set("Bấm lần lượt theo viền da mặt, theo một chiều. Đủ điểm rồi bấm Đặt mắt để kết thúc vẽ và chỉnh mắt.")
        self.redraw()

    def region(self):
        return {"policy": POLICY, "reviewed": True, "polygon_xy": self.points,
                "eyes_xy": self.eyes, "source_image_sha256": self.rows[self.index]["sha256"]}

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)
        coordinates = [value for p in self.points for value in self.screen_point(p)]
        if len(self.points) >= 3:
            self.canvas.create_polygon(coordinates, outline="#ff5050", fill="", width=2)
        for i, p in enumerate(self.points):
            x, y = self.screen_point(p)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#ff5050", outline="white")
            self.canvas.create_text(x + 9, y - 7, text=str(i + 1), fill="yellow")
        for point in self.eyes:
            x, y = self.screen_point(point)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#00ffff", outline="black")
        try:
            self.preview_image = ImageTk.PhotoImage(prepare_face(self.image, self.region(), 224))
            self.preview_label.configure(image=self.preview_image, text="")
        except ValueError:
            self.preview_label.configure(image="", text="Vẽ đủ viền hợp lệ và\nđặt hai mắt bên trong.")

    def move(self, step):
        self.index = max(0, min(len(self.rows) - 1, self.index + step))
        self.load()

    def persist(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.state_path, self.state)
        records = []
        for source in self.rows:
            decision = self.state.get(source["sample_id"], {})
            if decision.get("status") == "approved":
                records.append({key: source[key] for key in ("sample_id", "subject_id", "session_id", "device_id", "source_id")})
                records[-1].update(identity_id=source["subject_id"], identity_pool="registered",
                    image_path=(Path("..") / source["image_path"]).as_posix(), face_region=decision["region"])
        write_jsonl(self.state_path.parent / "recognition.jsonl", records)

    def save(self):
        from tkinter import messagebox
        if not self.confirm.get():
            messagebox.showinfo("SmartDoor", "Hãy chỉnh viền loại tóc/nền/cổ và xác nhận trước khi lưu.")
            return
        try:
            row = self.rows[self.index]
            region = self.region()
            validate_region_file(self.session / row["image_path"], region)
            preview = prepare_face(self.image, region, 224)
            previews = self.state_path.parent / "previews"
            previews.mkdir(parents=True, exist_ok=True)
            preview.save(previews / f"{row['sample_id']}.png")
            self.state[row["sample_id"]] = {"status": "approved", "region": region}
            self.persist()
            self.move(1)
        except Exception as error:
            messagebox.showerror("SmartDoor", str(error))

    def exclude(self):
        row = self.rows[self.index]
        self.state[row["sample_id"]] = {"status": "excluded", "reason": self.reason.get() or "operator exclusion"}
        self.persist()
        self.move(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--check-ui", action="store_true")
    args = parser.parse_args(argv)
    import tkinter as tk
    root = tk.Tk()
    editor = RegionEditor(root, args.session)
    root.update_idletasks()
    if args.check_ui:
        if editor.save_button.winfo_rooty() + editor.save_button.winfo_height() > root.winfo_rooty() + root.winfo_height():
            raise RuntimeError("Face-region editor controls are below the window")
        root.destroy()
        print("Face region editor layout and preview: PASS; no annotations saved")
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
