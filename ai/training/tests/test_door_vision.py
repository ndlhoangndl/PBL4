"""Synthetic SOFTWARE tests; outputs must never be called biometric accuracy."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires torch in the separate training environment")
class DoorVisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_all_models_update_weights_with_real_backprop(self):
        from pbl4_training.door_models import (ArcFaceHead, DoorAntiSpoof,
            DoorFaceDetector, DoorFaceEmbedder, detection_loss)
        images = torch.randn(2, 3, 64, 64)
        for model in (DoorAntiSpoof(), DoorFaceEmbedder(), DoorFaceDetector()):
            before = next(model.parameters()).detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            output = model(images)
            if isinstance(model, DoorFaceDetector):
                target = {"presence": torch.tensor([1., 0.]), "boxes": torch.tensor([[.1, .1, .8, .8], [0., 0., 0., 0.]]),
                          "landmarks": torch.rand(2, 5, 2), "visible": torch.ones(2, 5, dtype=torch.bool)}
                loss = detection_loss(output, target)
            elif isinstance(model, DoorFaceEmbedder):
                self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5))
                head = ArcFaceHead(2)
                loss = torch.nn.functional.cross_entropy(head(output, torch.tensor([0, 1])), torch.tensor([0, 1]))
            else:
                loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1]))
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            self.assertFalse(torch.equal(before, next(model.parameters())))

    def test_detection_ignores_background_geometry_and_invisible_landmarks(self):
        from pbl4_training.door_models import detection_loss
        output = {"presence": torch.zeros(2), "boxes": torch.rand(2, 4), "landmarks": torch.rand(2, 5, 2)}
        target = {"presence": torch.tensor([1., 0.]), "boxes": torch.zeros(2, 4),
                  "landmarks": torch.zeros(2, 5, 2), "visible": torch.zeros(2, 5, dtype=torch.bool)}
        first = detection_loss(output, target)
        target["boxes"][1] = 1000
        target["landmarks"][:] = 1000
        self.assertEqual(float(first), float(detection_loss(output, target)))

    def test_calibration_uses_both_classes_and_handles_negative_cosines(self):
        from pbl4_training.train_door_vision import select_threshold
        from pbl4_training.dataset import DatasetError
        threshold = select_threshold([.6, .7], [-.8, -.2], -1)
        self.assertGreater(threshold, -.2)
        self.assertLessEqual(threshold, .6)
        with self.assertRaises(DatasetError):
            select_threshold([], [.2])

    def create_splits(self, root, task):
        import numpy as np
        from PIL import Image
        from pbl4_training.dataset import write_jsonl
        rng = np.random.default_rng(42)
        folder = root / task
        folder.mkdir()
        for split in ("train", "validation", "test"):
            rows = []
            for i in range(4):
                sample = f"{task}-{split}-{i}"
                path = folder / f"{sample}.png"
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
                row = {"sample_id": sample, "source_id": sample, "image_path": str(path),
                       "subject_id": f"id-{i % 2}", "session_id": f"{split}-session", "device_id": "synthetic"}
                if task == "face_detection":
                    if i % 2:
                        row.update(no_face=True, subject_id=None)
                    else:
                        row.update(bbox_xywh=[8, 8, 40, 48], landmarks=[[16, 20, 2], [38, 20, 2], [27, 30, 2], [19, 42, 2], [36, 42, 2]])
                elif task == "face_recognition":
                    import hashlib
                    row.update(identity_id=f"id-{i % 2}", identity_pool="registered")
                    row["face_region"] = {"policy": "reviewed-face-skin-v1", "reviewed": True,
                        "source_image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "polygon_xy": [[8, 6], [56, 6], [56, 57], [8, 57]], "eyes_xy": [[20, 22], [42, 22]]}
                else:
                    row.update(capture_session_id=f"{split}-session", camera_id="synthetic", video_id=sample,
                               label="live" if i % 2 else "spoof", attack_type="none" if i % 2 else "screen_photo",
                               presentation_device_id=None if i % 2 else "synthetic-screen")
                    import hashlib
                    row["face_region"] = {"policy": "reviewed-face-skin-v1", "reviewed": True,
                        "source_image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "polygon_xy": [[8, 6], [56, 6], [56, 57], [8, 57]], "eyes_xy": [[20, 22], [42, 22]]}
                rows.append(row)
            write_jsonl(folder / f"{split}.jsonl", rows)
        return folder

    def test_end_to_end_all_three_tasks_save_changed_checkpoint_and_reports(self):
        from pbl4_training.train_door_vision import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in ("face_detection", "face_recognition", "antispoof"):
                splits = self.create_splits(root, task)
                output = root / f"run-{task}"
                command = [task, "--splits", str(splits), "--run-dir", str(output), "--epochs", "1", "--batch-size", "2", "--device", "cpu"]
                if task == "face_recognition":
                    command.append("--aligned-faces")
                main(command)
                comparison = json.loads((output / "comparison.json").read_text())
                self.assertGreater(comparison["changed_model_tensors"], 0)
                self.assertIn("test", comparison["trained"])
                checkpoint = torch.load(output / "best.pt", weights_only=True)
                self.assertEqual(checkpoint["task"], task)
                self.assertTrue(checkpoint["optimizer"]["state"])
                with self.assertRaises(SystemExit):
                    main(command)  # Never overwrite evidence, even on repeated invocation.

    def test_cross_split_session_leak_rejected(self):
        from pbl4_training.dataset import DatasetError, read_jsonl, write_jsonl
        from pbl4_training.train_door_vision import load_splits
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.create_splits(Path(temporary), "face_detection")
            rows = read_jsonl(folder / "test.jsonl")
            for row in rows:
                row["session_id"] = "train-session"
            write_jsonl(folder / "test.jsonl", rows)
            with self.assertRaisesRegex(DatasetError, "data leakage"):
                load_splits(folder, "face_detection")


if __name__ == "__main__":
    unittest.main()
