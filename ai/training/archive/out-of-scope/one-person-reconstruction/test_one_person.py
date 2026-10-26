from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "Requires torch")
class OnePersonPilotTests(unittest.TestCase):
    def fixture(self, root):
        import numpy as np
        from PIL import Image
        from pbl4_training.dataset import write_jsonl
        rng = np.random.default_rng(3)
        for split in ("train", "validation"):
            rows = []
            for i in range(3):
                path = root / f"{split}-{i}.png"
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
                rows.append({"sample_id": f"{split}-{i}", "subject_id": "person-001", "identity_id": "person-001",
                    "identity_pool": "registered", "session_id": split, "source_id": f"{split}-{i}", "device_id": "synthetic",
                    "image_path": str(path), "face_region": {"policy": "reviewed-face-skin-v1", "reviewed": True,
                        "source_image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "polygon_xy": [[8, 8], [55, 8], [55, 55], [8, 55]], "eyes_xy": [[20, 25], [43, 25]]}})
            write_jsonl(root / f"{split}.jsonl", rows)

    def test_pilot_updates_weights_without_claiming_identity_accuracy(self):
        from pbl4_training.train_one_person import main
        import torch
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            self.fixture(root)
            main(["--train-manifest", str(root / "train.jsonl"), "--validation-manifest", str(root / "validation.jsonl"),
                  "--run-dir", str(root / "run"), "--epochs", "2"])
            report = json.loads((root / "run" / "summary.json").read_text())
            self.assertGreater(report["changed_model_tensors"], 0)
            self.assertFalse(report["identity_verification_validated"])
            self.assertIsNone(report["threshold"])
            self.assertEqual(report["test_samples"], 0)
            self.assertTrue(torch.load(root / "run" / "best.pt", weights_only=True)["optimizer"]["state"])

    def test_same_session_cannot_be_used_for_validation(self):
        from pbl4_training.train_one_person import load_data
        from pbl4_training.dataset import DatasetError, read_jsonl, write_jsonl
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            self.fixture(root)
            rows = read_jsonl(root / "validation.jsonl")
            for row in rows:
                row["session_id"] = "train"
            write_jsonl(root / "validation.jsonl", rows)
            with self.assertRaisesRegex(DatasetError, "sessions must be disjoint"):
                load_data(root / "train.jsonl", root / "validation.jsonl")

    def test_reconstruction_loss_does_not_reward_background(self):
        import torch
        from pbl4_training.train_one_person import masked_mse
        target = torch.zeros(1, 3, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        prediction = torch.ones_like(target)
        first = masked_mse(prediction, target, mask)
        prediction[:, :, :2] = 100
        self.assertEqual(float(first[0]), float(masked_mse(prediction, target, mask)[0]))


if __name__ == "__main__":
    unittest.main()
