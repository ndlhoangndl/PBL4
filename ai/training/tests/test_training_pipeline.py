from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pbl4_training.dataset import DatasetError, read_jsonl, save_split_bundle, split_manifest, validate_manifest
from pbl4_training.metrics import evaluate_asr, evaluate_antispoof, evaluate_detection, normalize_otp


def face_row(index: int, *, source: str | None = None, device: str | None = None) -> dict:
    return {
        "sample_id": f"face-{index}", "subject_id": "subject-a", "session_id": f"session-{index}",
        "device_id": device or f"camera-{index}", "source_id": source or f"source-{index}",
        "image_path": f"images/{index}.jpg", "image_width": 640, "image_height": 480,
        "bbox_xywh": [10, 10, 100, 120], "landmarks": [[30, 40, 2], [80, 40, 2], [55, 60, 2], [35, 90, 2], [75, 90, 2]],
    }


class TrainingPipelineTests(unittest.TestCase):
    def test_deterministic_split_keeps_source_lineage_together_and_paths_resolve(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            manifest = root / "manifest.jsonl"
            # One augmented sample shares source_id with another capture row; both must stay together.
            rows = [face_row(i, source="same-origin" if i in (0, 1) else None) for i in range(12)]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            split = split_manifest(manifest, "face_detection", seed=12)
            again = split_manifest(manifest, "face_detection", seed=12)
            self.assertEqual({key: [x["sample_id"] for x in value] for key, value in split.items()},
                             {key: [x["sample_id"] for x in value] for key, value in again.items()})
            self.assertTrue(all(split_name == next(name for name, items in split.items() if any(row["sample_id"] == "face-0" for row in items))
                                for split_name, items in split.items() if any(row["sample_id"] == "face-1" for row in items)))
            save_split_bundle(root / "splits", "face_detection", split, 12, manifest)
            saved = read_jsonl(root / "splits" / "train.jsonl") + read_jsonl(root / "splits" / "validation.jsonl") + read_jsonl(root / "splits" / "test.jsonl")
            self.assertTrue(all(Path(row["image_path"]).is_absolute() for row in saved))

    def test_holdout_field_groups_devices_without_forcing_all_samples_to_test(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            manifest = Path(temp) / "manifest.jsonl"
            rows = [face_row(i, device=f"camera-{i // 2}") for i in range(12)]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            split = split_manifest(manifest, "face_detection", seed=4, holdout_field="device_id")
            self.assertTrue(all(split.values()))
            device_splits: dict[str, set[str]] = {}
            for name, items in split.items():
                for row in items:
                    device_splits.setdefault(row["device_id"], set()).add(name)
            self.assertTrue(all(len(names) == 1 for names in device_splits.values()))
            self.assertTrue(any(names == {"test"} for names in device_splits.values()))
            self.assertTrue(any(names == {"train"} for names in device_splits.values()))
            self.assertTrue(any(names == {"validation"} for names in device_splits.values()))

    def test_validator_rejects_source_leakage_between_splits(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            manifest = Path(temp) / "manifest.jsonl"
            first, second = face_row(1, source="same"), face_row(2, source="same")
            first["split"], second["split"] = "train", "test"
            manifest.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "source_id=.*both"):
                validate_manifest(manifest, "face_detection", check_files=False)

    def test_otp_keeps_leading_zero_semantics_and_metrics(self):
        self.assertEqual(normalize_otp("không tám ba bảy hai năm"), "083725")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            refs = [{"sample_id": "a", "transcript": "không tám", "otp_digits": "08"}]
            (root / "ref.jsonl").write_text(json.dumps(refs[0]) + "\n", encoding="utf-8")
            (root / "pred.jsonl").write_text(json.dumps({"sample_id": "a", "transcript": "không 8"}) + "\n", encoding="utf-8")
            result = evaluate_asr(root / "ref.jsonl", root / "pred.jsonl")
            self.assertEqual(result["otp_exact_accuracy"], 1)

    def test_metrics_reject_duplicate_prediction_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            (root / "ref.jsonl").write_text(json.dumps({"sample_id": "a", "transcript": "không", "otp_digits": "0"}) + "\n", encoding="utf-8")
            duplicate = {"sample_id": "a", "transcript": "không"}
            (root / "pred.jsonl").write_text(json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "duplicate sample_id"):
                evaluate_asr(root / "ref.jsonl", root / "pred.jsonl")

    def test_antispoof_reports_attack_type_counts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            refs = [
                {"sample_id": "live", "label": "live", "attack_type": "none"},
                {"sample_id": "print", "label": "spoof", "attack_type": "print"},
            ]
            (root / "ref.jsonl").write_text("".join(json.dumps(row) + "\n" for row in refs), encoding="utf-8")
            pred = [{"sample_id": "live", "score_live": .9}, {"sample_id": "print", "score_live": .8}]
            (root / "pred.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pred), encoding="utf-8")
            result = evaluate_antispoof(root / "ref.jsonl", root / "pred.jsonl", .5)
            self.assertEqual(result["apcer"], 1)
            self.assertEqual(result["by_attack"]["print"]["false_accept"], 1)

    def test_detection_metric_counts_false_boxes_on_background_images(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            refs = [
                {"sample_id": "face", "bbox_xywh": [10, 10, 20, 20], "landmarks": [[12, 12, 2], [28, 12, 2], [20, 18, 2], [14, 26, 2], [26, 26, 2]]},
                {"sample_id": "background", "no_face": True},
            ]
            preds = [
                {"sample_id": "face", "detections": [{"bbox_xywh": [10, 10, 20, 20], "landmarks": [[12, 12], [28, 12], [20, 18], [14, 26], [26, 26]]}]},
                {"sample_id": "background", "detections": [{"bbox_xywh": [1, 1, 5, 5]}]},
            ]
            (root / "ref.jsonl").write_text("".join(json.dumps(row) + "\n" for row in refs), encoding="utf-8")
            (root / "pred.jsonl").write_text("".join(json.dumps(row) + "\n" for row in preds), encoding="utf-8")
            result = evaluate_detection(root / "ref.jsonl", root / "pred.jsonl")
            self.assertEqual(result["precision"], .5)
            self.assertEqual(result["recall"], 1)


if __name__ == "__main__":
    unittest.main()
