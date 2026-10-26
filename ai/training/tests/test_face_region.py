from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from pbl4_training.face_region import POLICY, prepare_face, validate_region, validate_region_file


def annotation():
    return {"policy": POLICY, "reviewed": True, "polygon_xy": [[20, 15], [79, 15], [79, 84], [20, 84]],
            "eyes_xy": [[35, 40], [65, 40]]}


class FaceRegionTests(unittest.TestCase):
    def test_editor_export_contains_only_approved_sources_with_relative_paths(self):
        from pbl4_training.review_face_regions import RegionEditor
        from pbl4_training.dataset import read_jsonl, validate_manifest
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            session = Path(temporary)
            (session / "images").mkdir()
            path = session / "images" / "original.png"
            Image.new("RGB", (100, 100), "white").save(path)
            region = annotation()
            region["source_image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            source = {"sample_id": "accepted", "subject_id": "person-001", "session_id": "capture-one",
                      "source_id": "source-one", "device_id": "camera-0", "image_path": "images/original.png"}
            editor = RegionEditor.__new__(RegionEditor)
            editor.state_path = session / "face-regions" / "annotations.json"
            editor.rows = [source, {**source, "sample_id": "excluded"}]
            editor.state = {"accepted": {"status": "approved", "region": region},
                            "excluded": {"status": "excluded", "reason": "hair obscures forehead"}}
            editor.persist()
            manifest = editor.state_path.parent / "recognition.jsonl"
            self.assertEqual(validate_manifest(manifest, "face_recognition")["samples"], 1)
            row = read_jsonl(manifest)[0]
            self.assertEqual(row["sample_id"], "accepted")
            self.assertEqual((manifest.parent / row["image_path"]).resolve(), path.resolve())
            validate_region_file(path, row["face_region"])

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "Requires training torch environment")
    def test_identity_and_antispoof_refuse_full_frame_without_reviewed_region(self):
        from pbl4_training.dataset import DatasetError
        from pbl4_training.train_door_vision import VisionDataset
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "source.png"
            Image.new("RGB", (100, 100), "white").save(path)
            for task in ("face_recognition", "antispoof"):
                with self.assertRaisesRegex(DatasetError, "Missing reviewed face_region"):
                    VisionDataset([{"sample_id": "unreviewed", "image_path": str(path)}], task, {}, 112)

    def test_outside_pixels_cannot_influence_model_input(self):
        region = annotation()
        first = np.zeros((100, 100, 3), np.uint8)
        second = np.full_like(first, 255)
        rng = np.random.default_rng(3)
        mask = np.array(validate_region(region, 100, 100)) > 0
        skin = rng.integers(0, 256, first.shape, dtype=np.uint8)
        first[mask] = skin[mask]
        second[mask] = skin[mask]
        # Background/hair outside the same outline differ completely.
        a = np.array(prepare_face(Image.fromarray(first), region))
        b = np.array(prepare_face(Image.fromarray(second), region))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (112, 112, 3))

    def test_mask_is_applied_before_rotation_interpolation(self):
        region = annotation()
        region["eyes_xy"] = [[35, 30], [65, 50]]
        first = np.zeros((100, 100, 3), np.uint8)
        second = np.full_like(first, 255)
        mask = np.array(validate_region(region, 100, 100)) > 0
        first[mask] = second[mask] = [90, 150, 200]
        np.testing.assert_array_equal(np.array(prepare_face(Image.fromarray(first), region)),
                                      np.array(prepare_face(Image.fromarray(second), region)))

    def test_unreviewed_crossed_or_invalid_eye_regions_are_rejected(self):
        region = annotation()
        region["reviewed"] = False
        with self.assertRaises(ValueError):
            validate_region(region, 100, 100)
        region = annotation()
        region["eyes_xy"][0] = [0, 0]
        with self.assertRaises(ValueError):
            validate_region(region, 100, 100)
        region = annotation()
        region["polygon_xy"] = [[10, 10], [90, 90], [10, 90], [90, 10], [90, 50]]
        with self.assertRaises(ValueError):
            validate_region(region, 100, 100)

    def test_annotations_cannot_silently_apply_to_changed_image(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "original.png"
            Image.new("RGB", (100, 100), "white").save(path)
            region = annotation()
            region["source_image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_region_file(path, region)
            Image.new("RGB", (100, 100), "black").save(path)
            with self.assertRaises(ValueError):
                validate_region_file(path, region)


if __name__ == "__main__":
    unittest.main()
