from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pbl4_training.collect_faces import CaptureSession, TARGET, pose_at, validate_subject


def png_sample(index):
    from PIL import Image
    stream = BytesIO()
    Image.new("RGB", (8, 8), (index, 100, 150)).save(stream, format="PNG")
    return stream.getvalue()


class FaceCaptureTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent)

    def test_pose_guide_has_exactly_fifty_photos(self):
        from collections import Counter
        self.assertEqual(TARGET, 50)
        counts = Counter(pose_at(i)[0] for i in range(TARGET))
        self.assertEqual(sorted(counts.values()), [5, 5, 10, 10, 20])
        with self.assertRaises(ValueError):
            pose_at(50)

    def test_subject_cannot_escape_dataset_directory(self):
        self.assertEqual(validate_subject(" person-001 "), "person-001")
        for subject in ("../someone", "..", "a/b", "a\\b", "C:\\data", "", "a" * 41):
            with self.assertRaises(ValueError):
                validate_subject(subject)

    def test_complete_session_preserves_original_bytes_and_pending_labels(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            session = CaptureSession(root, "person-001", 0)
            for index in range(TARGET):
                png = png_sample(index)
                saved = session.save(png, 8, 8, {"laplacian_variance": 10.})
                self.assertEqual(saved.read_bytes(), png)
                self.assertTrue(saved.resolve().is_relative_to(root.resolve()))
            manifest = session.directory / "captures.jsonl"
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), TARGET)
            self.assertEqual(len({row["sample_id"] for row in rows}), TARGET)
            for row in rows:
                self.assertEqual(row["annotation_status"], "pending")
                self.assertNotIn("split", row)
                self.assertNotIn("landmarks", row)
                self.assertEqual(row["session_id"], session.session_id)
                self.assertTrue((manifest.parent / row["image_path"]).is_file())
            with self.assertRaises(ValueError):
                session.save(png_sample(51), 8, 8, {})

    def test_duplicate_rejected_and_separate_sessions_do_not_overwrite(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            first = CaptureSession(root, "person-001", 0)
            saved = first.save(png_sample(1), 8, 8, {})
            with self.assertRaises(ValueError):
                first.save(png_sample(1), 8, 8, {})
            second = CaptureSession(root, "person-001", 0)
            self.assertNotEqual(first.directory, second.directory)
            self.assertTrue(saved.exists())
            self.assertEqual(len(first.rows), 1)

    def test_failed_manifest_commit_does_not_leave_image_or_advance_count(self):
        with self.temporary() as temporary:
            session = CaptureSession(Path(temporary), "person-001", 0)
            with patch("pbl4_training.collect_faces.os.replace", side_effect=OSError("simulated write failure")):
                with self.assertRaises(OSError):
                    session.save(png_sample(1), 8, 8, {})
            self.assertFalse(list((session.directory / "images").glob("*.png")))
            self.assertEqual(session.rows, [])
            self.assertEqual(session.hashes, set())
            session.save(png_sample(1), 8, 8, {})
            self.assertEqual(len(session.rows), 1)


if __name__ == "__main__":
    unittest.main()
