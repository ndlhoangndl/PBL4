"""Audit captures using existing YuNet as annotation assistance, not identity training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def review(session: Path):
    ai = Path(__file__).resolve().parents[1] / "ai"
    bundle = json.loads((ai / "models-manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in bundle["files"] if e["path"] == "models/face_detection_yunet_2023mar.onnx")
    model = (ai / entry["path"]).read_bytes()
    if hashlib.sha256(model).hexdigest() != entry["sha256"]:
        raise ValueError("YuNet checksum mismatch")
    detector = cv2.FaceDetectorYN.create("onnx", np.frombuffer(model, np.uint8), np.empty(0, np.uint8), (320, 320), 0.8)
    rows = [json.loads(line) for line in (session / "captures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results, seen = [], set()
    for ordinal, row in enumerate(rows, 1):
        path = session / row["image_path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        result = {"ordinal": ordinal, "sample_id": row["sample_id"], "sha256": digest,
                  "capture_hash_matches": digest == row["sha256"], "duplicate_bytes": digest in seen,
                  "annotation_reviewed": False}
        seen.add(digest)
        if image is None:
            result["error"] = "Image could not be decoded"
            results.append(result)
            continue
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        result.update(image_width=width, image_height=height, face_count=0 if faces is None else len(faces))
        if faces is not None and len(faces) == 1:
            face = faces[0]
            x, y, w, h = map(float, face[:4])
            points = face[4:14].reshape(5, 2).tolist()
            result.update(bbox_xywh=[x, y, w, h], landmarks_xy=points, detection_score=float(face[14]))
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(width, int(x + w)), min(height, int(y + h))
            if x2 > x1 and y2 > y1:
                gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                result["face_laplacian_variance"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
            # A rough skin outline only. Hairline/forehead/chin MUST be reviewed.
            normalized = [(0.25, .07), (.5, .03), (.75, .07), (.94, .28), (.95, .55),
                          (.82, .82), (.65, .96), (.5, 1.), (.35, .96), (.18, .82), (.05, .55), (.06, .28)]
            polygon = [[round(min(width - 1, max(0, x + px * w)), 2), round(min(height - 1, max(0, y + py * h)), 2)] for px, py in normalized]
            result["suggested_region"] = {"polygon_xy": polygon, "eyes_xy": sorted(points[:2], key=lambda p: p[0])}
        results.append(result)
    report = {"schema": "smartdoor-face-review-v1", "photos": len(rows),
              "proposal_source": "Existing pretrained OpenCV YuNet; suggestions only, not ground-truth labels",
              "model_sha256": entry["sha256"], "single_face_photos": sum(r.get("face_count") == 1 for r in results),
              "all_capture_hashes_match": all(r["capture_hash_matches"] for r in results), "images": results}
    destination = session / "face-review.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, ensure_ascii=False))
    print(str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    review(parser.parse_args().session.resolve())
