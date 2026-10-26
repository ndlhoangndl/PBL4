from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .dataset import DatasetError, read_jsonl


DIGIT_WORDS = {word: str(i) for i, word in enumerate("không một hai ba bốn năm sáu bảy tám chín".split())}


def normalize_otp(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).lower()
    tokens = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split()
    output = []
    for token in tokens:
        if token in DIGIT_WORDS:
            output.append(DIGIT_WORDS[token])
        elif token.isascii() and token.isdigit():
            output.append(token)
    return "".join(output)


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _rows_by_id(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise DatasetError(f"{kind} row needs a non-empty sample_id")
        if sample_id in indexed:
            raise DatasetError(f"{kind} contains duplicate sample_id {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def evaluate_asr(manifest: Path, predictions: Path) -> dict[str, Any]:
    refs = _rows_by_id(manifest, "ASR reference manifest")
    pred_map = _rows_by_id(predictions, "ASR predictions")
    if set(refs) != set(pred_map):
        raise DatasetError("ASR prediction sample_ids must exactly match the evaluation manifest")
    word_errors = char_errors = words = chars = otp_correct = 0
    for row in pred_map.values():
        ref = refs[row["sample_id"]]
        hypothesis = row.get("transcript")
        if not isinstance(hypothesis, str):
            raise DatasetError(f"ASR prediction {row['sample_id']!r} needs transcript")
        reference = unicodedata.normalize("NFC", ref["transcript"]).lower().split()
        recognized = unicodedata.normalize("NFC", hypothesis).lower().split()
        word_errors += edit_distance(reference, recognized)
        words += len(reference)
        ref_chars = list("".join(reference))
        hyp_chars = list("".join(recognized))
        char_errors += edit_distance(ref_chars, hyp_chars)
        chars += len(ref_chars)
        otp_correct += normalize_otp(hypothesis) == ref["otp_digits"]
    count = len(pred_map)
    return {"task": "whisper_asr", "samples": count, "wer": word_errors / max(words, 1), "cer": char_errors / max(chars, 1), "otp_exact_accuracy": otp_correct / max(count, 1)}


def evaluate_antispoof(manifest: Path, predictions: Path, threshold: float) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise DatasetError("anti-spoof threshold must be in [0,1]")
    refs = _rows_by_id(manifest, "anti-spoof reference manifest")
    pred_map = _rows_by_id(predictions, "anti-spoof predictions")
    if set(refs) != set(pred_map):
        raise DatasetError("anti-spoof prediction sample_ids must exactly match the evaluation manifest")
    spoofs = lives = spoof_accepted = live_rejected = 0
    by_attack: dict[str, dict[str, int]] = {}
    for row in pred_map.values():
        sample_id = row["sample_id"]
        score = row.get("score_live")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise DatasetError(f"anti-spoof prediction {sample_id!r} score_live must be in [0,1]")
        ref = refs[sample_id]
        accepted = score >= threshold
        attack = ref["attack_type"]
        bucket = by_attack.setdefault(attack, {"samples": 0, "false_accept": 0, "false_reject": 0})
        bucket["samples"] += 1
        if ref["label"] == "spoof":
            spoofs += 1
            spoof_accepted += accepted
            bucket["false_accept"] += accepted
        else:
            lives += 1
            live_rejected += not accepted
            bucket["false_reject"] += not accepted
    if not spoofs or not lives:
        raise DatasetError("anti-spoof test must contain both live and spoof samples")
    apcer = spoof_accepted / spoofs
    bpcer = live_rejected / lives
    return {"task": "antispoof", "samples": len(pred_map), "threshold": threshold, "apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2, "by_attack": by_attack}


def evaluate_face_pairs(predictions: Path, threshold: float) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise DatasetError("face similarity threshold must be in [0,1]")
    import csv
    with predictions.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"label", "similarity"} <= set(rows[0]):
        raise DatasetError("face pair predictions CSV needs label,similarity columns (label=genuine|impostor)")
    genuine = impostor = false_reject = false_accept = 0
    for i, row in enumerate(rows, 2):
        try:
            score = float(row["similarity"])
        except ValueError as exc:
            raise DatasetError(f"line {i}: invalid similarity") from exc
        if not 0 <= score <= 1:
            raise DatasetError(f"line {i}: cosine similarity must be in [0,1] for normalized embeddings")
        if row["label"] == "genuine":
            genuine += 1
            false_reject += score < threshold
        elif row["label"] == "impostor":
            impostor += 1
            false_accept += score >= threshold
        else:
            raise DatasetError(f"line {i}: label must be genuine or impostor")
    if not genuine or not impostor:
        raise DatasetError("face verification evaluation needs genuine and impostor pairs")
    far = false_accept / impostor
    frr = false_reject / genuine
    return {"task": "face_recognition", "pairs": len(rows), "threshold": threshold, "far_unknown_accept": far, "frr_known_reject": frr, "tar": 1 - frr}


def evaluate_detection(manifest: Path, predictions: Path, iou_threshold: float = 0.5) -> dict[str, Any]:
    if not 0 <= iou_threshold <= 1:
        raise DatasetError("IoU threshold must be in [0,1]")
    refs = _rows_by_id(manifest, "detection reference manifest")
    pred_map = _rows_by_id(predictions, "detection predictions")
    if set(refs) != set(pred_map):
        raise DatasetError("detection prediction sample_ids must exactly match the evaluation manifest")
    matched = false_positive = false_negative = 0
    ious: list[float] = []
    landmark_errors: list[float] = []
    for pred in pred_map.values():
        ref = refs[pred["sample_id"]]
        detections = pred.get("detections")
        if not isinstance(detections, list):
            raise DatasetError(f"detection prediction {pred['sample_id']!r} needs detections list")
        if ref.get("no_face") is True:
            for detection in detections:
                box = detection.get("bbox_xywh") if isinstance(detection, dict) else None
                if not isinstance(box, list) or len(box) != 4:
                    raise DatasetError("each detection needs bbox_xywh")
            false_positive += len(detections)
            continue
        gt = ref["bbox_xywh"]
        best_iou = 0.0
        best = None
        for detection in detections:
            box = detection.get("bbox_xywh") if isinstance(detection, dict) else None
            if not isinstance(box, list) or len(box) != 4:
                raise DatasetError("each detection needs bbox_xywh")
            x1, y1, w1, h1 = map(float, gt)
            x2, y2, w2, h2 = map(float, box)
            intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            union_area = w1 * h1 + w2 * h2 - intersection
            score = intersection / union_area if union_area > 0 else 0
            if score > best_iou:
                best_iou, best = score, detection
        if best is not None and best_iou >= iou_threshold:
            matched += 1
            ious.append(best_iou)
            gt_points, pred_points = ref["landmarks"], best.get("landmarks")
            if isinstance(pred_points, list) and len(pred_points) == 5:
                visible_errors = []
                for truth, estimate in zip(gt_points, pred_points):
                    if truth[2] == 2 and isinstance(estimate, list) and len(estimate) >= 2:
                        visible_errors.append(((float(truth[0]) - float(estimate[0])) ** 2 + (float(truth[1]) - float(estimate[1])) ** 2) ** 0.5)
                if visible_errors:
                    left, right = gt_points[0], gt_points[1]
                    eye_distance = ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5 if left[2] == right[2] == 2 else 0
                    scale = eye_distance or (float(gt[2]) ** 2 + float(gt[3]) ** 2) ** 0.5
                    landmark_errors.append(sum(visible_errors) / len(visible_errors) / max(scale, 1.0))
            false_positive += max(len(detections) - 1, 0)
        else:
            false_negative += 1
            false_positive += len(detections)
    precision = matched / max(matched + false_positive, 1)
    recall = matched / max(matched + false_negative, 1)
    return {"task": "face_detection", "samples": len(pred_map), "iou_threshold": iou_threshold, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12), "mean_matched_iou": sum(ious) / max(len(ious), 1), "landmark_nme": sum(landmark_errors) / len(landmark_errors) if landmark_errors else None, "landmark_samples": len(landmark_errors)}


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
