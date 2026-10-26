from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


TASKS = ("face_detection", "face_recognition", "whisper_asr", "antispoof")
SPLITS = ("train", "validation", "test")
COMMON = ("sample_id", "session_id", "device_id", "source_id")
REQUIRED = {
    "face_detection": COMMON + ("image_path",),
    "face_recognition": COMMON + ("subject_id", "identity_id", "identity_pool", "image_path"),
    "whisper_asr": COMMON + ("speaker_id", "audio_path", "transcript", "otp_digits", "sample_rate", "channels", "duration_sec"),
    "antispoof": ("sample_id", "subject_id", "capture_session_id", "camera_id", "video_id", "image_path", "label", "attack_type", "presentation_device_id", "source_id"),
}


class DatasetError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{line_no}: every JSONL line must be an object")
            row["_line"] = line_no
            rows.append(row)
    if not rows:
        raise DatasetError(f"{path}: no manifest rows")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            stream.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")


def _file_path(row: dict[str, Any], root: Path, key: str, line: int) -> Path:
    raw = row.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetError(f"line {line}: {key} must be a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise DatasetError(f"line {line}: missing file for {key}: {path}")
    if path.stat().st_size == 0:
        raise DatasetError(f"line {line}: empty file for {key}: {path}")
    return path


def _image_size(path: Path, line: int) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DatasetError("Pillow is required to decode images; install requirements-data.txt") from exc
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise DatasetError(f"line {line}: corrupt or unsupported image: {path}") from exc
    if width < 1 or height < 1:
        raise DatasetError(f"line {line}: image has invalid dimensions: {path}")
    return width, height


def _audio_metadata(path: Path, line: int) -> tuple[int, int, float]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise DatasetError("soundfile is required to inspect audio; install requirements-data.txt") from exc
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise DatasetError(f"line {line}: corrupt or unsupported audio: {path}") from exc
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise DatasetError(f"line {line}: audio has invalid metadata: {path}")
    return int(info.samplerate), int(info.channels), info.frames / info.samplerate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, label: str, line: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"line {line}: {label} must be numeric")
    result = float(value)
    if not (result == result and abs(result) != float("inf")):
        raise DatasetError(f"line {line}: {label} must be finite")
    return result


def validate_manifest(path: Path, task: str, check_files: bool = True) -> dict[str, Any]:
    if task not in TASKS:
        raise DatasetError(f"unknown task {task!r}; choose one of {', '.join(TASKS)}")
    manifest = path.resolve()
    rows = read_jsonl(manifest)
    seen_ids: set[str] = set()
    seen_hashes: dict[str, tuple[str, str]] = {}
    seen_split_groups: dict[tuple[str, str], str] = {}
    split_counts = defaultdict(int)
    identity_pools: dict[str, str] = {}
    root = manifest.parent

    for row in rows:
        line = row.pop("_line")
        missing = [key for key in REQUIRED[task] if key not in row]
        if missing:
            raise DatasetError(f"line {line}: missing fields: {', '.join(missing)}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise DatasetError(f"line {line}: sample_id must be a non-empty string")
        if sample_id in seen_ids:
            raise DatasetError(f"line {line}: duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        for key in REQUIRED[task]:
            if key.endswith("_id") and key not in ("presentation_device_id",) and (not isinstance(row.get(key), str) or not row[key].strip()):
                raise DatasetError(f"line {line}: {key} must be a non-empty string")
        if "split" in row:
            if row["split"] not in SPLITS:
                raise DatasetError(f"line {line}: split must be one of {SPLITS}")
            split_counts[row["split"]] += 1

        file_key = "audio_path" if task in ("whisper_asr",) else "image_path"
        data_file = _file_path(row, root, file_key, line) if check_files else Path(row[file_key])
        if check_files:
            digest = _sha256_file(data_file)
            if digest in seen_hashes:
                other_id, other_path = seen_hashes[digest]
                raise DatasetError(f"line {line}: duplicate file bytes for {sample_id!r}; matches {other_id!r} ({other_path})")
            seen_hashes[digest] = (sample_id, str(data_file))

        if task in ("face_detection", "face_recognition", "antispoof"):
            width, height = _image_size(data_file, line) if check_files else (row.get("image_width"), row.get("image_height"))
            if task == "face_detection":
                no_face = row.get("no_face", False)
                if not isinstance(no_face, bool):
                    raise DatasetError(f"line {line}: no_face must be boolean when present")
                if no_face:
                    if "bbox_xywh" in row or "landmarks" in row:
                        raise DatasetError(f"line {line}: negative no_face sample must not carry face annotations")
                else:
                    if not isinstance(row.get("subject_id"), str) or not row["subject_id"].strip():
                        raise DatasetError(f"line {line}: positive face detection sample needs subject_id")
                    box = row.get("bbox_xywh")
                    if not isinstance(box, list) or len(box) != 4:
                        raise DatasetError(f"line {line}: bbox_xywh must be [x,y,width,height] in pixels")
                    x, y, w, h = [_number(v, "bbox_xywh", line) for v in box]
                    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                        raise DatasetError(f"line {line}: bbox_xywh lies outside image dimensions {width}x{height}")
                    points = row.get("landmarks")
                    if not isinstance(points, list) or len(points) != 5:
                        raise DatasetError(f"line {line}: landmarks must contain five [x,y,visibility] points")
                    for point in points:
                        if not isinstance(point, list) or len(point) != 3:
                            raise DatasetError(f"line {line}: each landmark must be [x,y,visibility]")
                        px, py, visibility = point
                        if visibility not in (0, 1, 2):
                            raise DatasetError(f"line {line}: landmark visibility must be 0 (invalid), 1 (occluded), or 2 (visible)")
                        if visibility == 0:
                            continue
                        px, py = _number(px, "landmark x", line), _number(py, "landmark y", line)
                        if not 0 <= px < width or not 0 <= py < height:
                            raise DatasetError(f"line {line}: landmark coordinate outside image dimensions {width}x{height}")
            if task == "antispoof":
                if row["label"] not in ("live", "spoof"):
                    raise DatasetError(f"line {line}: label must be live or spoof")
                if row["label"] == "live" and row["attack_type"] not in ("none", "live"):
                    raise DatasetError(f"line {line}: live samples must use attack_type='none' or 'live'")
                if row["label"] == "spoof" and row["attack_type"] in ("none", "live", ""):
                    raise DatasetError(f"line {line}: spoof samples need an attack_type such as print or screen")
                if row["label"] == "live" and row.get("presentation_device_id") not in (None, ""):
                    raise DatasetError(f"line {line}: live samples should have null presentation_device_id")
        if task == "whisper_asr":
            sr, channels, duration = _audio_metadata(data_file, line) if check_files else (row["sample_rate"], row["channels"], row["duration_sec"])
            for key, actual in (("sample_rate", sr), ("channels", channels)):
                if isinstance(row[key], bool) or int(row[key]) != actual or actual <= 0:
                    raise DatasetError(f"line {line}: {key} metadata {row[key]!r} does not match file value {actual}")
            stated_duration = _number(row["duration_sec"], "duration_sec", line)
            if stated_duration <= 0 or abs(stated_duration - duration) > max(0.05, 0.02 * duration):
                raise DatasetError(f"line {line}: duration_sec differs from file duration {duration:.3f}s")
            digits = row["otp_digits"]
            if not isinstance(digits, str) or len(digits) != 6 or not digits.isascii() or not digits.isdigit():
                raise DatasetError(f"line {line}: otp_digits must be a six-character string (keep leading zeroes)")
            if not isinstance(row["transcript"], str) or not row["transcript"].strip():
                raise DatasetError(f"line {line}: transcript must be manually verified text")


        if task == "face_recognition":
            if row["identity_pool"] not in ("registered", "unknown"):
                raise DatasetError(f"line {line}: identity_pool must be registered or unknown")
            identity = row["identity_id"]
            prior = identity_pools.setdefault(identity, row["identity_pool"])
            if prior != row["identity_pool"]:
                raise DatasetError(f"line {line}: identity {identity!r} has inconsistent identity_pool")

        split_value = row.get("split")
        if split_value:
            leak_fields = ("video_id", "capture_session_id", "source_id") if task == "antispoof" else ("session_id", "source_id")
            for field in leak_fields:
                group_value = row.get(field)
                if group_value in (None, ""):
                    continue
                token = (field, str(group_value))
                prior_split = seen_split_groups.setdefault(token, split_value)
                if prior_split != split_value:
                    raise DatasetError(f"line {line}: data leakage: {field}={group_value!r} appears in both {prior_split} and {split_value}")
    return {"task": task, "manifest": str(manifest), "samples": len(rows), "split_counts": dict(split_counts), "identities": len(identity_pools)}


def split_manifest(path: Path, task: str, seed: int = 42, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), holdout_field: str | None = None) -> dict[str, list[dict[str, Any]]]:
    if task not in TASKS:
        raise DatasetError(f"unknown task {task!r}")
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8 or any(r == 0 for r in ratios):
        raise DatasetError("split ratios must be three positive values summing to 1")
    rows = read_jsonl(path)
    if any("split" in row for row in rows):
        raise DatasetError("manifest already has split values; remove them before splitting")
    group_fields = {
        "face_detection": ["session_id", "source_id"],
        "face_recognition": ["session_id", "source_id"],
        "whisper_asr": ["session_id", "source_id"],
        "antispoof": ["video_id", "capture_session_id", "source_id"],
    }[task]
    if holdout_field:
        if holdout_field not in {"speaker_id", "presentation_device_id", "device_id", "camera_id", "subject_id"}:
            raise DatasetError("unsupported holdout field")
        if holdout_field not in rows[0]:
            raise DatasetError(f"holdout field {holdout_field!r} is absent")
        missing_holdout = [str(row.get("sample_id", index)) for index, row in enumerate(rows) if row.get(holdout_field) in (None, "")]
        if missing_holdout and holdout_field != "presentation_device_id":
            raise DatasetError(f"holdout field {holdout_field!r} is empty for samples: {', '.join(missing_holdout[:10])}")
        if task == "face_recognition" and holdout_field == "subject_id":
            raise DatasetError("registered identity splitting must hold out sessions; use identity_pool=unknown for unknown subjects")

    # Union rows that must never cross splits: capture/video session and source/augmentation lineage.
    parent = list(range(len(rows)))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    token_owner: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        line = row.pop("_line")
        for field in group_fields:
            value = row.get(field)
            if value in (None, ""):
                continue
            token = (field, str(value))
            if token in token_owner:
                union(i, token_owner[token])
            else:
                token_owner[token] = i
        if holdout_field and row.get(holdout_field) not in (None, ""):
            token = ("holdout:" + holdout_field, str(row[holdout_field]))
            if token in token_owner:
                union(i, token_owner[token])
            else:
                token_owner[token] = i

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    rng = random.Random(seed)
    groups = list(components.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    holdout_values: set[str] = set()
    if holdout_field:
        available_values = sorted({str(row[holdout_field]) for row in rows if row.get(holdout_field) not in (None, "")})
        if len(available_values) < 3:
            raise DatasetError(f"holdout on {holdout_field} needs at least three independent values to reserve unseen test data and retain train/validation")
        rng.shuffle(available_values)
        holdout_count = min(len(available_values) - 2, max(1, round(len(available_values) * ratios[2])))
        holdout_values = set(available_values[:holdout_count])
    names = list(SPLITS)
    total = len(rows)
    targets = {name: total * ratio for name, ratio in zip(names, ratios)}
    counts = {name: 0 for name in names}
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for indices in groups:
        force_test = any(rows[i].get("identity_pool") == "unknown" for i in indices)
        if holdout_field:
            force_test = force_test or any(str(rows[i].get(holdout_field)) in holdout_values for i in indices)
        if force_test:
            split_name = "test"
        else:
            # Greedy weighted allocation keeps the largest deficit first and is seed-stable.
            split_name = max(names, key=lambda name: (targets[name] - counts[name], -names.index(name)))
        for i in indices:
            row = dict(rows[i])
            row["split"] = split_name
            output[split_name].append(row)
        counts[split_name] += len(indices)
    for split_name in names:
        output[split_name].sort(key=lambda row: row["sample_id"])
    for split_name in names:
        if not output[split_name]:
            raise DatasetError(f"cannot create non-empty {split_name} split from {len(groups)} independent groups")
    if task == "face_recognition":
        by_identity: dict[str, set[str]] = defaultdict(set)
        for split_name, group_rows in output.items():
            for row in group_rows:
                if row.get("identity_pool") == "registered":
                    by_identity[row["identity_id"]].add(split_name)
        bad = [identity for identity, values in by_identity.items() if not {"train", "test"} <= values]
        if bad:
            raise DatasetError("registered identities need independent sessions in both train and test; insufficient sessions for: " + ", ".join(sorted(bad)))
    return output


def save_split_bundle(output_dir: Path, task: str, splits: dict[str, list[dict[str, Any]]], seed: int, source_manifest: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DatasetError(f"split output directory must be empty to preserve the locked split: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    files = {}
    for name, rows in splits.items():
        resolved_rows = []
        for original in rows:
            row = dict(original)
            media_key = "audio_path" if task in ("whisper_asr",) else "image_path"
            media_path = Path(row[media_key])
            if not media_path.is_absolute():
                media_path = source_manifest.resolve().parent / media_path
            row[media_key] = str(media_path.resolve())
            resolved_rows.append(row)
        file = output_dir / f"{name}.jsonl"
        write_jsonl(file, resolved_rows)
        files[name] = {"path": file.name, "samples": len(resolved_rows), "sha256": _sha256_file(file), "sample_ids": [row["sample_id"] for row in resolved_rows]}
    bundle = {"task": task, "seed": seed, "source_manifest": str(source_manifest.resolve()), "grouping": "session/video + source lineage; unknown identities forced to test", "files": files}
    path = output_dir / "split-manifest.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
