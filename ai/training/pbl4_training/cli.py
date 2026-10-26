from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import DatasetError, save_split_bundle, split_manifest, validate_manifest
from .metrics import (evaluate_antispoof, evaluate_asr, evaluate_detection,
                      evaluate_face_pairs, write_report)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pbl4-train", description="PBL4 dataset validation, splitting and evaluation tools")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a task-specific JSONL manifest and all referenced files")
    validate.add_argument("task", choices=("face_detection", "face_recognition", "whisper_asr", "antispoof"))
    validate.add_argument("manifest", type=Path)
    split = commands.add_parser("split", help="create reproducible group-aware train/validation/test files")
    split.add_argument("task", choices=("face_detection", "face_recognition", "whisper_asr", "antispoof"))
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--holdout-field", choices=("speaker_id", "presentation_device_id", "device_id", "camera_id", "subject_id"))
    evaluate = commands.add_parser("evaluate", help="compute task metrics from predictions on a held-out manifest")
    evaluate.add_argument("task", choices=("face_detection", "face_recognition", "whisper_asr", "antispoof"))
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--iou", type=float, default=0.5)
    evaluate.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_manifest(args.manifest, args.task)
        elif args.command == "split":
            if args.manifest.resolve() == args.output.resolve():
                raise DatasetError("output must be a new directory, separate from source manifest")
            splits = split_manifest(args.manifest, args.task, seed=args.seed, holdout_field=args.holdout_field)
            bundle = save_split_bundle(args.output, args.task, splits, args.seed, args.manifest)
            result = {"task": args.task, "seed": args.seed, "counts": {k: len(v) for k, v in splits.items()}, "split_manifest": str(bundle)}
        else:
            if args.task == "whisper_asr":
                result = evaluate_asr(args.manifest, args.predictions)
            elif args.task == "antispoof":
                result = evaluate_antispoof(args.manifest, args.predictions, args.threshold)
            elif args.task == "face_recognition":
                result = evaluate_face_pairs(args.predictions, args.threshold)
            else:
                result = evaluate_detection(args.manifest, args.predictions, args.iou)
            result["evaluation_manifest"] = str(args.manifest.resolve())
            result["prediction_file"] = str(args.predictions.resolve())
            write_report(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
