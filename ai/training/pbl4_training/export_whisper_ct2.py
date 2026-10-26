from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .dataset import DatasetError, read_jsonl, validate_manifest
from .metrics import edit_distance, normalize_otp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a fine-tuned Whisper checkpoint, convert it to CTranslate2, and evaluate the exported model on held-out test audio.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best Transformers checkpoint or PEFT adapter directory")
    parser.add_argument("--base-checkpoint", required=True, help="same Transformers base checkpoint used for training")
    parser.add_argument("--test-manifest", type=Path, required=True, help="locked whisper_asr test.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="new CTranslate2 output directory; must not exist")
    parser.add_argument("--compute-type", choices=("float16", "int8_float16", "int8", "float32"), default="float16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_dir():
            raise DatasetError(f"checkpoint directory not found: {checkpoint}")
        if args.output.exists():
            raise DatasetError(f"output already exists; refusing to overwrite: {args.output}")
        validate_manifest(args.test_manifest, "whisper_asr")
        rows = read_jsonl(args.test_manifest)
        if any(row.get("split") != "test" for row in rows):
            raise DatasetError("export verification manifest must contain only rows explicitly marked split=test")
        try:
            from faster_whisper import WhisperModel
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise DatasetError("install requirements-whisper.txt, including faster-whisper and Transformers") from exc
        converter = shutil.which("ct2-transformers-converter")
        if not converter:
            raise DatasetError("ct2-transformers-converter is missing; install requirements-whisper.txt")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="whisper-merge-", dir=args.output.parent) as temp:
            merged = Path(temp) / "merged-transformers"
            adapter_config = checkpoint / "adapter_config.json"
            if adapter_config.is_file():
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise DatasetError("PEFT adapter provided; install peft to merge it before CTranslate2 conversion") from exc
                base = WhisperForConditionalGeneration.from_pretrained(args.base_checkpoint, local_files_only=args.local_files_only)
                tuned = PeftModel.from_pretrained(base, checkpoint, local_files_only=args.local_files_only)
                tuned = tuned.merge_and_unload(safe_merge=True)
                tuned.save_pretrained(merged, safe_serialization=True)
                del tuned, base
            else:
                model = WhisperForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True)
                model.save_pretrained(merged, safe_serialization=True)
                del model
            processor = WhisperProcessor.from_pretrained(args.base_checkpoint, language="vi", task="transcribe",
                                                         local_files_only=args.local_files_only)
            processor.save_pretrained(merged)
            command = [converter, "--model", str(merged), "--output_dir", str(args.output.resolve()),
                       "--quantization", args.compute_type, "--force"]
            subprocess.run(command, check=True)
        expected = (args.output / "model.bin", args.output / "config.json", args.output / "vocabulary.json")
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise DatasetError("CTranslate2 conversion did not produce required files: " + ", ".join(missing))

        engine = WhisperModel(str(args.output.resolve()), device=args.device,
                              compute_type=args.compute_type if args.device == "cuda" else "int8")
        predictions = []
        word_errors = char_errors = words = chars = exact = 0
        for row in rows:
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = args.test_manifest.resolve().parent / audio_path
            segments, info = engine.transcribe(str(audio_path.resolve()), language="vi", task="transcribe", beam_size=5,
                                               vad_filter=False, condition_on_previous_text=False)
            hypothesis = "".join(segment.text for segment in segments).strip()
            predictions.append({"sample_id": row["sample_id"], "transcript": hypothesis})
            ref_words, hyp_words = row["transcript"].lower().split(), hypothesis.lower().split()
            word_errors += edit_distance(ref_words, hyp_words); words += len(ref_words)
            ref_chars, hyp_chars = list("".join(ref_words)), list("".join(hyp_words))
            char_errors += edit_distance(ref_chars, hyp_chars); chars += len(ref_chars)
            exact += normalize_otp(hypothesis) == row["otp_digits"]
        inference_compute_type = args.compute_type if args.device == "cuda" else "int8"
        metrics = {"task": "whisper_asr", "split": "test", "samples": len(predictions),
                   "wer": word_errors / max(words, 1), "cer": char_errors / max(chars, 1),
                   "otp_exact_accuracy": exact / max(len(predictions), 1), "exported_engine": "faster-whisper/CTranslate2",
                   "model_version": sha256_file(args.output / "model.bin"), "conversion_quantization": args.compute_type,
                   "inference_compute_type": inference_compute_type,
                   "device": args.device, "test_used_for_selection": False}
        (args.output / "predictions-test.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions), encoding="utf-8")
        (args.output / "export-evaluation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        provenance = {"source_checkpoint": str(checkpoint), "source_checkpoint_files": {
                          str(path.relative_to(checkpoint)): sha256_file(path) for path in checkpoint.rglob("*") if path.is_file()},
                      "base_checkpoint": args.base_checkpoint, "test_manifest": str(args.test_manifest.resolve()),
                      "test_manifest_sha256": sha256_file(args.test_manifest), "conversion_command": command,
                      "converted_model_sha256": sha256_file(args.output / "model.bin"), "compute_type": args.compute_type,
                      "exported_unix": time.time(), "evaluation": metrics}
        (args.output / "export.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    except (DatasetError, OSError, ImportError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
