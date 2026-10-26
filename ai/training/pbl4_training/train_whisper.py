from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from .dataset import DatasetError, read_jsonl, validate_manifest
from .metrics import edit_distance, normalize_otp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def require_transformers_checkpoint(checkpoint: str, local_only: bool) -> None:
    path = Path(checkpoint)
    if path.is_dir():
        candidates = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
        if not any((path / item).is_file() for item in candidates):
            raise DatasetError(f"{checkpoint} is not a Transformers PyTorch checkpoint; the project's faster-whisper model.bin is CTranslate2 inference format and cannot be fine-tuned here")
    elif local_only:
        raise DatasetError("--local-files-only requires a local Transformers model directory with safetensors or PyTorch weights")


def load_audio(path: Path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(rate) // divisor).astype("float32")
    if not len(audio):
        raise DatasetError(f"empty decoded audio: {path}")
    return np.asarray(audio, dtype="float32")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune multilingual Whisper for Vietnamese spoken OTP using a real labeled manifest.")
    parser.add_argument("--splits", type=Path, required=True, help="directory with train.jsonl, validation.jsonl and test.jsonl")
    parser.add_argument("--checkpoint", default="openai/whisper-base", help="Transformers checkpoint id or local PyTorch checkpoint folder; never use faster-whisper model.bin")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank; set 0 for full fine-tuning (requires substantially more VRAM)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-audio-seconds", type=float, default=20)
    parser.add_argument("--speed-factors", nargs="+", type=float, default=[0.85, 0.9, 1.0, 1.1, 1.15],
                        help="train-only speech-rate perturbation factors; include 1.0, range 0.8–1.25")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 evaluates every item")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.lr <= 0:
        parser.error("epochs, batch size, grad accumulation and learning rate must be positive")
    if (not args.speed_factors or 1.0 not in args.speed_factors
            or any(not 0.8 <= factor <= 1.25 for factor in args.speed_factors)):
        parser.error("speed factors must include 1.0 and stay between 0.8 and 1.25")
    try:
        train_path = args.splits / "train.jsonl"
        val_path = args.splits / "validation.jsonl"
        test_path = args.splits / "test.jsonl"
        for path in (train_path, val_path, test_path):
            if not path.is_file():
                raise DatasetError(f"split file not found: {path}")
            validate_manifest(path, "whisper_asr")
        expected_state = (args.run_dir / "last-training-state.pt").resolve()
        if args.resume:
            if args.resume.resolve() != expected_state or not expected_state.is_file():
                raise DatasetError("resume must point to this run directory's last-training-state.pt")
        elif args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise DatasetError(f"run directory is not empty; choose a new directory or resume its checkpoint: {args.run_dir}")
        require_transformers_checkpoint(args.checkpoint, args.local_files_only)
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        if not torch.cuda.is_available():
            raise DatasetError("CUDA training was requested by this pipeline but PyTorch reports no CUDA device. Install the matching CUDA PyTorch build or add an explicit CPU mode.")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        def load_rows(path: Path) -> list[dict[str, Any]]:
            rows = read_jsonl(path)
            if args.max_audio_seconds:
                rows = [r for r in rows if float(r["duration_sec"]) <= args.max_audio_seconds]
            if not rows:
                raise DatasetError(f"no samples left in {path} after the duration filter")
            return rows[:args.max_eval_samples] if args.max_eval_samples else rows

        train_rows = load_rows(train_path)
        val_rows = load_rows(val_path)
        test_rows = load_rows(test_path)
        processor = WhisperProcessor.from_pretrained(args.checkpoint, language="vi", task="transcribe", local_files_only=args.local_files_only)
        model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=args.local_files_only)
        model.config.use_cache = False
        model.generation_config.language = "vi"
        model.generation_config.task = "transcribe"
        if args.lora_rank > 0:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise DatasetError("LoRA requested; install this task's requirements-whisper.txt (peft)") from exc
            lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
                              target_modules=["q_proj", "v_proj"], task_type=TaskType.SEQ_2_SEQ_LM,
                              bias="none")
            model = get_peft_model(model, lora)
            model.print_trainable_parameters()
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise DatasetError("no trainable parameters; refusing to start a false fine-tune")

        class AudioDataset(Dataset):
            def __init__(self, rows, augment_speed=False):
                self.rows, self.augment_speed = rows, augment_speed
            def __len__(self): return len(self.rows)
            def __getitem__(self, index):
                row = self.rows[index]
                raw = row["audio_path"]
                path = Path(raw)
                if not path.is_absolute(): path = train_path.parent.parent / path
                waveform = load_audio(path)
                if self.augment_speed:
                    factor = random.choice(args.speed_factors)
                    if factor != 1.0:
                        from scipy.signal import resample_poly
                        ratio = Fraction(str(factor)).limit_denominator(100)
                        waveform = resample_poly(waveform, ratio.denominator, ratio.numerator).astype("float32")
                features = processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").input_features[0]
                tokens = processor.tokenizer(row["transcript"], add_special_tokens=True).input_ids
                return {"input_features": features, "labels": torch.tensor(tokens, dtype=torch.long), "sample_id": row["sample_id"]}

        def collate(items):
            features = torch.stack([item["input_features"] for item in items])
            labels = [item["labels"] for item in items]
            padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
            padded[padded == processor.tokenizer.pad_token_id] = -100
            if padded.shape[1] and torch.all(padded[:, 0] == processor.tokenizer.bos_token_id):
                padded = padded[:, 1:]
            return features, padded

        train_loader = DataLoader(AudioDataset(train_rows, augment_speed=True), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        val_loader = DataLoader(AudioDataset(val_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
        test_loader = DataLoader(AudioDataset(test_rows), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
        start_epoch, best_val = 1, float("inf")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state"], strict=True)
            optimizer.load_state_dict(state["optimizer_state"])
            start_epoch = int(state["epoch"]) + 1
            best_val = float(state["best_val_loss"])
            torch.set_rng_state(state["torch_rng_state"])
            random.setstate(state["python_rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            if torch.cuda.is_available() and "cuda_rng_state_all" in state:
                torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

        initial = [p.detach().float().clone() for p in trainable]
        fp16 = torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
        log_path = args.run_dir / "metrics.jsonl"
        run_meta = {"task": "whisper_asr", "checkpoint": args.checkpoint, "dataset_splits": str(args.splits.resolve()),
                    "split_file_sha256": {path.name: sha256(path) for path in (train_path, val_path, test_path)},
                    "training_script_sha256": sha256(Path(__file__).resolve()),
                    "seed": args.seed, "trainable_parameters": sum(p.numel() for p in trainable),
                    "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
                    "lora_rank": args.lora_rank, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "train_only_speed_factors": args.speed_factors,
                    "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(), "python": platform.python_version(),
                    "source_commit": git_revision(), "model_revision": getattr(model.config, "_commit_hash", None),
                    "started_unix": time.time()}
        (args.run_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

        def loss_epoch(loader, training: bool) -> float:
            model.train(training)
            total = 0.0
            n = 0
            if training: optimizer.zero_grad(set_to_none=True)
            for step, (features, labels) in enumerate(loader, 1):
                features, labels = features.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                with torch.set_grad_enabled(training), torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    loss = model(input_features=features, labels=labels).loss
                if not torch.isfinite(loss):
                    raise DatasetError("non-finite Whisper loss; stopping without promoting a checkpoint")
                if training:
                    group_start = ((step - 1) // args.grad_accum) * args.grad_accum
                    group_size = min(args.grad_accum, len(loader) - group_start)
                    scaler.scale(loss / group_size).backward()
                    if step % args.grad_accum == 0 or step == len(loader):
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        if not torch.isfinite(grad_norm):
                            raise DatasetError("non-finite trainable gradient; stopping")
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach().cpu())
                n += 1
            if n == 0:
                raise DatasetError("empty dataloader")
            return total / n

        if start_epoch == 1:
            baseline = {"train_loss": loss_epoch(train_loader, False), "validation_loss": loss_epoch(val_loader, False)}
            (args.run_dir / "baseline-loss.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
            baseline_metrics = generate_metrics(model, test_loader, test_rows, processor)
            baseline_metrics.update({"task": "whisper_asr", "split": "test", "checkpoint": args.checkpoint,
                                     "stage": "pre_finetune_baseline", "test_used_for_selection": False})
            (args.run_dir / "baseline-test-metrics.json").write_text(json.dumps(baseline_metrics, indent=2) + "\n", encoding="utf-8")

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = loss_epoch(train_loader, True)
            val_loss = loss_epoch(val_loader, False)
            record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss, "selected_by": "minimum validation loss"}
            with log_path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(record) + "\n")
            state = {"epoch": epoch, "best_val_loss": min(best_val, val_loss), "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_state_all": torch.cuda.get_rng_state_all(), "python_rng_state": random.getstate(),
                     "numpy_rng_state": np.random.get_state(), "config": run_meta}
            torch.save(state, args.run_dir / "last-training-state.pt")
            if val_loss < best_val:
                best_val = val_loss
                if args.lora_rank > 0:
                    model.save_pretrained(args.run_dir / "best-adapter")
                else:
                    model.save_pretrained(args.run_dir / "best-transformers")
                processor.save_pretrained(args.run_dir / "best-transformers")
                (args.run_dir / "best-metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record))
        deltas = [float((p.detach().float() - before).norm().cpu()) for p, before in zip(trainable, initial)]
        changed = sum(delta > 0 for delta in deltas)
        if changed == 0:
            raise DatasetError("no trainable parameter changed; this run does not count as fine-tuning")
        run_meta["changed_trainable_tensor_count"] = changed
        run_meta["trainable_delta_l2_sum"] = sum(deltas)
        run_meta["completed_unix"] = time.time()
        (args.run_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

        # Reload validation-selected weights before the final held-out test. Test data never selects the checkpoint.
        if args.lora_rank > 0:
            from peft import PeftModel
            base = WhisperForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=args.local_files_only).cuda()
            model = PeftModel.from_pretrained(base, args.run_dir / "best-adapter").cuda()
        else:
            model = WhisperForConditionalGeneration.from_pretrained(args.run_dir / "best-transformers").cuda()
        model.eval()
        final_metrics = generate_metrics(model, test_loader, test_rows, processor)
        final_metrics.update({"task": "whisper_asr", "split": "test", "checkpoint_selected_by": "validation_loss",
                              "test_used_for_selection": False})
        (args.run_dir / "test-metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
        selected_dir = args.run_dir / ("best-adapter" if args.lora_rank > 0 else "best-transformers")
        run_meta["selected_checkpoint"] = str(selected_dir.resolve())
        run_meta["selected_checkpoint_sha256"] = {str(path.relative_to(selected_dir)): sha256(path)
                                                    for path in selected_dir.rglob("*") if path.is_file()}
        run_meta["test_manifest_sha256"] = sha256(test_path)
        run_meta["final_test_metrics"] = final_metrics
        (args.run_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(final_metrics, ensure_ascii=False))
        return 0
    except (DatasetError, OSError, ImportError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def generate_metrics(model, loader, rows, processor) -> dict[str, Any]:
    import torch
    model.eval()
    hypotheses = []
    forced_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    for features, _ in loader:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            generated = model.generate(features.cuda(), forced_decoder_ids=forced_ids, max_new_tokens=32)
        hypotheses.append(processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
    refs = rows[:len(hypotheses)]
    we = ce = wc = cc = exact = 0
    for ref, hyp in zip(refs, hypotheses):
        rwords, hwords = ref["transcript"].lower().split(), hyp.lower().split()
        we += edit_distance(rwords, hwords); wc += len(rwords)
        rc, hc = list("".join(rwords)), list("".join(hwords))
        ce += edit_distance(rc, hc); cc += len(rc)
        exact += normalize_otp(hyp) == ref["otp_digits"]
    return {"samples": len(hypotheses), "wer": we / max(wc, 1), "cer": ce / max(cc, 1),
            "otp_exact_accuracy": exact / max(len(hypotheses), 1)}


if __name__ == "__main__":
    raise SystemExit(main())
