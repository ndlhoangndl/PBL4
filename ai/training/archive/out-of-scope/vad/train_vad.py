from __future__ import annotations

import argparse
import json
import hashlib
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from .dataset import DatasetError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune only the official Silero-VAD 16 kHz decoder from a compatible local TorchScript checkpoint.")
    parser.add_argument("--upstream", type=Path, required=True, help="checkout of snakers4/silero-vad at a pinned revision")
    parser.add_argument("--expected-commit", required=True, help="full git commit SHA to ensure the tuning code is pinned")
    parser.add_argument("--data", type=Path, required=True, help="train.feather, validation.feather, test.feather from prepare_vad_feather")
    parser.add_argument("--checkpoint", type=Path, required=True, help="compatible 16 kHz Silero VAD TorchScript checkpoint, not the faster-whisper default VAD")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length-sec", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--noise-loss", type=float, default=0.5)
    parser.add_argument("--augmentation-probability", type=float, default=0.2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.epochs < 1 or args.batch_size < 1 or args.max_length_sec < 1 or args.lr <= 0:
            raise DatasetError("epochs, batch size, max length and learning rate must be positive")
        upstream = args.upstream.resolve()
        tuning_dir = upstream / "tuning"
        if not (tuning_dir / "utils.py").is_file() or not (tuning_dir / "tune.py").is_file():
            raise DatasetError("upstream checkout must contain tuning/utils.py and tuning/tune.py")
        actual_commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        expected = args.expected_commit.lower()
        if len(expected) != 40 or actual_commit.lower() != expected:
            raise DatasetError(f"Silero source commit mismatch; expected full SHA {expected}, found {actual_commit}")
        for split in ("train", "validation", "test"):
            if not (args.data / f"{split}.feather").is_file():
                raise DatasetError(f"missing {split}.feather; run prepare_vad_feather first")
        if not args.checkpoint.is_file():
            raise DatasetError("local compatible Silero TorchScript checkpoint is required; refusing to download or substitute a model silently")
        expected_state = (args.run_dir / "last-training-state.pt").resolve()
        if args.resume:
            if args.resume.resolve() != expected_state or not expected_state.is_file():
                raise DatasetError("resume must point to this run directory's last-training-state.pt")
        elif args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise DatasetError(f"run directory is not empty; choose a new directory or resume its checkpoint: {args.run_dir}")
        if args.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise DatasetError("CUDA requested, but installed PyTorch has no CUDA runtime")
        try:
            import torch
            import torch.nn as nn
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise DatasetError("install the isolated VAD environment using requirements-vad.txt and a CUDA PyTorch build") from exc
        sys.path.insert(0, str(tuning_dir))
        from utils import (SileroVadDataset, SileroVadPadder, VADDecoderRNNJIT,
                          init_jit_model, train as official_train, validate as official_validate)
        from torch.utils.data import DataLoader
        random.seed(args.seed); torch.manual_seed(args.seed)
        if args.device == "cuda": torch.cuda.manual_seed_all(args.seed)
        checkpoint_hash = sha256_file(args.checkpoint)
        data_hashes = {f"{split}.feather": sha256_file(args.data / f"{split}.feather") for split in ("train", "validation", "test")}
        signature = {"upstream_commit": actual_commit, "checkpoint_sha256": checkpoint_hash,
                     "data_sha256": data_hashes, "seed": args.seed, "device": args.device,
                     "epochs": args.epochs, "batch_size": args.batch_size, "max_length_sec": args.max_length_sec,
                     "lr": args.lr, "noise_loss": args.noise_loss, "augmentation_probability": args.augmentation_probability}
        args.run_dir.mkdir(parents=True, exist_ok=True)
        config = OmegaConf.create({
            "jit_model_path": str(args.checkpoint.resolve()), "use_torchhub": False, "tune_8k": False,
            "train_dataset_path": str((args.data / "train.feather").resolve()),
            "val_dataset_path": str((args.data / "validation.feather").resolve()),
            "model_save_path": str((args.run_dir / "best-vad-16k.jit").resolve()),
            "noise_loss": args.noise_loss, "max_train_length_sec": args.max_length_sec,
            "aug_prob": args.augmentation_probability, "learning_rate": args.lr,
            "batch_size": args.batch_size, "num_workers": 0, "num_epochs": args.epochs,
            "device": args.device, "seed": args.seed,
        })
        (args.run_dir / "config.yml").write_text(OmegaConf.to_yaml(config), encoding="utf-8")
        model = init_jit_model(str(args.checkpoint.resolve()), device=torch.device(args.device))
        branch = getattr(model, "_model", None)
        if branch is None or not hasattr(branch, "decoder") or not hasattr(branch, "encoder") or not hasattr(branch, "stft"):
            raise DatasetError("TorchScript checkpoint is not the expected official Silero 16 kHz model layout; no partial state loading was attempted")
        decoder = VADDecoderRNNJIT().to(args.device)
        decoder.load_state_dict(branch.decoder.state_dict(), strict=True)
        for parameter in decoder.parameters(): parameter.requires_grad_(True)
        trainable = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
        if not trainable:
            raise DatasetError("official decoder has no trainable parameters")
        optimizer = torch.optim.Adam(trainable, lr=args.lr)
        train_data = SileroVadDataset(config, mode="train")
        val_data = SileroVadDataset(config, mode="val")
        test_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        test_config.val_dataset_path = str((args.data / "test.feather").resolve())
        test_data = SileroVadDataset(test_config, mode="val")
        if min(len(train_data), len(val_data), len(test_data)) == 0:
            raise DatasetError("train/validation/test VAD datasets must all be non-empty")
        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=SileroVadPadder, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=SileroVadPadder, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=SileroVadPadder, num_workers=0)
        criterion = nn.BCELoss(reduction="none")
        start_epoch, best_auc = 1, -1.0
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            if state.get("resume_signature") != signature:
                raise DatasetError("resume signature differs (checkpoint/data hashes, source revision, seed, device, or hyperparameters); start a new run instead")
            decoder.load_state_dict(state["decoder_state"], strict=True)
            optimizer.load_state_dict(state["optimizer_state"])
            start_epoch, best_auc = int(state["epoch"]) + 1, float(state["best_validation_auc"])
            torch.set_rng_state(state["torch_rng_state"])
            random.setstate(state["python_rng_state"])
            if args.device == "cuda" and "cuda_rng_state_all" in state:
                torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        before = [parameter.detach().clone() for parameter in trainable]
        baseline_path = args.run_dir / "baseline.json"
        if start_epoch == 1:
            baseline_loss, baseline_auc = official_validate(config, val_loader, model, decoder, criterion, args.device)
            baseline_test_loss, baseline_test_auc = official_validate(test_config, test_loader, model, decoder, criterion, args.device)
            baseline = {"validation_loss": baseline_loss, "validation_roc_auc": baseline_auc,
                        "test_loss": baseline_test_loss, "test_roc_auc": baseline_test_auc, "test_used_for_selection": False}
            baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        else:
            if not baseline_path.is_file():
                raise DatasetError("resume run is missing baseline.json; refusing to substitute a resumed checkpoint as the pretrained baseline")
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        log = args.run_dir / "metrics.jsonl"
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = official_train(config, train_loader, model, decoder, criterion, optimizer, args.device)
            val_loss, val_auc = official_validate(config, val_loader, model, decoder, criterion, args.device)
            record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                      "validation_roc_auc": val_auc, "updated_component": "16 kHz decoder only"}
            with log.open("a", encoding="utf-8") as stream: stream.write(json.dumps(record) + "\n")
            state = {"epoch": epoch, "best_validation_auc": max(best_auc, val_auc),
                     "decoder_state": decoder.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
                     "cuda_rng_state_all": torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
                     "config": OmegaConf.to_container(config, resolve=True), "resume_signature": signature}
            torch.save(state, args.run_dir / "last-training-state.pt")
            if val_auc > best_auc:
                best_auc = val_auc
                branch.decoder.load_state_dict(decoder.state_dict(), strict=True)
                torch.jit.save(model, str(args.run_dir / "best-vad-16k.jit"))
                (args.run_dir / "best-metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record))
        deltas = [float((parameter.detach() - initial).norm().cpu()) for parameter, initial in zip(trainable, before)]
        if not any(delta > 0 for delta in deltas):
            raise DatasetError("no trainable decoder parameter changed; this run does not count as fine-tuning")
        best_record = json.loads((args.run_dir / "best-metrics.json").read_text(encoding="utf-8"))
        # Re-evaluate the exact best-validation checkpoint exported as TorchScript on the untouched test set.
        exported = torch.jit.load(str(args.run_dir / "best-vad-16k.jit"), map_location=args.device)
        exported_decoder = VADDecoderRNNJIT().to(args.device)
        exported_decoder.load_state_dict(exported._model.decoder.state_dict(), strict=True)
        test_loss, test_auc = official_validate(test_config, test_loader, exported, exported_decoder, criterion, args.device)
        run = {"task": "silero_vad", "source_url": "https://github.com/snakers4/silero-vad", "source_commit": actual_commit,
               "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
               "training_script_sha256": sha256_file(Path(__file__).resolve()), "data_sha256": data_hashes,
               "seed": args.seed, "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None, "python": platform.python_version(),
               "updated_component": "16 kHz decoder only; official STFT and encoder frozen",
               "trainable_parameter_count": sum(p.numel() for p in trainable), "trainable_delta_l2_sum": sum(deltas),
               "exported_checkpoint": str((args.run_dir / "best-vad-16k.jit").resolve()),
               "exported_checkpoint_sha256": sha256_file(args.run_dir / "best-vad-16k.jit"),
               "baseline": baseline, "selected_validation": best_record,
               "test_after_finetune": {"loss": test_loss, "roc_auc": test_auc, "used_for_selection": False},
               "completed_unix": time.time()}
        (args.run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(run, indent=2))
        return 0
    except (DatasetError, OSError, ImportError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
