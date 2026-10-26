"""Train-only Whisper pilot for a human-confirmed single capture session.

Fine-tunes the decoder of a pretrained multilingual Whisper; the audio encoder
stays frozen and is cached. All metrics here reuse TRAINING recordings, never a
held-out accuracy claim. Normal multi-session training remains train_whisper.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time
import unicodedata

from .dataset import DatasetError, read_jsonl, validate_manifest, write_jsonl
from .metrics import DIGIT_WORDS, edit_distance, write_report
from .train_whisper import load_audio, require_transformers_checkpoint, sha256


def strict_digits(text):
    tokens = re.sub(r"[,.;:!?]", " ", unicodedata.normalize("NFC", text).lower()).split()
    result = ""
    for token in tokens:
        if token in DIGIT_WORDS:
            result += DIGIT_WORDS[token]
        elif re.fullmatch(r"[0-9]+", token):
            result += token
        else:
            return None
    return result if re.fullmatch(r"[0-9]{6}", result) else None


def prepare_labels(ids, start_id):
    """Remove decoder start only; keep the EOS token (it is not batch padding)."""
    return ids[1:] if ids and ids[0] == start_id else ids


def score_training_predictions(rows, predictions):
    if len(rows) != len(predictions):
        raise DatasetError("Prediction count does not match training recordings")
    word_errors = char_errors = words = chars = exact = 0
    for row, prediction in zip(rows, predictions):
        truth = row["transcript"].lower().split()
        guessed = prediction["transcript"].lower().split()
        word_errors += edit_distance(truth, guessed)
        words += len(truth)
        ref_chars, hyp_chars = list("".join(truth)), list("".join(guessed))
        char_errors += edit_distance(ref_chars, hyp_chars)
        chars += len(ref_chars)
        exact += strict_digits(prediction["transcript"]) == row["otp_digits"]
    return {"split": "train_reuse", "samples": len(rows), "exact_otp_count": exact,
        "otp_exact_accuracy_on_training_recordings": exact / len(rows),
        "wer_on_training_recordings": word_errors / max(words, 1),
        "cer_on_training_recordings": char_errors / max(chars, 1),
        "held_out_evaluation": False,
        "note": "WER/CER depend on numeric-vs-word spelling; these reused recordings do not measure generalization"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels-confirmed", action="store_true", help="The operator listened and confirmed every transcript")
    args = parser.parse_args(argv)
    if not args.labels_confirmed:
        parser.error("Human label confirmation required; do not rewrite labels from baseline ASR")
    if args.epochs < 1 or args.grad_accum < 1 or not 0 < args.learning_rate < .01:
        parser.error("Use positive epochs/grad-accum and learning-rate in (0,.01)")
    if args.run_dir.exists():
        parser.error("Use a new output directory; evidence cannot be overwritten")
    require_transformers_checkpoint(str(args.checkpoint), True)
    validate_manifest(args.manifest, "whisper_asr")
    rows = read_jsonl(args.manifest)
    for row in rows:
        path = Path(row["audio_path"])
        row["audio_path"] = str((path if path.is_absolute() else args.manifest.parent / path).resolve())
        if row.get("sha256") and sha256(Path(row["audio_path"])) != row["sha256"]:
            raise DatasetError(f"{row['sample_id']}: captured audio hash changed")
        if strict_digits(row["transcript"]) != row["otp_digits"]:
            raise DatasetError(f"{row['sample_id']}: transcript and six-digit label disagree")
    provenance_path = args.checkpoint / "download-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
    if provenance:
        for name, expected in provenance["files_sha256"].items():
            if sha256(args.checkpoint / name) != expected:
                raise DatasetError(f"Base checkpoint file hash changed: {name}")
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.modeling_outputs import BaseModelOutput
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    print("Loading local pretrained Whisper; encoder frozen, decoder fine-tuning on CPU", flush=True)
    processor = WhisperProcessor.from_pretrained(str(args.checkpoint), language="vi", task="transcribe", local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(args.checkpoint), local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad_(False)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    initial = {name: parameter.detach().clone() for name, parameter in trainable}
    def encoder_hash():
        digest = hashlib.sha256()
        for parameter in model.model.encoder.parameters():
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    frozen_hash = encoder_hash()
    model.eval()
    cached = []
    for i, row in enumerate(rows, 1):
        waveform = load_audio(Path(row["audio_path"]))
        features = processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").input_features
        ids = prepare_labels(processor.tokenizer(row["transcript"]).input_ids, model.config.decoder_start_token_id)
        with torch.no_grad():
            encoded = model.model.encoder(features).last_hidden_state.detach()
        cached.append((features, encoded, torch.tensor([ids], dtype=torch.long)))
        print(f"Prepared {i}/{len(rows)} recordings", flush=True)
    args.run_dir.mkdir(parents=True)
    write_jsonl(args.run_dir / "train-manifest.jsonl", rows)
    metadata = {"task": "whisper_asr", "experiment": "single-session train-only pilot", "train_only": True,
        "base_checkpoint": str(args.checkpoint.resolve()), "base_provenance": provenance,
        "train_scope": "decoder only; encoder frozen and cached", "encoder_sha256_before": frozen_hash,
        "device": "cpu", "seed": args.seed, "epochs": args.epochs, "learning_rate": args.learning_rate,
        "gradient_accumulation": args.grad_accum, "batch_size": 1,
        "samples": len(rows), "total_audio_seconds": sum(r["duration_sec"] for r in rows),
        "speakers": sorted({r["speaker_id"] for r in rows}), "sessions": sorted({r["session_id"] for r in rows}),
        "validation_samples": 0, "test_samples": 0, "checkpoint_selection": "final configured epoch; no validation selection",
        "human_labels_confirmed": True, "torch": str(torch.__version__),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "trainable_parameter_names": [name for name, _ in trainable],
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "manifest_sha256": sha256(args.run_dir / "train-manifest.jsonl"),
        "audio_sha256": {r["sample_id"]: sha256(Path(r["audio_path"])) for r in rows},
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("train_whisper_pilot.py", "train_whisper.py", "metrics.py")},
        "started_unix": time.time(), "deployed": False,
        "limitations": ["all evaluation reuses the training recordings", "one speaker/session; no generalization claim",
                        "decoder adaptation of pretrained Whisper, not a new speech architecture", "no speaker identity verification"]}
    write_report(metadata, args.run_dir / "run.json")

    def loss_for(item):
        _, encoded, labels = item
        return model(encoder_outputs=BaseModelOutput(last_hidden_state=encoded), labels=labels, use_cache=False).loss

    def measure_loss():
        model.eval()
        with torch.no_grad():
            return sum(float(loss_for(item)) for item in cached) / len(cached)

    def generate(stage):
        model.eval()
        predictions = []
        for i, ((features, _, _), row) in enumerate(zip(cached, rows), 1):
            with torch.no_grad():
                ids = model.generate(features, language="vi", task="transcribe", max_new_tokens=32,
                                     num_beams=1, do_sample=False, condition_on_prev_tokens=False)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            predictions.append({"sample_id": row["sample_id"], "transcript": text, "digits": strict_digits(text)})
            print(f"{stage}: decoded {i}/{len(rows)}", flush=True)
        write_jsonl(args.run_dir / f"{stage}-train-predictions.jsonl", predictions)
        result = score_training_predictions(rows, predictions)
        write_report(result, args.run_dir / f"{stage}-train-metrics.json")
        return result

    baseline_loss = measure_loss()
    baseline_metrics = generate("baseline")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.learning_rate)
    updates = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.model.encoder.eval()
        order = list(range(len(cached)))
        random.shuffle(order)
        total = 0.
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(order, 1):
            loss = loss_for(cached[index])
            if not torch.isfinite(loss):
                raise DatasetError("Non-finite training loss")
            group_start = ((step - 1) // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(order) - group_start)
            (loss / group_size).backward()
            total += float(loss.detach())
            if step % args.grad_accum == 0 or step == len(order):
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.)
                if not torch.isfinite(norm):
                    raise DatasetError("Non-finite training gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                print(f"epoch={epoch} sample={step}/{len(order)} optimizer_updates={updates}", flush=True)
        record = {"epoch": epoch, "training_loss": total / len(cached),
                  "training_eval_loss": measure_loss(), "optimizer_updates": updates,
                  "validation_loss": None, "test_loss": None}
        with (args.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        model.save_pretrained(args.run_dir / "final-model", safe_serialization=True)
        processor.save_pretrained(args.run_dir / "final-model")
        torch.save({"epoch": epoch, "optimizer": optimizer.state_dict(), "config": metadata}, args.run_dir / "optimizer-state.pt")
        print(json.dumps(record), flush=True)
    changed = {name: float((parameter.detach() - initial[name]).norm()) for name, parameter in trainable}
    if not any(value > 0 for value in changed.values()) or encoder_hash() != frozen_hash:
        raise DatasetError("Decoder failed to change, or frozen encoder changed unexpectedly")
    final_metrics = generate("finetuned")
    metadata.update(completed_unix=time.time(), optimizer_updates=updates, baseline_training_loss=baseline_loss,
        final_training_loss=measure_loss(), changed_trainable_tensor_count=sum(v > 0 for v in changed.values()),
        trainable_delta_l2=changed, encoder_sha256_after=encoder_hash(),
        baseline_on_training_recordings=baseline_metrics, finetuned_on_training_recordings=final_metrics,
        output_sha256={str(p.relative_to(args.run_dir / "final-model")): sha256(p) for p in (args.run_dir / "final-model").rglob("*") if p.is_file()})
    write_report(metadata, args.run_dir / "run.json")
    write_report({"train_only": True, "baseline": baseline_metrics, "finetuned": final_metrics,
                  "baseline_training_loss": baseline_loss, "final_training_loss": metadata["final_training_loss"],
                  "optimizer_updates": updates, "changed_trainable_tensors": metadata["changed_trainable_tensor_count"],
                  "held_out_metrics": None}, args.run_dir / "summary.json")
    print(f"Fine-tuning finished: {args.run_dir / 'final-model'}; all metrics are TRAIN-SET ONLY", flush=True)


if __name__ == "__main__":
    main()
