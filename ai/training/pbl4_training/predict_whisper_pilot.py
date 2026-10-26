"""Run the saved fine-tuned model on a local WAV; never grants door access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .train_whisper import load_audio, sha256
from .train_whisper_pilot import strict_digits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    torch.set_num_threads(4)
    processor = WhisperProcessor.from_pretrained(str(args.checkpoint), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(args.checkpoint), local_files_only=True)
    model.eval()
    audio = load_audio(args.audio)
    if not .3 <= len(audio) / 16000 <= 20:
        raise ValueError("Choose a recording between 0.3 and 20 seconds")
    features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        ids = model.generate(features.input_features, attention_mask=features.attention_mask,
            language="vi", task="transcribe", max_new_tokens=32, num_beams=1,
            do_sample=False, condition_on_prev_tokens=False)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    metadata = args.checkpoint.parent / "run.json"
    reused = None
    if metadata.is_file():
        report = json.loads(metadata.read_text(encoding="utf-8"))
        reused = sha256(args.audio) in report.get("audio_sha256", {}).values()
    print(json.dumps({"transcript": text, "digits": strict_digits(text),
        "reused_training_recording": reused, "checkpoint": str(args.checkpoint.resolve()),
        "otp_verified": False, "unlock_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
