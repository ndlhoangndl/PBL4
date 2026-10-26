from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import DatasetError, read_jsonl, validate_manifest


def convert(source: Path, output: Path) -> int:
    import pandas as pd
    validate_manifest(source, "silero_vad")
    rows = read_jsonl(source)
    data = []
    for row in rows:
        relative = Path(row["audio_path"])
        audio_path = relative if relative.is_absolute() else source.resolve().parent / relative
        data.append({"sample_id": row["sample_id"], "audio_path": str(audio_path.resolve()),
                     "speaker_id": row["speaker_id"], "session_id": row["session_id"],
                     "speech_ts": [{"start": float(seg["start_sec"]), "end": float(seg["end_sec"])} for seg in row["speech_segments"]]})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_feather(output)
    return len(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PBL4 VAD JSONL splits to the official Silero tuning .feather schema.")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = {name: convert(args.splits / f"{name}.jsonl", args.output / f"{name}.feather") for name in ("train", "validation", "test")}
        print(json.dumps({"samples": result, "output": str(args.output.resolve()), "format": "official Silero tuning feather: audio_path,speech_ts"}, indent=2))
        return 0
    except (DatasetError, OSError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
