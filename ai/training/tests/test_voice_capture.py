from __future__ import annotations
from io import BytesIO
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from pbl4_training.collect_voice import AudioSession, prompt_for, wav_bytes


class VoiceCaptureTests(unittest.TestCase):
    def pcm(self):
        return b"".join(struct.pack("<h", int(1000 * math.sin(i * .2))) for i in range(8000))

    def test_prompt_preserves_leading_zero(self):
        self.assertEqual(prompt_for("012345"), "không một hai ba bốn năm")
        with self.assertRaises(ValueError):
            prompt_for("12345")

    def test_wave_and_manifest_match_actual_pcm(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            session = AudioSession(Path(temporary), "speaker-001", 0)
            path = session.save(self.pcm(), 16000, "012345", prompt_for("012345"))
            row = json.loads((session.directory / "whisper.jsonl").read_text(encoding="utf-8"))
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getframerate(), row["sample_rate"])
                self.assertEqual(audio.getnchannels(), row["channels"])
                self.assertEqual(audio.getnframes() / audio.getframerate(), row["duration_sec"])
                self.assertEqual(audio.readframes(audio.getnframes()), self.pcm())
            self.assertEqual(row["otp_digits"], "012345")
            self.assertEqual((session.directory / row["audio_path"]).resolve(), path.resolve())
            from pbl4_training.dataset import validate_manifest
            self.assertEqual(validate_manifest(session.directory / "whisper.jsonl", "whisper_asr")["samples"], 1)

    def test_wrong_label_short_clip_and_duplicate_prompt_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            session = AudioSession(Path(temporary), "speaker-001", 0)
            with self.assertRaises(ValueError):
                session.save(self.pcm(), 16000, "012345", prompt_for("012346"))
            with self.assertRaises(ValueError):
                session.save(b"\0\0", 16000, "012345", prompt_for("012345"))
            session.save(self.pcm(), 16000, "012345", prompt_for("012345"))
            with self.assertRaises(ValueError):
                session.save(self.pcm(), 16000, "012345", prompt_for("012345"))
            with self.assertRaises(ValueError):
                session.save(self.pcm(), 16000, "012346", prompt_for("012346"))
            self.assertEqual(len(session.rows), 1)


if __name__ == "__main__":
    unittest.main()
