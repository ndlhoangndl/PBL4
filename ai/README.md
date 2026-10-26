# SmartDoor AI

`runtime/` provides the local Python API; `models/` contains the assets currently loaded by the application. `datasets/` contains private captures and stays local. `training/` contains collection, review, training and evaluation code.

## Current runtime assets

| Asset | Current task | Status |
|---|---|---|
| YuNet ONNX | Find face and landmarks | Existing pretrained runtime model; custom SmartDoor detector is still under training development. |
| SFace ONNX | Face embeddings | Existing pretrained runtime model; custom SmartDoor embedder needs multiple identities and evaluation. |
| Whisper Base CTranslate2 | Vietnamese speech-to-text | Current app model. A Whisper Tiny decoder fine-tuning pilot is saved separately and has not replaced runtime. |

These existing assets stay in place while project models are collected, trained, evaluated and integrated. The custom anti-spoof model is not yet part of the runtime unlock flow.

## Active training scope

The current project plan has four tasks: face detection with five landmarks, face identity embeddings, live/spoof classification, and Vietnamese OTP transcription. Speech-to-text reads the spoken digit words and maps them to a six-character OTP while preserving leading zeroes; it does not identify the speaker. A separate VAD training task is unnecessary for this requirement. See [training overview](training/README.md), [face capture](training/FACE_CAPTURE.md), and [voice capture](training/VOICE_CAPTURE.md).
