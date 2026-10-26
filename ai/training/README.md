# SmartDoor AI training

## Active model scope

SmartDoor currently has four model tasks: three project-defined face CNNs and one Vietnamese OTP speech recognizer. The optional standalone VAD-training pipeline and the single-person face reconstruction experiment are archived under `archive/out-of-scope/`. Runtime faster-whisper may use its existing built-in VAD filtering; no separate VAD model is needed for the current requirement.

| Task | Model and method | Data/status |
|---|---|---|
| Face detection + five landmarks | `DoorFaceDetector`, trained from project-initialized weights with face presence BCE and box/landmark SmoothL1 losses | Code exists; needs manually checked boxes and five landmark points plus background images. |
| Face identity | `DoorFaceEmbedder`, 128-D normalized embedding with ArcFace training head | Code exists; needs reviewed face-only crops for all four registered people in one shared training manifest, collected across independent sessions. |
| Anti-spoof | `DoorAntiSpoof`, binary live/spoof classifier | Code exists; needs live samples and print/screen attack samples recorded across sessions/devices. |
| Vietnamese OTP speech | Whisper Tiny pretrained checkpoint with project decoder fine-tuning; training-only speed perturbation covers 0.85×–1.15× | 20 confirmed samples from one speaker/session; 5-epoch pilot exists and read 10/20 training samples exactly on the training set. The multi-session trainer supports speed perturbation on train audio only; collect real slow/normal/fast speech and evaluate on held-out speakers. Runtime app still uses Whisper Base. |

The 60 existing face captures belong to one person across two sessions. A separate reconstruction CNN was trained from them and is preserved in the archive; it does not recognize identity. Collect and review the four people separately, then train one shared embedder with all four identity labels in the same run. Training one identity from random initialization at a time would overwrite or distort the previous identity representation.

## Runtime and safety

The app currently loads YuNet, SFace, and Whisper Base from `ai/models`. Those files remain because the project-defined face models are not trained/validated/integrated yet and Whisper Tiny has not replaced the app recognizer. The new face models in `pbl4_training/door_models.py` are research baselines and are not connected to unlocking.

## Collect and review data

- Face capture: use the ESP32 stream or a USB webcam from `scripts/Collect-Faces.cmd`; the person-specific launchers are `scripts/Collect-Faces-Person-002.cmd`, `scripts/Collect-Faces-Person-003.cmd`, and `scripts/Collect-Faces-Person-004.cmd`. Each session guides 50 photos (20 front, 10 left, 10 right, 5 down, 5 up). Confirm the displayed identity code before starting; update the ESP32 stream URL when its DHCP address changes. The preview reader discards stale frames to reduce display delay. For ESP32 lag, try SVGA 800×600 and JPEG Quality 12. Follow [FACE_CAPTURE.md](FACE_CAPTURE.md).
- Audio: run `scripts/Collect-Voice.cmd`; follow [VOICE_CAPTURE.md](VOICE_CAPTURE.md). Audio labels must be checked by listening; collect multiple speakers reading at slow, normal and fast rates.
- Face identity and anti-spoof data require reviewed forehead-to-chin regions. Keep raw data local and out of Git.

## Train and evaluate

- Face models: [DOOR_VISION.md](DOOR_VISION.md), trainer `pbl4_training/train_door_vision.py`. Split by capture session/source and keep held-out data separate.
- OTP speech pilot: [WHISPER_PILOT_DEMO.md](WHISPER_PILOT_DEMO.md), trainer `pbl4_training/train_whisper_pilot.py`.
- Multi-session Whisper experiments: `pbl4_training/train_whisper.py`; the CTranslate2 export helper is for a later reviewed runtime integration.
- Dataset fields and review rules: [DATASET_SCHEMA.md](DATASET_SCHEMA.md).

