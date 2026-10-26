# Dataset manifest schema

All manifests are UTF-8 JSON Lines: one labeled sample per line. Paths are relative to the manifest file unless absolute. Use opaque IDs instead of names, keep consent/source and license records outside the image/audio payload, and never commit biometric media. `source_id` links every augmentation to its original sample; augmented copies inherit the original split and must only be generated for train.

Copy the appropriate `manifests/*.jsonl.example` and replace all example values. The validator decodes each media file, checks sample IDs, duplicate file hashes, labels, geometry, duration, OTP strings and interval order, then checks that sessions/videos/augmentation lineages stay in one split.

For face detection, a positive image row has `subject_id`, `bbox_xywh`, and five `landmarks` in pixel coordinates in the original image. Landmark order is left eye, right eye, nose tip, left mouth corner, right mouth corner. Each landmark is `[x,y,visibility]`; `2` means visible, `1` means occluded but labeled, and `0` means unavailable (coordinates are ignored). Include background/negative images with `no_face:true`, null `subject_id`, and no box or landmarks; this is needed to measure false detections. The current schema scores one annotated face per positive image, so use one face per image for this pipeline. The project-defined `DoorFaceDetector` trainer learns the box and all five landmarks from these annotations. It is a custom research model and is not an OpenCV `FaceDetectorYN` ONNX replacement.

For recognition, `identity_id` is a stable pseudonymous label and `identity_pool` is `registered` or `unknown`. The project target is four registered identities. Each consenting identity needs multiple separate capture sessions, with the same four labels present in the shared train/validation/test manifests. Unknown identities belong only in evaluation. Four people support a narrow project pilot, not a claim of general face recognition.

The custom SmartDoor recognition and anti-spoof trainers additionally require a `face_region` object: `policy: "reviewed-face-skin-v1"`, `reviewed: true`, `polygon_xy` (3–80 ordered pixel-coordinate points following visible forehead/hairline, cheeks and chin), `eyes_xy` (two eye centers), and `source_image_sha256` (hash of the original image bytes). Use `Review-Face-Regions.cmd` to review/edit these outlines. Points must be inside the image, eyes inside the polygon, and the source hash must match. Pixels outside the outline are replaced by neutral gray before eye leveling/resampling; background and hair outside that reviewed outline cannot contribute to the input. Hair inside an incorrect outline is NOT automatically removed. Detector manifests still refer to original full frames and box/landmark labels. Generic manifest validation supports the original schema; the custom trainer enforces the additional face-region requirement.

For audio, `transcript` is the manually checked utterance and `otp_digits` is a separate six-character string, preserving leading zeroes. Do not place the target OTP in model prompts. `sample_rate`, `channels`, and `duration_sec` are cross-checked with the audio file. For the OTP recognizer, label the Vietnamese spoken words in `transcript` and the exact six-digit string in `otp_digits`; verify each recording by listening.

For anti-spoof, images are camera frames. `label` is `live` or `spoof`; spoof records also need an attack type such as `print`, `screen_photo`, `screen_video`, `mask`, or another documented type. `presentation_device_id` is null for live samples. Frames from a video, capture session, or source augmentation stay together. The project-defined `DoorAntiSpoof` trainer uses the binary `live`/`spoof` labels above and records attack type for separate evaluation.

## Commands

From `SmartDoor/ai/training`:

```powershell
python -m pip install -r requirements-data.txt
python -m pbl4_training.cli validate whisper_asr ..\datasets\whisper.jsonl
python -m pbl4_training.cli split whisper_asr ..\datasets\whisper.jsonl --output ..\experiments\splits\whisper --seed 42
```

The splitter writes fixed `train.jsonl`, `validation.jsonl`, `test.jsonl` and a `split-manifest.json` containing sample IDs and SHA256 values. It fails if independent groups are insufficient to create all three splits. Review each generated split before training.
