"""Train project-defined vision baselines from labeled, session-separated data.

Default initialization is random; --init-checkpoint fine-tunes only our matching
architecture. OpenCV YuNet/SFace ONNX files are not training checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .dataset import DatasetError, read_jsonl, validate_manifest, write_jsonl
from .door_models import (ArcFaceHead, DoorAntiSpoof, DoorFaceDetector,
                          DoorFaceEmbedder, detection_loss)
from .metrics import evaluate_antispoof, evaluate_detection, write_report
from .face_region import POLICY, prepare_face, validate_region_file

MODELS = {"face_detection": DoorFaceDetector,
          "face_recognition": DoorFaceEmbedder, "antispoof": DoorAntiSpoof}


def load_splits(folder: Path, task: str):
    result = {}
    for split in ("train", "validation", "test"):
        path = folder / f"{split}.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            if row.get("split", split) != split:
                raise DatasetError(f"{path}: incorrect split field")
            image = Path(row["image_path"])
            row["image_path"] = str((image if image.is_absolute() else path.parent / image).resolve())
            row["split"] = split
        result[split] = rows
    # One combined validation catches duplicate bytes and group leaks across files.
    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary) / "combined.jsonl"
        write_jsonl(manifest, sum(result.values(), []))
        validate_manifest(manifest, task)
    for split, rows in result.items():
        if task == "antispoof" and {r["label"] for r in rows} != {"live", "spoof"}:
            raise DatasetError(f"{split}: need both live and spoof images")
        if task == "face_detection" and {r.get("no_face", False) for r in rows} != {False, True}:
            raise DatasetError(f"{split}: need face and background images")
    return result


class VisionDataset(Dataset):
    def __init__(self, rows, task, identities, size):
        self.rows, self.task, self.identities, self.size = rows, task, identities, size
        if task in ("face_recognition", "antispoof"):
            for row in rows:
                try:
                    validate_region_file(Path(row["image_path"]), row.get("face_region"))
                except ValueError as error:
                    raise DatasetError(f"{row['sample_id']}: {error}") from error

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            width, height = image.size
            image = prepare_face(image, row["face_region"], self.size) if self.task in ("face_recognition", "antispoof") else image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        if self.task == "face_detection":
            present = not row.get("no_face", False)
            target = {"presence": torch.tensor(float(present)), "boxes": torch.zeros(4),
                      "landmarks": torch.zeros(5, 2), "visible": torch.zeros(5, dtype=torch.bool)}
            if present:
                x, y, w, h = row["bbox_xywh"]
                target["boxes"] = torch.tensor([x / width, y / height, (x + w) / width, (y + h) / height])
                for p, (px, py, visibility) in enumerate(row["landmarks"]):
                    if visibility > 0:
                        target["landmarks"][p] = torch.tensor([px / width, py / height])
                        target["visible"][p] = True
        elif self.task == "antispoof":
            target = torch.tensor(int(row["label"] == "live"))
        else:
            target = torch.tensor(self.identities.get(row["identity_id"], -1))
        return tensor, target, index


def move_target(target, device):
    return {key: value.to(device) for key, value in target.items()} if isinstance(target, dict) else target.to(device)


def loss_for(model, head, task, images, target):
    output = model(images)
    if task == "face_detection":
        return detection_loss(output, target)
    return F.cross_entropy(head(output, target) if head is not None else output, target)


def epoch_loss(model, head, task, loader, device, optimizer=None):
    model.train(optimizer is not None)
    if head is not None:
        head.train(optimizer is not None)
    total = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, target, _ in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, head, task, images.to(device), move_target(target, device))
            if not torch.isfinite(loss):
                raise DatasetError("Non-finite loss; run stopped")
            if optimizer is not None:
                loss.backward()
                parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
            total += float(loss.detach()) * len(images)
            count += len(images)
    return total / count


@torch.no_grad()
def predict(model, loader, device, task):
    model.eval()
    records = []
    for images, _, indices in loader:
        output = model(images.to(device))
        for j, index in enumerate(indices.tolist()):
            row = loader.dataset.rows[index]
            record = {"sample_id": row["sample_id"]}
            if task == "antispoof":
                record["score_live"] = float(output.softmax(1)[j, 1])
            elif task == "face_recognition":
                record["embedding"] = output[j].cpu().tolist()
            else:
                with Image.open(row["image_path"]) as image:
                    width, height = image.size
                box = output["boxes"][j].cpu().tolist()
                points = output["landmarks"][j].cpu().tolist()
                record["detections"] = []
                if float(output["presence"][j].sigmoid()) >= 0.5:
                    record["detections"].append({"bbox_xywh": [box[0] * width, box[1] * height,
                        (box[2] - box[0]) * width, (box[3] - box[1]) * height],
                        "landmarks": [[x * width, y * height] for x, y in points]})
            records.append(record)
    return records


def select_threshold(positive, negative, lower=0.0):
    if not positive or not negative:
        raise DatasetError("Threshold calibration requires positive and negative validation scores")
    def errors(t):
        far = sum(s >= t for s in negative) / len(negative)
        frr = sum(s < t for s in positive) / len(positive)
        return ((far + frr) / 2, far, -t)
    return float(min(np.linspace(lower, 1, 201), key=errors))


def recognition_scores(records, rows, gallery):
    output = []
    names = sorted(gallery)
    centers = torch.stack([gallery[name] for name in names])
    for record, row in zip(records, rows):
        similarities = centers @ torch.tensor(record["embedding"])
        best = int(similarities.argmax())
        known = row["identity_pool"] == "registered"
        good = float(similarities[names.index(row["identity_id"])]) if known else None
        wrong = [float(s) for name, s in zip(names, similarities) if not known or name != row["identity_id"]]
        output.append({"sample_id": row["sample_id"], "known": known, "genuine_score": good,
                       "impostor_score": max(wrong), "best_score": float(similarities[best]),
                       "predicted_identity": names[best], "identity_id": row["identity_id"]})
    return output


def recognition_report(scores, threshold):
    known = [r for r in scores if r["known"]]
    unknown = [r for r in scores if not r["known"]]
    return {"threshold_cosine": threshold, "known_probes": len(known), "unknown_probes": len(unknown),
            "genuine_false_reject_rate": sum(r["genuine_score"] < threshold for r in known) / len(known) if known else None,
            "wrong_identity_pair_accept_rate": sum(r["impostor_score"] >= threshold for r in scores) / len(scores),
            "correct_identification_and_accept_rate": sum(r["best_score"] >= threshold and r["predicted_identity"] == r["identity_id"] for r in known) / len(known) if known else None,
            "unknown_false_accept_rate": sum(r["best_score"] >= threshold for r in unknown) / len(unknown) if unknown else None}


def evaluate(model, loaders, device, task, run, label):
    predictions = {split: predict(model, loaders[split], device, task) for split in ("validation", "test")}
    report = {}
    if task == "face_recognition":
        train = predict(model, loaders["gallery"], device, task)
        grouped = {}
        for item, row in zip(train, loaders["gallery"].dataset.rows):
            grouped.setdefault(row["identity_id"], []).append(torch.tensor(item["embedding"]))
        gallery = {name: F.normalize(torch.stack(values).mean(0), dim=0) for name, values in grouped.items()}
        predictions = {s: recognition_scores(predictions[s], loaders[s].dataset.rows, gallery) for s in predictions}
        validation = predictions["validation"]
        threshold = select_threshold([r["genuine_score"] for r in validation],
                                     [r["impostor_score"] for r in validation], -1)
        report = {s: recognition_report(values, threshold) for s, values in predictions.items()}
    else:
        threshold = 0.5
        if task == "antispoof":
            rows = loaders["validation"].dataset.rows
            threshold = select_threshold([p["score_live"] for p, r in zip(predictions["validation"], rows) if r["label"] == "live"],
                                         [p["score_live"] for p, r in zip(predictions["validation"], rows) if r["label"] == "spoof"])
        for split, values in predictions.items():
            reference = run / f"{split}.jsonl"
            path = run / f"{label}-{split}-predictions.jsonl"
            write_jsonl(path, values)
            report[split] = evaluate_antispoof(reference, path, threshold) if task == "antispoof" else evaluate_detection(reference, path)
    for split, values in predictions.items():
        write_jsonl(run / f"{label}-{split}-predictions.jsonl", values)
    write_report(report, run / f"{label}-evaluation.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(MODELS))
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--aligned-faces", action="store_true", help="Confirm recognition face-skin polygons and eye centers were reviewed")
    parser.add_argument("--init-checkpoint", type=Path, help="Fine-tune a matching SmartDoor checkpoint; never an ONNX file")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < 1:
        parser.error("epochs/batch-size must be positive; learning-rate must lie in (0,1)")
    if args.task == "face_recognition" and not args.aligned_faces:
        parser.error("Recognition requires --aligned-faces and reviewed face_region polygons/eye centers")
    if args.run_dir.exists():
        parser.error("--run-dir must be new; overwriting training evidence is disabled")
    rows = load_splits(args.splits, args.task)
    identities = {name: i for i, name in enumerate(sorted({r["identity_id"] for r in rows["train"]}))} if args.task == "face_recognition" else {}
    if identities:
        if len(identities) < 2:
            raise DatasetError("Recognition needs at least two training identities, plus held-out sessions")
        for split in rows:
            for row in rows[split]:
                if row["identity_pool"] == "unknown" and split != "test":
                    raise DatasetError("Unknown identities are test-only")
                if row["identity_pool"] == "registered" and row["identity_id"] not in identities:
                    raise DatasetError("Registered identity has no training gallery samples")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    size = 160 if args.task == "face_detection" else 112
    datasets = {s: VisionDataset(values, args.task, identities, size) for s, values in rows.items()}
    loaders = {s: DataLoader(dataset, batch_size=args.batch_size, shuffle=(s == "train"), num_workers=0) for s, dataset in datasets.items()}
    loaders["gallery"] = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=False)
    model = MODELS[args.task]().to(args.device)
    head = ArcFaceHead(len(identities)).to(args.device) if identities else None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=args.device, weights_only=True)
        if checkpoint["task"] != args.task or checkpoint["architecture"] != type(model).__name__ or checkpoint["identities"] != identities:
            raise DatasetError("Initialization checkpoint architecture/task/identity labels do not match")
        if args.task in ("face_recognition", "antispoof") and checkpoint.get("face_region_policy") != POLICY:
            raise DatasetError("Checkpoint has an older/different face-region preprocessing policy")
        model.load_state_dict(checkpoint["model"])
        if head is not None:
            head.load_state_dict(checkpoint["head"])
    parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0001)
    args.run_dir.mkdir(parents=True)
    for split in rows:
        write_jsonl(args.run_dir / f"{split}.jsonl", rows[split])
    metadata = {"task": args.task, "architecture": type(model).__name__, "identities": identities,
                "input_size": size, "normalization": "RGB / 127.5 - 1; face ROI: neutral mask, eye leveling, square letterbox" if args.task in ("face_recognition", "antispoof") else "RGB / 127.5 - 1; bilinear square resize",
                "face_region_policy": POLICY if args.task in ("face_recognition", "antispoof") else None,
                "seed": args.seed, "torch_version": str(torch.__version__), "device": args.device,
                "initialization": str(args.init_checkpoint.resolve()) if args.init_checkpoint else "random; no third-party pretrained weights",
                "parameters": sum(p.numel() for p in parameters),
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("door_models.py", "train_door_vision.py", "face_region.py")},
                "split_sha256": {s: hashlib.sha256((args.run_dir / f"{s}.jsonl").read_bytes()).hexdigest() for s in rows},
                "selection": "minimum validation loss; test evaluated only after checkpoint selection",
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_report(metadata, args.run_dir / "run.json")
    write_report({r["sample_id"]: hashlib.sha256(Path(r["image_path"]).read_bytes()).hexdigest()
                  for values in rows.values() for r in values}, args.run_dir / "media-sha256.json")
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = epoch_loss(model, head, args.task, loaders["train"], args.device, optimizer)
        validation_loss = epoch_loss(model, head, args.task, loaders["validation"], args.device)
        state = {**metadata, "model": model.state_dict(), "head": head.state_dict() if head is not None else None,
                 "optimizer": optimizer.state_dict(), "epoch": epoch, "validation_loss": validation_loss}
        torch.save(state, args.run_dir / "last.pt")
        if validation_loss < best:
            best = validation_loss
            torch.save(state, args.run_dir / "best.pt")
        log = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        with (args.run_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
    model.load_state_dict(torch.load(args.run_dir / "best.pt", map_location=args.device, weights_only=True)["model"])
    changed = sum(not torch.equal(value.cpu(), initial[key]) for key, value in model.state_dict().items())
    if not changed:
        raise DatasetError("No model tensor changed; cannot report successful training")
    final_report = evaluate(model, loaders, args.device, args.task, args.run_dir, "trained")
    model.load_state_dict(initial)
    baseline = evaluate(model, loaders, args.device, args.task, args.run_dir, "initial")
    write_report({"changed_model_tensors": changed, "initial": baseline, "trained": final_report,
                  "checkpoint_sha256": hashlib.sha256((args.run_dir / "best.pt").read_bytes()).hexdigest(),
                  "deployment_status": "research only; not integrated into authentication"}, args.run_dir / "comparison.json")
    print(f"Saved training evidence to {args.run_dir}")


if __name__ == "__main__":
    main()
