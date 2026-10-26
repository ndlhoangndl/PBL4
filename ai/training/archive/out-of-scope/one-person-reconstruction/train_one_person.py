"""One-person reconstruction pilot, NOT an identity verifier or an ArcFace model.

Train/validation must come from disjoint sessions. No stranger false-accept rate,
test accuracy or authentication threshold can be inferred from this positive-only
experiment. Model is never copied into the running door application.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .dataset import DatasetError, read_jsonl, validate_manifest, write_jsonl
from .face_region import POLICY, prepare_face, validate_region_file
from .metrics import write_report


class DoorFaceAutoencoder(nn.Module):
    """Small CNN, random initialization, 32-dimensional reconstruction bottleneck."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(), nn.Flatten(), nn.Linear(64 * 8 * 8, 32))
        self.expand = nn.Sequential(nn.Linear(32, 64 * 8 * 8), nn.SiLU())
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, image):
        return self.decoder(self.expand(self.encoder(image)).reshape(-1, 64, 8, 8))


def masked_mse(prediction, target, mask):
    return ((prediction - target).square() * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * 3).clamp_min(1)


class FaceOnlyData(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.data = []
        for row in rows:
            path = Path(row["image_path"])
            validate_region_file(path, row.get("face_region"))
            with Image.open(path) as image:
                face, mask = prepare_face(image, row["face_region"], 64, return_mask=True)
            pixels = torch.from_numpy(np.asarray(face, dtype=np.float32).copy() / 255).permute(2, 0, 1)
            weights = torch.from_numpy((np.asarray(mask).copy() > 0).astype(np.float32))[None]
            self.data.append((pixels, weights))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(train_path, validation_path):
    groups = []
    seen_ids, seen_hashes, subjects = set(), set(), set()
    for path in (train_path, validation_path):
        validate_manifest(path, "face_recognition")
        rows = read_jsonl(path)
        for row in rows:
            image = Path(row["image_path"])
            row["image_path"] = str((image if image.is_absolute() else path.parent / image).resolve())
            digest = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
            if digest in seen_hashes or row["sample_id"] in seen_ids:
                raise DatasetError("Duplicate image bytes or sample ID across the pilot manifests")
            seen_hashes.add(digest)
            seen_ids.add(row["sample_id"])
            subjects.add(row["identity_id"])
            if row["identity_pool"] != "registered":
                raise DatasetError("Pilot accepts only the consenting target person's images")
        groups.append(rows)
    if len(subjects) != 1:
        raise DatasetError("This pilot expects exactly one identity")
    if {r["session_id"] for r in groups[0]} & {r["session_id"] for r in groups[1]}:
        raise DatasetError("Train and validation sessions must be disjoint")
    if {r["source_id"] for r in groups[0]} & {r["source_id"] for r in groups[1]}:
        raise DatasetError("Train and validation source lineages must be disjoint")
    return groups


def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    total = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for image, mask in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(image), image, mask).mean()
            if not torch.isfinite(loss):
                raise DatasetError("Non-finite reconstruction loss")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            total += float(loss.detach()) * len(image)
            count += len(image)
    return total / count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.run_dir.exists():
        parser.error("Use positive epochs and a NEW run directory")
    train_rows, validation_rows = load_data(args.train_manifest, args.validation_manifest)
    train, validation = FaceOnlyData(train_rows), FaceOnlyData(validation_rows)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    model = DoorFaceAutoencoder()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(validation, batch_size=8, shuffle=False)
    baseline = run_epoch(model, val_loader)
    mean_face = torch.stack([image for image, _ in train.data]).mean(0, keepdim=True)
    constant_loss = sum(float(masked_mse(mean_face.expand_as(image), image, mask).sum()) for image, mask in val_loader) / len(validation)
    args.run_dir.mkdir(parents=True)
    for label, rows in (("train", train_rows), ("validation", validation_rows)):
        write_jsonl(args.run_dir / f"{label}.jsonl", rows)
    metadata = {"architecture": "DoorFaceAutoencoder", "experiment": "one-person reconstruction pilot",
        "identity_id": train_rows[0]["identity_id"], "face_region_policy": POLICY,
        "input_size": 64, "normalization": "RGB / 255; reviewed skin mask applied before alignment",
        "initialization": "random; no third-party pretrained weights", "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()), "torch_version": str(torch.__version__),
        "device": "cpu", "train_samples": len(train), "validation_samples": len(validation),
        "train_sessions": sorted({r["session_id"] for r in train_rows}),
        "validation_sessions": sorted({r["session_id"] for r in validation_rows}),
        "selection": "minimum validation masked MSE", "test_samples": 0,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("train_one_person.py", "face_region.py")},
        "manifest_sha256": {name: hashlib.sha256((args.run_dir / f"{name}.jsonl").read_bytes()).hexdigest() for name in ("train", "validation")},
        "media_sha256": {row["sample_id"]: row["face_region"]["source_image_sha256"] for row in train_rows + validation_rows},
        "threshold": None, "identity_verification_validated": False,
        "limitations": ["positive-only data; no stranger rejection test", "two sessions only; no independent test set",
                       "reconstruction is not identity recognition accuracy", "not integrated into door authentication"]}
    write_report(metadata, args.run_dir / "run.json")
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        validation_loss = run_epoch(model, val_loader)
        state = {**metadata, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "validation_masked_mse": validation_loss}
        torch.save(state, args.run_dir / "last.pt")
        if validation_loss < best:
            best = validation_loss
            torch.save(state, args.run_dir / "best.pt")
        log = {"epoch": epoch, "train_masked_mse": train_loss, "validation_masked_mse": validation_loss}
        with (args.run_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
    selected = torch.load(args.run_dir / "best.pt", weights_only=True)
    model.load_state_dict(selected["model"])
    changed = sum(not torch.equal(value, initial[key]) for key, value in model.state_dict().items())
    if not changed:
        raise DatasetError("No model weights changed")
    with torch.no_grad():
        rows = []
        for index, (image, mask) in enumerate(validation.data):
            error = float(masked_mse(model(image[None]), image[None], mask[None])[0])
            rows.append({"sample_id": validation_rows[index]["sample_id"], "masked_reconstruction_mse": error})
    write_jsonl(args.run_dir / "validation-predictions.jsonl", rows)
    summary = {**metadata, "epochs": args.epochs, "best_epoch": selected["epoch"], "initial_validation_masked_mse": baseline,
        "constant_train_mean_validation_masked_mse": constant_loss, "best_validation_masked_mse": best,
        "changed_model_tensors": changed, "checkpoint_sha256": hashlib.sha256((args.run_dir / "best.pt").read_bytes()).hexdigest()}
    write_report(summary, args.run_dir / "summary.json")
    print(f"Pilot checkpoint saved: {args.run_dir / 'best.pt'}; identity verification remains unvalidated")


if __name__ == "__main__":
    main()
