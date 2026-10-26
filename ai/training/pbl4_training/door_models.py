"""Small project-defined research models. No YuNet/SFace/third-party weights.

These architectures deliberately have their own names and output contracts.
They cannot replace OpenCV FaceDetectorYN/FaceRecognizerSF model files.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(8, outputs), nn.SiLU(),
        )


class DoorBackbone(nn.Sequential):
    def __init__(self):
        super().__init__(ConvBlock(3, 16), ConvBlock(16, 32),
                         ConvBlock(32, 64), ConvBlock(64, 128))


class DoorFaceDetector(nn.Module):
    """One face per image: presence logit, normalized xyxy, five xy landmarks.

    Spatial pooling preserves position, unlike a global average classifier.
    This is a supervised single-face baseline, not a multi-face detector.
    """
    def __init__(self):
        super().__init__()
        self.features = DoorBackbone()
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                  nn.Linear(128 * 4 * 4, 128), nn.SiLU(),
                                  nn.Linear(128, 15))

    def forward(self, images):
        raw = self.head(self.features(images))
        corners = raw[:, 1:5].sigmoid().reshape(-1, 2, 2)
        boxes = torch.cat((corners.min(1).values, corners.max(1).values), dim=1)
        return {"presence": raw[:, 0], "boxes": boxes,
                "landmarks": raw[:, 5:].sigmoid().reshape(-1, 5, 2)}


def detection_loss(output, target):
    """Background has no box/landmark loss; invisible landmarks are masked."""
    presence = F.binary_cross_entropy_with_logits(output["presence"], target["presence"])
    positive = target["presence"] > 0.5
    boxes = F.smooth_l1_loss(output["boxes"][positive], target["boxes"][positive]) if positive.any() else output["boxes"].sum() * 0
    visible = target["visible"] & positive[:, None]
    points = F.smooth_l1_loss(output["landmarks"][visible], target["landmarks"][visible]) if visible.any() else output["landmarks"].sum() * 0
    return presence + 5 * boxes + 5 * points


class DoorFaceEmbedder(nn.Module):
    """128-D unit embeddings from already cropped/aligned RGB faces."""
    def __init__(self):
        super().__init__()
        self.features = DoorBackbone()
        self.projection = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 128))

    def forward(self, images):
        return F.normalize(self.projection(self.features(images)), dim=1)


class ArcFaceHead(nn.Module):
    """Additive angular margin head, used only during identity training.

    Formula: Deng et al., ArcFace, CVPR 2019. Architecture/loss code lives here;
    the mathematical technique is attributed, not claimed as a new invention.
    """
    def __init__(self, identities: int, margin: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(identities, 128))
        nn.init.xavier_uniform_(self.weight)
        self.margin, self.scale = margin, scale

    def forward(self, embeddings, labels):
        cosine = F.linear(embeddings, F.normalize(self.weight, dim=1)).clamp(-1 + 1e-6, 1 - 1e-6)
        sine = (1 - cosine.square()).clamp_min(1e-12).sqrt()
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        phi = torch.where(cosine > math.cos(math.pi - self.margin), phi,
                          cosine - math.sin(math.pi - self.margin) * self.margin)
        mask = F.one_hot(labels, self.weight.shape[0]).bool()
        return self.scale * torch.where(mask, phi, cosine)


class DoorAntiSpoof(nn.Module):
    """Single-frame RGB research classifier; classes: spoof=0, live=1."""
    def __init__(self):
        super().__init__()
        self.features = DoorBackbone()
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, images):
        return self.head(self.features(images))
