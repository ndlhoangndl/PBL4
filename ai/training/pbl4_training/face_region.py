"""Reviewed face-skin region preprocessing shared by training and future inference.

The polygon follows visible forehead/hairline, cheeks and chin. It is not an
automatic hair segmenter. Hair covering skin must be excluded by the reviewer.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image, ImageDraw

POLICY = "reviewed-face-skin-v1"


def validate_region(region, width, height):
    if not isinstance(region, dict) or region.get("policy") != POLICY or region.get("reviewed") is not True:
        raise ValueError("Face region must be manually reviewed under reviewed-face-skin-v1")
    polygon = region.get("polygon_xy")
    eyes = region.get("eyes_xy")
    if not isinstance(polygon, list) or not 3 <= len(polygon) <= 80:
        raise ValueError("Face outline needs 3–80 ordered points")
    if not isinstance(eyes, list) or len(eyes) != 2:
        raise ValueError("Face alignment needs two reviewed eye centers")
    for point in polygon + eyes:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each face-region point must be [x,y]")
        x, y = point
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (x, y)):
            raise ValueError("Face coordinates must be finite numbers")
        if not 0 <= x <= width - 1 or not 0 <= y <= height - 1:
            raise ValueError("Face-region point is outside the original image")
    area = abs(sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][0] * polygon[i][1] for i in range(len(polygon)))) / 2
    if area < 25:
        raise ValueError("Face region is too small")
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        if a == b:
            raise ValueError("Face outline contains consecutive duplicate points")
        for j in range(i + 2, len(polygon)):
            if i == 0 and j == len(polygon) - 1:
                continue
            c, d = polygon[j], polygon[(j + 1) % len(polygon)]
            if cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0:
                raise ValueError("Face outline must not cross itself")
    if math.dist(eyes[0], eyes[1]) < 4:
        raise ValueError("Eye centers are too close")
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon], fill=255)
    for eye in eyes:
        if not mask.getpixel((round(eye[0]), round(eye[1]))):
            raise ValueError("Both eye centers must be inside the reviewed face outline")
    return mask


def validate_region_file(path: Path, region):
    if not isinstance(region, dict):
        raise ValueError("Missing reviewed face_region; full-frame recognition is disabled")
    if region.get("source_image_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Face-region annotation hash does not match source image")
    with Image.open(path) as image:
        validate_region(region, *image.size)


def prepare_face(image: Image.Image, region, size=112, return_mask=False):
    """Mask BEFORE interpolation, level the eyes, crop skin ROI, square letterbox.

    Constant neutral pixels replace background/hair outside the human polygon.
    Exact semantic exclusion is the reviewer's job, not a promise of this mask.
    """
    if size < 16:
        raise ValueError("Face output size must be at least 16")
    image = image.convert("RGB")
    mask = validate_region(region, *image.size)
    neutral = (127, 127, 127)
    masked = Image.composite(image, Image.new("RGB", image.size, neutral), mask)
    left, right = sorted(region["eyes_xy"], key=lambda p: p[0])
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    masked = masked.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=neutral)
    mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
    masked = Image.composite(masked, Image.new("RGB", masked.size, neutral), mask)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Empty face region")
    face = masked.crop(bounds)
    face_mask = mask.crop(bounds)
    side = max(face.size)
    square = Image.new("RGB", (side, side), neutral)
    offset = ((side - face.width) // 2, (side - face.height) // 2)
    square.paste(face, offset)
    output = square.resize((size, size), Image.Resampling.BILINEAR)
    if return_mask:
        square_mask = Image.new("L", (side, side), 0)
        square_mask.paste(face_mask, offset)
        return output, square_mask.resize((size, size), Image.Resampling.NEAREST)
    return output
