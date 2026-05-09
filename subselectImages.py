#!/usr/bin/env python3
"""
subselectImages.py

Usage:
    python subselectImages.py imageFileList outFileName

Reads a text file containing one image path per line and writes out only those
images that appear to contain a squirrel perched at / next to a birdfeeder.

Approach:
1. Zero-shot object detection with OWL-ViT.
2. Look for both squirrel-like and feeder-like prompts.
3. Score squirrel-feeder pairs using confidence plus spatial proximity.
4. Keep images whose best pair score crosses a threshold.

This version is tuned for higher recall while still trying to avoid obviously
bad matches.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


SQUIRREL_QUERIES = [
    "a squirrel",
    "a gray squirrel",
    "a brown squirrel",
    "a squirrel on a feeder",
]

FEEDER_QUERIES = [
    "a bird feeder",
    "a hanging bird feeder",
    "a backyard bird feeder",
    "a feeder full of seeds",
]

DEFAULT_SCORE_THRESHOLD = 0.04
MIN_SQUIRREL_SCORE = 0.06
MIN_FEEDER_SCORE = 0.05
PAIR_SCORE_THRESHOLD = 0.16

MAX_CENTER_DISTANCE_FACTOR = 2.6
MAX_EDGE_GAP_FACTOR = 1.25
OVERLAP_BONUS_IOU_THRESHOLD = 0.02


Box = Tuple[float, float, float, float]
DetectedObject = Tuple[str, float, Box]


def load_image_paths(list_file: Path) -> List[Path]:
    paths: List[Path] = []
    for raw in list_file.read_text(encoding="utf-8-sig").splitlines():
        raw = raw.strip().lstrip("\ufeff")
        if raw:
            paths.append(Path(raw))
    return paths


def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_size(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (max(1.0, x2 - x1), max(1.0, y2 - y1))


def box_edge_gap(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def pair_match_score(squirrel_box: Box, feeder_box: Box) -> float:
    sx, sy = box_center(squirrel_box)
    fx, fy = box_center(feeder_box)
    sw, sh = box_size(squirrel_box)
    fw, fh = box_size(feeder_box)

    center_distance = math.hypot(sx - fx, sy - fy)
    size_scale = max(sw, sh, fw, fh)
    edge_gap = box_edge_gap(squirrel_box, feeder_box)
    iou = box_iou(squirrel_box, feeder_box)

    center_score = max(
        0.0,
        1.0 - center_distance / max(1.0, MAX_CENTER_DISTANCE_FACTOR * size_scale),
    )
    edge_score = max(
        0.0,
        1.0 - edge_gap / max(1.0, MAX_EDGE_GAP_FACTOR * size_scale),
    )
    overlap_score = 0.25 if iou >= OVERLAP_BONUS_IOU_THRESHOLD else 0.0

    return max(center_score, edge_score) + overlap_score


def load_detector(device: str):
    model_id = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_id, use_fast=False)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def detect_objects(
    image: Image.Image,
    processor: OwlViTProcessor,
    model: OwlViTForObjectDetection,
    device: str,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> List[DetectedObject]:
    queries = SQUIRREL_QUERIES + FEEDER_QUERIES

    inputs = processor(text=[queries], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
        text_labels=[queries],
    )[0]

    detections: List[DetectedObject] = []
    for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
        detections.append((label, float(score.item()), tuple(float(v) for v in box.tolist())))

    return detections


def select_image(detections: Sequence[DetectedObject]) -> bool:
    squirrel_labels = set(SQUIRREL_QUERIES)
    feeder_labels = set(FEEDER_QUERIES)

    squirrels = [
        (score, box)
        for label, score, box in detections
        if label in squirrel_labels and score >= MIN_SQUIRREL_SCORE
    ]
    feeders = [
        (score, box)
        for label, score, box in detections
        if label in feeder_labels and score >= MIN_FEEDER_SCORE
    ]

    if not squirrels or not feeders:
        return False

    best_pair_score = 0.0
    for squirrel_score, squirrel_box in squirrels:
        for feeder_score, feeder_box in feeders:
            spatial_score = pair_match_score(squirrel_box, feeder_box)
            total_score = squirrel_score + feeder_score + spatial_score
            best_pair_score = max(best_pair_score, total_score)

    return best_pair_score >= PAIR_SCORE_THRESHOLD


def main(argv: Sequence[str]) -> int:
    if len(argv) != 3:
        print("Usage: python subselectImages.py imageFileList outFileName", file=sys.stderr)
        return 1

    list_file = Path(argv[1])
    out_file = Path(argv[2])
    image_paths = load_image_paths(list_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_detector(device)

    selected: List[str] = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            detections = detect_objects(image, processor, model, device)
            if select_image(detections):
                selected.append(str(img_path))
                print(f"SELECTED: {img_path}")
            else:
                print(f"SKIPPED:   {img_path}")
        except Exception as exc:
            print(f"ERROR reading {img_path}: {exc}", file=sys.stderr)

    out_file.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    print(f"Wrote {len(selected)} selected image(s) to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
