#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import OwlViTForObjectDetection, OwlViTProcessor, SamModel, SamProcessor


DETECTION_QUERIES = ["squirrel"]
DETECTION_THRESHOLD = 0.05
MASK_PADDING_FRACTION = 0.08

Box = Tuple[float, float, float, float]


def load_paths(list_file: Path) -> List[Path]:
    return [
        Path(line.strip().lstrip("\ufeff"))
        for line in list_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def expand_box(box: Box, image_width: int, image_height: int, fraction: float = MASK_PADDING_FRACTION) -> Box:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    pad_x = w * fraction
    pad_y = h * fraction
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_width - 1), x2 + pad_x),
        min(float(image_height - 1), y2 + pad_y),
    )


def detect_best_squirrel(image: Image.Image, processor, model, device: str) -> Box:
    inputs = processor(text=[DETECTION_QUERIES], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=DETECTION_THRESHOLD,
        text_labels=[DETECTION_QUERIES],
    )[0]

    squirrel_boxes = []

    for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
        if label == "squirrel":
            squirrel_boxes.append((float(score.item()), tuple(float(v) for v in box.tolist())))

    if not squirrel_boxes:
        raise RuntimeError("No squirrel detected.")

    squirrel_boxes.sort(key=lambda item: item[0], reverse=True)
    return squirrel_boxes[0][1]


def generate_mask(image: Image.Image, box: Box, sam_processor, sam_model, device: str) -> Image.Image:
    image_width, image_height = image.size
    padded_box = expand_box(box, image_width, image_height)

    inputs = sam_processor(
        images=image,
        input_boxes=[[[list(padded_box)]]],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    ious = outputs.iou_scores[0, 0].cpu().numpy()
    best_idx = int(np.argmax(ious))
    mask = masks[0][0][best_idx].numpy() > 0

    pil_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    pil_mask = pil_mask.filter(ImageFilter.MaxFilter(19))
    pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=2))
    pil_mask = pil_mask.point(lambda p: 255 if p > 32 else 0, mode="1").convert("L")

    return pil_mask


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("Usage: python segmentSquirrels.py sqImageFiles", file=sys.stderr)
        return 1

    list_file = Path(argv[1])
    image_paths = load_paths(list_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    owl_id = "google/owlvit-base-patch32"
    sam_id = "facebook/sam-vit-base"

    processor = OwlViTProcessor.from_pretrained(owl_id, use_fast=False)
    model = OwlViTForObjectDetection.from_pretrained(owl_id).to(device)

    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(device)

    model.eval()
    sam_model.eval()

    for i, img_path in enumerate(image_paths, start=1):
        try:
            print(f"[{i}/{len(image_paths)}] Processing: {img_path}")

            image = Image.open(img_path).convert("RGB")

            sq_box = detect_best_squirrel(image, processor, model, device)
            mask = generate_mask(image, sq_box, sam_processor, sam_model, device)

            mask_path = img_path.with_name(f"{img_path.stem}-sqMask.png")
            mask.save(mask_path)

            print(f"Saved mask: {mask_path}")

        except Exception as exc:
            print(f"ERROR processing {img_path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))