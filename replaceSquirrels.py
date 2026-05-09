#!/usr/bin/env python3
"""
replaceSquirrels.py

Usage:
    python replaceSquirrels.py sqImageFiles

Loads each squirrel image listed in sqImageFiles and its corresponding
imageName-sqMask.png, then inpaints the masked squirrel region with a bird.

Output:
    imageName-squirrelReplaced.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting


PROMPT = (
    "photorealistic small songbird perched naturally next to the birdfeeder, "
    "realistic feathers, matching camera angle, matching daylight, wildlife photography"
)
NEGATIVE_PROMPT = (
    "squirrel, rodent, cartoon, painting, multiple birds, deformed bird, blurry, artifacts, unrealistic"
)
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def load_paths(list_file: Path) -> List[Path]:
    return [Path(line.strip()) for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pipeline(device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("Usage: python replaceSquirrels.py sqImageFiles", file=sys.stderr)
        return 1

    list_file = Path(argv[1])
    image_paths = load_paths(list_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(device)

    for img_path in image_paths:
        mask_path = img_path.with_name(f"{img_path.stem}-sqMask.png")
        out_path = img_path.with_name(f"{img_path.stem}-squirrelReplaced.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            result = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                mask_image=mask,
                guidance_scale=8.0,
                strength=0.99,
                num_inference_steps=40,
            ).images[0]
            result.save(out_path, quality=95)
            print(f"Saved: {out_path}")
        except Exception as exc:
            print(f"ERROR processing {img_path}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
