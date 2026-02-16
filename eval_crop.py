#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image  # pip install pillow

"""
python eval_crop.py qualitative_crop_folder/chair qualitative_crop_folder/chair/cropped --bbox 240 190 480 370
python eval_crop.py qualitative_crop_folder/drums qualitative_crop_folder/drums/cropped --bbox 240 190 480 370
python eval_crop.py qualitative_crop_folder/lego qualitative_crop_folder/lego/cropped --bbox 290 280 530 460
python eval_crop.py qualitative_crop_folder/materials qualitative_crop_folder/materials/cropped --bbox 280 260 520 440
python eval_crop.py qualitative_crop_folder/mic qualitative_crop_folder/mic/cropped --bbox 115 70 340 265
python eval_crop.py qualitative_crop_folder/ship qualitative_crop_folder/ship/cropped --bbox 280 260 640 530

python eval_crop.py qualitative_crop_folder/family qualitative_crop_folder/family/cropped --bbox 660 140 900 320
python eval_crop.py qualitative_crop_folder/caterpillar qualitative_crop_folder/caterpillar/cropped --bbox 360 120 840 500
python eval_crop.py qualitative_crop_folder/ignatius qualitative_crop_folder/ignatius/cropped --bbox 460 20 620 140
python eval_crop.py qualitative_crop_folder/truck qualitative_crop_folder/truck/cropped --bbox 100 120 580 500

python eval_crop.py qualitative_crop_folder/dynerf_beef qualitative_crop_folder/dynerf_beef/cropped --bbox 220 260 700 640
python eval_crop.py qualitative_crop_folder/dynerf_spinach qualitative_crop_folder/dynerf_spinach/cropped --bbox 220 260 700 640

python eval_crop.py qualitative_crop_folder/nhr_sport1 qualitative_crop_folder/nhr_sport1/cropped --bbox 330 140 480 280

python eval_crop.py qualitative_crop_folder/nhr_sport1 qualitative_crop_folder/nhr_sport1/cropped --bbox 365 130 475 205
"""
def find_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def determine_target_resolution(image_paths: List[Path]) -> Tuple[int, int]:
    """
    Determine a unified resolution for all images.

    If all images have the same size, return that.
    Otherwise, choose the lowest resolution by taking the minimum width
    and minimum height across all images and return (min_width, min_height).
    """
    if not image_paths:
        raise RuntimeError("No images found in the input directory.")

    sizes = []
    widths = []
    heights = []

    for p in image_paths:
        with Image.open(p) as im:
            w, h = im.size
            sizes.append((w, h))
            widths.append(w)
            heights.append(h)

    unique_sizes = sorted(set(sizes))
    if len(unique_sizes) == 1:
        base_size = unique_sizes[0]
        print(f"All images share the same size: {base_size}")
        return base_size

    # At least one mismatch – choose the lowest resolution
    target_w = min(widths)
    target_h = min(heights)
    print("Images have differing resolutions:")
    for sz in unique_sizes:
        print(f"  - {sz}")
    print(f"Using lowest resolution as target: ({target_w}, {target_h})")

    return target_w, target_h


def main():
    parser = argparse.ArgumentParser(
        description="Crop a fixed bounding box from all images in a folder."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the folder containing input images (jpg/png).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the folder where cropped images will be saved.",
    )
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="Bounding box to crop, in pixels: X1 Y1 X2 Y2",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    x1, y1, x2, y2 = args.bbox

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(input_dir)
    if not image_paths:
        raise SystemExit("No jpg/jpeg/png images found in the input directory.")

    # Determine unified resolution (possibly downsampling others)
    width, height = determine_target_resolution(image_paths)

    # Validate bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SystemExit(
            f"Invalid bbox {args.bbox} for image size (width={width}, height={height})."
        )

    print(f"Found {len(image_paths)} images.")
    print(f"Image resolution: {width}x{height}")
    print(f"Cropping bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Crop and save
    for p in image_paths:
        with Image.open(p) as im:
            if im.size != (width, height):
                print(f"Resizing {p.name} from {im.size} to {(width, height)}")
                im = im.resize((width, height), Image.LANCZOS)

            crop = im.crop((x1, y1, x2, y2))  # (left, upper, right, lower)
            out_name = f"{p.stem}_crop_{x1}_{y1}_{x2}_{y2}{p.suffix}"
            out_path = output_dir / out_name
            crop.save(out_path)
            print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
