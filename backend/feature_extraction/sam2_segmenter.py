"""
Step 1 of HeatScape Feature Extraction Pipeline
================================================
sam2_segmenter.py

PURPOSE
-------
Given a single aerial/drone/satellite RGB image, this script:
  1. Runs SAM 2 in "Automatic Mask Generation" mode — it segments
     EVERY distinct region in the image without needing any prompts.
  2. Saves the raw segmentation masks as a numpy array (.npz) and
     a visual overlay PNG for inspection.

WHY SAM 2 IN AUTO MODE?
------------------------
For aerial imagery we don't know where objects are in advance, so
we use SAM 2's automatic mask generator which tiles a dense grid of
points across the image and generates a mask per "thing" it finds.
This gives us hundreds of clean masks covering the whole image.

WHAT HAPPENS NEXT?
------------------
Step 2 (clip_labeller.py) takes these raw masks, crops each one and
asks CLIP "is this: vegetation, road, building, water, bare soil?"
The label with the highest CLIP similarity score wins.

USAGE
-----
  python sam2_segmenter.py --image path/to/aerial.png --output out/masks

REQUIREMENTS
------------
  pip install git+https://github.com/facebookresearch/sam2.git
  (torch and torchvision must be installed first — see requirements.txt)

SAM2 CHECKPOINT
---------------
Download the tiny checkpoint (fastest, good enough for classification):
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
Place it at: checkpoints/sam2_hiera_tiny.pt
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── SAM2 import guard ──────────────────────────────────────────────────────────
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print(
        "\n[ERROR] SAM2 is not installed.\n"
        "Run:  pip install git+https://github.com/facebookresearch/sam2.git\n"
        "After installing torch >= 2.1 first.\n"
    )
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "sam2_hiera_tiny.pt"
MODEL_CONFIG    = "sam2_hiera_t.yaml"   # bundled with SAM2 package


def load_sam2(device: str = "cpu"):
    """Load SAM 2 model from local checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}.\n"
            "Download from:\n"
            "  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt\n"
            f"Place at: {CHECKPOINT_PATH}"
        )
    print(f"[SAM2] Loading model on {device}…")
    sam2 = build_sam2(MODEL_CONFIG, str(CHECKPOINT_PATH), device=device)
    return sam2


def generate_masks(image_path: str, device: str = "cpu") -> tuple[np.ndarray, list[dict]]:
    """
    Run SAM 2 automatic mask generation on a single image.

    Returns
    -------
    image_rgb : (H, W, 3) uint8 array
    masks     : list of SAM2 mask dicts, each containing:
                  'segmentation' (H,W bool), 'area', 'bbox', 'stability_score'
    """
    # Load image
    pil_img   = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img)

    # Build the automatic mask generator
    # points_per_side=32 gives a dense 32×32 grid of seed points → good for aerial
    sam2   = load_sam2(device)
    generator = SAM2AutomaticMaskGenerator(
        model                 = sam2,
        points_per_side       = 32,   # increase to 64 for very high-res images (slower)
        pred_iou_thresh       = 0.82, # keep only confident masks
        stability_score_thresh= 0.90,
        min_mask_region_area  = 150,  # ignore tiny noise regions (pixels)
    )

    print(f"[SAM2] Generating masks for {image_path}…")
    masks = generator.generate(image_rgb)
    print(f"[SAM2] Generated {len(masks)} masks.")
    return image_rgb, masks


def save_results(image_rgb: np.ndarray, masks: list[dict], output_dir: str):
    """Save raw mask data (.npz) and a visual overlay."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save masks as numpy for Step 2
    mask_arrays = np.stack([m["segmentation"] for m in masks], axis=0)  # (N, H, W) bool
    np.savez_compressed(str(out / "masks.npz"), masks=mask_arrays)
    print(f"[SAM2] Saved {mask_arrays.shape[0]} masks → {out / 'masks.npz'}")

    # ── Visualisation ──────────────────────────────────────────────────────────
    overlay = image_rgb.copy().astype(np.float32)
    colours = plt.cm.tab20(np.linspace(0, 1, len(masks)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    for i, (mask_dict, colour) in enumerate(zip(masks, colours)):
        mask = mask_dict["segmentation"]
        coloured = np.zeros((*mask.shape, 4), dtype=np.float32)
        coloured[mask] = [*colour[:3], 0.5]
        axes[1].imshow(coloured)
    axes[1].set_title(f"SAM2 Segmentation ({len(masks)} regions)", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    vis_path = str(out / "sam2_overlay.png")
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAM2] Visual overlay → {vis_path}")

    return str(out / "masks.npz")


def main():
    parser = argparse.ArgumentParser(description="HeatScape SAM2 Segmenter")
    parser.add_argument("--image",  required=True, help="Path to aerial/drone image")
    parser.add_argument("--output", default="output/masks", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_rgb, masks = generate_masks(args.image, args.device)
    save_results(image_rgb, masks, args.output)
    print("\n[SAM2] ✓ Done. Run clip_labeller.py next to classify these masks.")


if __name__ == "__main__":
    main()
