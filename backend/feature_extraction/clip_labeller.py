"""
Step 2 of HeatScape Feature Extraction Pipeline
================================================
clip_labeller.py

*** GROUND-LEVEL IMAGERY MODE ***
----------------------------------
HeatScape collects images at STREET / CAMPUS GROUND LEVEL (phone camera),
NOT top-down aerial images.

This changes CLIP prompts significantly:
  ✗ Old aerial: "satellite image of green vegetation from above"
  ✓ New ground: "trees and leafy branches seen from street level"
  ✓ New ground: "open sky visible above buildings"

Why ground-level is scientifically valuable for UHI:
  1. SKY VIEW FACTOR (SVF) — the single most important variable for
     nocturnal urban heat retention — is best measured from street photos
     by computing what fraction of the image is classified as "sky".
  2. BUILDING FACADE MATERIAL — glass vs dark concrete directly affects
     solar absorption and radiative heat loss.
  3. VEGETATION PRESENCE — tree canopy density creates shade and latent
     cooling (evapotranspiration).
  4. PAVEMENT TYPE — impervious vs permeable surfaces.

The thermal ground truth (LST) still comes from Landsat/ECOSTRESS for
the GPS coordinate of each photo, merged at Step 4.
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import clip
    import torch
except ImportError:
    import sys
    print("\n[ERROR] CLIP or torch not installed.\nRun: pip install openai-clip torch\n")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
#  URBAN SURFACE CLASSES — GROUND-LEVEL STREET / CAMPUS PROMPTS
#  These are tuned for images taken with a phone camera at eye level.
# ──────────────────────────────────────────────────────────────────────────────
CLASSES = {
    "sky": [
        "open blue sky",
        "clear sky with no clouds",
        "bright sky between buildings",
        "overcast sky viewed from street level",
        "sky visible above urban buildings",
    ],
    "vegetation_tree": [
        "green tree canopy seen from below",
        "leaves and branches of a tree",
        "dense leafy trees along a campus road",
        "green foliage and bushes at street level",
        "tree trunk and branches",
    ],
    "building_facade": [
        "concrete wall of a building",
        "glass facade of an office building",
        "brick wall of a campus building",
        "building exterior wall seen from ground",
        "painted concrete building surface",
    ],
    "road_pavement": [
        "asphalt road from ground level",
        "concrete footpath or sidewalk",
        "paved campus pathway",
        "road surface seen from standing height",
        "grey concrete or asphalt floor",
    ],
    "grass_ground": [
        "grass lawn at ground level",
        "dry grass or mud on the ground",
        "bare soil or dirt path",
        "unpaved ground with gravel",
        "green grass on a campus lawn",
    ],
    "shadow": [
        "dark shadow on a wall or road",
        "shaded area under a tree or building",
        "deep shadow cast on the ground",
    ],
    "other_structure": [
        "metal railing or fence",
        "parked vehicle or car",
        "lamp post or utility pole",
        "rooftop seen from below at an angle",
        "water tank or roof equipment",
    ],
}

# Visual colours for the overlay
CLASS_COLOURS = {
    "sky"              : (147, 210, 255, 160),  # light blue
    "vegetation_tree"  : (34,  197,  94, 160),  # green
    "building_facade"  : (249, 115,  22, 160),  # orange
    "road_pavement"    : (148, 163, 184, 160),  # slate grey
    "grass_ground"     : (101, 163,  13, 160),  # olive green
    "shadow"           : (30,   30,  50, 160),  # near-black
    "other_structure"  : (200, 200, 200, 120),  # light grey
}


def load_clip(device: str = "cpu"):
    print("[CLIP] Loading ViT-B/32 model…")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


def encode_class_prompts(model, device: str = "cpu") -> dict:
    """Pre-encode all class text prompts (averaged for robustness)."""
    class_embeddings = {}
    with torch.no_grad():
        for class_name, prompts in CLASSES.items():
            tokens = clip.tokenize(prompts).to(device)
            embeds = model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            class_embeddings[class_name] = embeds.mean(dim=0)
    print(f"[CLIP] Encoded {len(class_embeddings)} class embeddings.")
    return class_embeddings


def classify_mask_crop(pil_image, mask, model, preprocess, class_embeddings, device):
    """Classify a single SAM 2 mask using CLIP similarity scoring."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return "other_structure", 0.0

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    if (y2 - y1) < 10 or (x2 - x1) < 10:
        return "other_structure", 0.0

    crop       = np.array(pil_image)[y1:y2, x1:x2].copy()
    mask_crop  = mask[y1:y2, x1:x2]
    crop[~mask_crop] = 0

    crop_pil  = Image.fromarray(crop)
    processed = preprocess(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embed = model.encode_image(processed)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

    scores = {
        cls: (image_embed @ emb.unsqueeze(-1)).item()
        for cls, emb in class_embeddings.items()
    }

    best_class = max(scores, key=scores.get)
    return best_class, scores[best_class]


def label_all_masks(image_path: str, masks_path: str, output_dir: str, device: str = "cpu"):
    """Full labelling pipeline for all masks."""
    out       = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pil_image = Image.open(image_path).convert("RGB")
    data      = np.load(masks_path)
    masks     = data["masks"]
    N         = masks.shape[0]
    print(f"[CLIP] Labelling {N} masks from ground-level image…")

    model, preprocess   = load_clip(device)
    class_embeddings    = encode_class_prompts(model, device)

    labels, confidences = [], []
    for i, mask in enumerate(masks):
        label, conf = classify_mask_crop(
            pil_image, mask, model, preprocess, class_embeddings, device
        )
        labels.append(label)
        confidences.append(conf)
        if (i + 1) % 20 == 0 or i == N - 1:
            print(f"  [{i+1}/{N}] {label} ({conf:.2f})")

    labels_arr      = np.array(labels)
    confidences_arr = np.array(confidences, dtype=np.float32)

    npz_path = str(out / "labelled_masks.npz")
    np.savez_compressed(npz_path, masks=masks, labels=labels_arr, confidences=confidences_arr)
    print(f"\n[CLIP] Saved → {npz_path}")

    # Visualisation
    image_np     = np.array(pil_image)
    overlay      = Image.fromarray(image_np).convert("RGBA")
    colour_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    for mask, label in zip(masks, labels):
        colour = CLASS_COLOURS.get(label, (200, 200, 200, 120))
        region = Image.new("RGBA", overlay.size, colour)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        colour_layer.paste(region, mask=mask_img)

    blended = Image.alpha_composite(overlay, colour_layer)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(image_np);  axes[0].set_title("Original Campus Image"); axes[0].axis("off")
    axes[1].imshow(blended);   axes[1].set_title("CLIP-Labelled (Ground-Level)"); axes[1].axis("off")
    legend_patches = [
        mpatches.Patch(color=np.array(c[:3]) / 255, label=name)
        for name, c in CLASS_COLOURS.items()
    ]
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=10)
    plt.tight_layout()
    vis_path = str(out / "labelled_overlay.png")
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[CLIP] Visual overlay → {vis_path}")
    return npz_path


def main():
    parser = argparse.ArgumentParser(description="HeatScape CLIP Labeller (Ground-Level Mode)")
    parser.add_argument("--image",  required=True)
    parser.add_argument("--masks",  required=True)
    parser.add_argument("--output", default="output/labels")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    label_all_masks(args.image, args.masks, args.output, args.device)
    print("\n[CLIP] ✓ Done. Run feature_calculator.py next.")


if __name__ == "__main__":
    main()
