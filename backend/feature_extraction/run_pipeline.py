"""
Step 4 of HeatScape Feature Extraction Pipeline (Utility / Batch Orchestrator)
================================================================================
run_pipeline.py

PURPOSE
-------
Orchestrates Steps 1→2→3 in one command for either:
  (a) A single image
  (b) A whole folder of images (batch mode)

Also acts as the integration point where:
  - Member 1's features CSV  (from Steps 1-3 above)
  - Member 2's thermal CSV   (Landsat LST, NDVI, air_temp, humidity)
  …are merged into the final dataset for Member 3 to train the ML model.

USAGE — single image
----------------------
  python run_pipeline.py \\
      --image  data/sample_aerial.jpg \\
      --output output/

USAGE — batch folder
----------------------
  python run_pipeline.py \\
      --batch   data/images/ \\
      --output  output/batch/ \\
      --thermal data/thermal.csv

  The thermal CSV should have columns:
    image_name, lat, lon, lst, air_temperature, humidity, ndvi, ndbi

  The pipeline matches rows by image_name (filename without extension).

OUTPUT
------
  output/
  ├── masks/              ← SAM2 masks per image
  ├── labels/             ← CLIP-labelled masks
  ├── features/           ← Extracted feature JSONs/CSVs
  └── heatscape_dataset.csv ← Final merged dataset for Member 3
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

from sam2_segmenter  import generate_masks, save_results
from clip_labeller   import label_all_masks
from feature_calculator import extract_features, save_features


# ── Supported image formats ────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def process_single(image_path: str, output_dir: str, device: str) -> dict:
    """Run the full 3-step pipeline on a single image."""
    img_stem    = Path(image_path).stem
    out_root    = Path(output_dir)
    masks_dir   = str(out_root / "masks"   / img_stem)
    labels_dir  = str(out_root / "labels"  / img_stem)
    features_dir= str(out_root / "features"/ img_stem)

    print(f"\n{'─'*60}")
    print(f"Processing: {image_path}")
    print(f"{'─'*60}")

    # Step 1: SAM2 Segmentation
    image_rgb, masks = generate_masks(image_path, device)
    save_results(image_rgb, masks, masks_dir)
    masks_npz = str(Path(masks_dir) / "masks.npz")

    # Step 2: CLIP Labelling
    labels_npz = label_all_masks(image_path, masks_npz, labels_dir, device)

    # Step 3: Feature Calculation
    features = extract_features(image_path, labels_npz)
    features["image_name"] = Path(image_path).stem
    save_features(features, features_dir)

    return features


def merge_with_thermal(features_df: pd.DataFrame, thermal_csv: str) -> pd.DataFrame:
    """
    Merge Member 1's spatial features with Member 2's thermal data.

    Thermal CSV expected columns:
      image_name, lat, lon, lst, air_temperature, humidity
    """
    thermal_df = pd.read_csv(thermal_csv)
    print(f"\n[MERGE] Thermal data: {len(thermal_df)} rows")
    print(f"[MERGE] Features data: {len(features_df)} rows")

    # Merge on image_name
    merged = pd.merge(
        features_df,
        thermal_df,
        on    = "image_name",
        how   = "left",
        suffixes=("_m1", "_m2"),
    )

    # Prefer Member 2's NDVI/NDBI over RGB proxies (they are more accurate)
    for col in ["ndvi", "ndbi"]:
        m2_col = f"{col}_m2"
        m1_col = f"{col}_m1"
        if m2_col in merged.columns:
            merged[col] = merged[m2_col].fillna(merged.get(m1_col, merged[m2_col]))
            merged.drop(columns=[c for c in [m1_col, m2_col] if c in merged.columns], inplace=True)

    missing_lst = merged["lst"].isna().sum()
    if missing_lst > 0:
        print(f"  [!] {missing_lst} rows missing thermal data (no Landsat match). "
              f"These will be dropped from training set.")

    return merged


def save_final_dataset(df: pd.DataFrame, output_dir: str):
    out_path = Path(output_dir) / "heatscape_dataset.csv"
    # Keep only rows with LST (needed for supervised learning)
    trainable = df.dropna(subset=["lst"])
    df.to_csv(str(Path(output_dir) / "heatscape_dataset_full.csv"), index=False)
    trainable.to_csv(str(out_path), index=False)
    print(f"\n{'═'*60}")
    print(f"  FINAL DATASET")
    print(f"  Total rows       : {len(df)}")
    print(f"  Trainable rows   : {len(trainable)}  (have LST ground truth)")
    print(f"  Saved to         : {out_path}")
    print(f"{'═'*60}")
    print("\n  Feature columns:")
    for col in trainable.columns:
        print(f"    • {col}")
    print("\n[DONE] Ready to hand off to Member 3 for ML training!")


def main():
    parser = argparse.ArgumentParser(
        description="HeatScape Feature Extraction Pipeline Orchestrator"
    )
    # Single image mode
    parser.add_argument("--image",   help="Single aerial image path")

    # Batch mode
    parser.add_argument("--batch",   help="Folder of aerial images")
    parser.add_argument("--thermal", help="Member 2's thermal CSV (for merging)")

    parser.add_argument("--output",  default="output", help="Base output directory")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Single mode ───────────────────────────────────────────────────────────
    if args.image:
        features = process_single(args.image, args.output, args.device)
        print("\n✓ Feature extraction complete for single image.")
        print("  To merge with thermal data, re-run with --batch and --thermal flags.")
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        image_dir  = Path(args.batch)
        image_paths = [
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ]
        if not image_paths:
            print(f"[ERROR] No images found in {image_dir}")
            sys.exit(1)

        print(f"[BATCH] Found {len(image_paths)} images in {image_dir}")
        all_features = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                features = process_single(img_path, args.output, args.device)
                all_features.append(features)
            except Exception as e:
                print(f"  [WARN] Skipped {img_path}: {e}")

        features_df = pd.DataFrame(all_features)
        features_csv= str(Path(args.output) / "member1_features.csv")
        features_df.to_csv(features_csv, index=False)
        print(f"\n[BATCH] Member 1 features saved → {features_csv}")

        # Merge with Member 2's thermal data if provided
        if args.thermal:
            merged_df = merge_with_thermal(features_df, args.thermal)
            save_final_dataset(merged_df, args.output)
        else:
            print("\n[INFO] No --thermal CSV provided.")
            print(f"[INFO] Member 1 features saved to {features_csv}")
            print("[INFO] Once Member 2 provides thermal.csv, run:")
            print(f"       python run_pipeline.py --batch {args.batch} "
                  f"--thermal <thermal.csv> --output {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
