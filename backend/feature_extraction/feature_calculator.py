"""
Step 3 of HeatScape Feature Extraction Pipeline
================================================
feature_calculator.py — GROUND-LEVEL VERSION

FEATURES EXTRACTED FROM GROUND-LEVEL CAMPUS IMAGES
----------------------------------------------------

1. SKY VIEW FACTOR (SVF) — Most critical UHI variable
   ────────────────────────────────────────────────────
   Definition: Fraction of the visible hemisphere occupied by sky
   (vs blocked by buildings/trees).
   
   Ground-level measurement is SUPERIOR to aerial estimation because
   you literally see the sky from the point where heat accumulates.
   
   Formula used:
     SVF = (pixels labelled "sky") / (total image pixels)
   
   Physical meaning:
     SVF ≈ 1.0  →  Open field, excellent long-wave radiation escape
     SVF ≈ 0.3  →  Deep urban canyon, heat trapped overnight
   
   Scientific note: This aligns exactly with the Oke (1981) SVF method
   using hemispherical (fisheye) photography — a standard in urban
   climatology. Even regular photos give a useful proxy.

2. VEGETATION PRESENCE RATIO
   ──────────────────────────
   vegetation_pct = pixels labelled "vegetation_tree" / total
   
   Trees cool via:
     (a) SHADING — reduces solar gain on surfaces below
     (b) EVAPOTRANSPIRATION — releases water vapour (latent cooling)
   
3. PAVEMENT / IMPERVIOUS SURFACE RATIO
   ──────────────────────────────────────
   pavement_pct — contributes to heat absorption and storage

4. BUILDING FACADE AREA (proxy for urban density)
   ─────────────────────────────────────────────────
   building_pct — more facade = more thermal mass and less sky view

5. GRASS / BARE SOIL RATIO
   ─────────────────────────
   grass_pct — dry grass stores heat; wet/green grass cools

6. SHADOW FRACTION
   ─────────────────
   shadow_pct — high shadow = shading is effective at this location

7. SPECTRAL INDICES (RGB proxies, replaced by Landsat values later)
   ndvi_estimate  — vegetation greenness proxy from green/red channels
   ndbi_estimate  — built-up index proxy from red/blue channels

8. IMAGE METADATA
   gps_lat, gps_lon (added manually or from EXIF), timestamp

THERMAL GROUND TRUTH (from Member 2)
--------------------------------------
The LST for each photo location comes from:
  • Landsat 8/9 Band 10 → 100m resolution LST
  • ECOSTRESS → 70m resolution, ISS-based, daily overpasses
  • Bhuvan (ISRO) → Resourcesat, Cartosat — India-specific
These are merged at Step 4 (run_pipeline.py) using GPS coordinates.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


# ──────────────────────────────────────────────────────────────────────────────
#  GPS EXIF EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def _convert_to_degrees(value):
    """Convert GPS EXIF DMS (degrees/minutes/seconds) to decimal degrees."""
    d, m, s = value
    return float(d) + float(m) / 60 + float(s) / 3600


def extract_gps_from_exif(image_path: str) -> dict:
    """
    Try to read GPS coordinates embedded in JPEG EXIF by the phone camera.
    Returns dict with 'lat' and 'lon', or None if unavailable.
    """
    try:
        pil = Image.open(image_path)
        exif_data = pil._getexif()
        if not exif_data:
            return {"lat": None, "lon": None}

        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
        gps_info_raw = exif.get("GPSInfo", {})
        gps_info = {GPSTAGS.get(k, k): v for k, v in gps_info_raw.items()}

        lat = _convert_to_degrees(gps_info["GPSLatitude"])
        lon = _convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info.get("GPSLatitudeRef") == "S":
            lat = -lat
        if gps_info.get("GPSLongitudeRef") == "W":
            lon = -lon
        return {"lat": round(lat, 6), "lon": round(lon, 6)}
    except Exception:
        return {"lat": None, "lon": None}


# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE COMPUTATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_area_percentages(masks: np.ndarray, labels: np.ndarray, H: int, W: int) -> tuple:
    """
    Compute pixel-percentage of each urban class.
    First mask wins for overlapping pixels.
    """
    assigned = np.full((H, W), -1, dtype=np.int32)
    for i, mask in enumerate(masks):
        unassigned = assigned == -1
        assigned[mask & unassigned] = i

    all_classes = [
        "sky", "vegetation_tree", "building_facade",
        "road_pavement", "grass_ground", "shadow", "other_structure"
    ]
    total = H * W
    pcts  = {}
    for cls in all_classes:
        cls_idx   = np.where(labels == cls)[0]
        cls_pixels = np.sum(np.isin(assigned, cls_idx))
        pcts[cls] = round(cls_pixels / total * 100, 2)

    pcts["unlabelled_pct"] = round(np.sum(assigned == -1) / total * 100, 2)
    return pcts, assigned


def compute_sky_view_factor(pcts: dict) -> float:
    """
    Ground-level Sky View Factor:
      SVF = sky_pct / 100
    
    This is a direct measurement from the image — much more accurate
    than any aerial proxy. For a proper hemispherical measurement,
    use a fisheye lens; otherwise this linear approximation is accepted
    in urban climatology literature for standard camera images.
    
    Range: 0.0 (completely enclosed) to 1.0 (completely open sky)
    """
    return round(pcts.get("sky", 0.0) / 100.0, 4)


def compute_canopy_density(pcts: dict) -> float:
    """
    Tree canopy density proxy from ground view.
    Combines overhead tree branches (vegetation_tree) and
    grass at ground level (grass_ground).
    """
    return round((pcts.get("vegetation_tree", 0) + pcts.get("grass_ground", 0)) / 100.0, 4)


def estimate_building_height_proxy(masks: np.ndarray, labels: np.ndarray, H: int) -> float:
    """
    From ground-level images, building height can be estimated by the
    vertical extent of the building_facade masks as a fraction of image height.
    (Full 3D height comes from TRELLIS — this is a placeholder proxy.)
    """
    building_masks = masks[labels == "building_facade"]
    if len(building_masks) == 0:
        return 0.0

    fractions = []
    for mask in building_masks:
        rows = np.any(mask, axis=1)
        if rows.any():
            extent = np.where(rows)[0][-1] - np.where(rows)[0][0]
            fractions.append(extent / H)

    return round(float(np.mean(fractions)) if fractions else 0.0, 4)


def estimate_ndvi_rgb(image_array: np.ndarray) -> float:
    """
    Proxy NDVI from RGB: (Green - Red) / (Green + Red).
    Replaced by real Landsat NDVI from Member 2.
    """
    r = image_array[:, :, 0].astype(float)
    g = image_array[:, :, 1].astype(float)
    return round(float(np.mean((g - r) / (g + r + 1e-6))), 4)


def estimate_ndbi_rgb(image_array: np.ndarray) -> float:
    """
    Proxy NDBI from RGB: (Red - Blue) / (Red + Blue).
    Replaced by real Landsat NDBI from Member 2.
    """
    r = image_array[:, :, 0].astype(float)
    b = image_array[:, :, 2].astype(float)
    return round(float(np.mean((r - b) / (r + b + 1e-6))), 4)


def compute_solar_exposure_proxy(pcts: dict) -> float:
    """
    Solar exposure: areas with high sky view and low shadow get more sun.
    Proxy: exposed = (sky_pct + grass_pct) - shadow_pct, normalised.
    """
    exposed = pcts.get("sky", 0) + pcts.get("grass_ground", 0) - pcts.get("shadow", 0)
    return round(max(0.0, min(1.0, exposed / 100.0)), 4)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(image_path: str, labels_path: str, manual_lat: float = None, manual_lon: float = None) -> dict:
    """
    Full feature extraction from a ground-level campus image.
    Returns a feature dict matching the HeatScape ML schema.
    """
    pil_image   = Image.open(image_path).convert("RGB")
    image_array = np.array(pil_image)
    H, W        = image_array.shape[:2]

    data        = np.load(labels_path, allow_pickle=True)
    masks       = data["masks"]
    labels      = data["labels"]
    confidences = data["confidences"]

    # GPS from EXIF (auto) or manual override
    gps = extract_gps_from_exif(image_path)
    if manual_lat is not None:
        gps["lat"] = manual_lat
    if manual_lon is not None:
        gps["lon"] = manual_lon

    # Core percentages
    pcts, _ = compute_area_percentages(masks, labels, H, W)

    # Derived features
    svf           = compute_sky_view_factor(pcts)
    canopy_density= compute_canopy_density(pcts)
    bldg_height   = estimate_building_height_proxy(masks, labels, H)
    ndvi          = estimate_ndvi_rgb(image_array)
    ndbi          = estimate_ndbi_rgb(image_array)
    solar_exp     = compute_solar_exposure_proxy(pcts)

    features = {
        # ── Identity ────────────────────────────────────────────────────────
        "image_name"          : Path(image_path).stem,
        "image_path"          : str(image_path),
        "lat"                 : gps["lat"],
        "lon"                 : gps["lon"],

        # ── Core ML features (matching HeatScape Streamlit schema) ──────────
        "vegetation_pct"      : pcts.get("vegetation_tree", 0.0),
        "pavement_pct"        : pcts.get("road_pavement", 0.0),
        "building_pct"        : pcts.get("building_facade", 0.0),
        "tree_canopy_pct"     : pcts.get("vegetation_tree", 0.0),
        "grass_pct"           : pcts.get("grass_ground", 0.0),
        "sky_pct"             : pcts.get("sky", 0.0),
        "shadow_pct"          : pcts.get("shadow", 0.0),

        # ── Derived physics-based features ───────────────────────────────────
        "sky_view_factor"     : svf,          # KEY UHI variable from ground photo
        "canopy_density"      : canopy_density,
        "building_height_m"   : bldg_height,  # replaced by TRELLIS later
        "solar_exposure"      : solar_exp,

        # ── Spectral proxies (replaced by Landsat values from Member 2) ──────
        "ndvi"                : ndvi,
        "ndbi"                : ndbi,

        # ── QA metadata ──────────────────────────────────────────────────────
        "n_masks"             : int(len(masks)),
        "mean_clip_confidence": round(float(np.mean(confidences)), 4),
        "unlabelled_pct"      : pcts.get("unlabelled_pct", 0.0),
        "image_width"         : W,
        "image_height"        : H,

        # ── Thermal targets (filled by Member 2's CSV merge in Step 4) ───────
        "lst"                 : None,   # Landsat/ECOSTRESS Land Surface Temp
        "air_temperature"     : None,   # IMD weather station
        "humidity"            : None,   # IMD weather station
    }

    return features


def save_features(features: dict, output_dir: str) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = str(out / "features.json")
    with open(json_path, "w") as f:
        json.dump(features, f, indent=2)

    df       = pd.DataFrame([features])
    csv_path = str(out / "features.csv")
    df.to_csv(csv_path, index=False)

    _pretty_print(features)
    print(f"\n[FEATURES] Saved → {json_path}")
    return csv_path


def _pretty_print(features: dict):
    print("\n" + "═" * 58)
    print("  HeatScape Ground-Level Feature Extraction Summary")
    print("═" * 58)

    print(f"  GPS:  lat={features.get('lat')}, lon={features.get('lon')}")
    print(f"  File: {features.get('image_name')}")
    print()

    display = [
        ("sky_view_factor",   "SVF (key UHI variable)"),
        ("vegetation_pct",    "Vegetation %"),
        ("pavement_pct",      "Pavement %"),
        ("building_pct",      "Building facade %"),
        ("sky_pct",           "Sky %"),
        ("shadow_pct",        "Shadow %"),
        ("ndvi",              "NDVI (RGB proxy)"),
        ("ndbi",              "NDBI (RGB proxy)"),
        ("solar_exposure",    "Solar exposure"),
    ]

    for key, label in display:
        val = features.get(key)
        print(f"  {label:<30} {str(val):>8}")

    print("\n  NOTE: ndvi, ndbi, lst, air_temperature, humidity will be")
    print("        overwritten by real Landsat/ECOSTRESS data (Member 2).")
    print("═" * 58)


def main():
    parser = argparse.ArgumentParser(description="HeatScape Feature Calculator (Ground-Level)")
    parser.add_argument("--image",  required=True, help="Ground-level campus photo")
    parser.add_argument("--labels", required=True, help="labelled_masks.npz from clip_labeller.py")
    parser.add_argument("--output", default="output/features")
    parser.add_argument("--lat",    type=float, default=None, help="Manual GPS lat override")
    parser.add_argument("--lon",    type=float, default=None, help="Manual GPS lon override")
    args = parser.parse_args()

    features = extract_features(args.image, args.labels, args.lat, args.lon)
    save_features(features, args.output)
    print("\n[FEATURES] ✓ Done. Merge with Member 2's thermal CSV using run_pipeline.py.")


if __name__ == "__main__":
    main()
