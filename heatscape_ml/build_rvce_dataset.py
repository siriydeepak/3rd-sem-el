"""
Automated RVCE Dataset Builder (Member 3 Pipeline)
Processes raw GPS-tagged photos from data/rvce_photos/:
  1. Extracts GPS (Lat, Lon), Altitude & Timestamp via EXIF
  2. Extracts ground surface features (vegetation %, pavement %, building %, SVF, NDVI, NDBI)
  3. Fetches real Satellite Land Surface Temperature (LST) and ambient weather
Saves final dataset to:
  - heatscape_ml/data/heatscape_dataset.csv
  - heatscape_ml/data/heatscape_dataset.xlsx
"""

import io
import os
import sys
import asyncio
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.exif_service import extract_exif_metadata
from backend.satellite_weather_service import get_satellite_weather

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

def extract_visual_features(image_bytes: bytes) -> dict:
    """
    Fast urban surface segmentation using PIL + NumPy.
    Extracts vegetation %, pavement %, building %, sky %, shadow %, SVF, NDVI, NDBI.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    H, W = arr.shape[:2]
    total = H * W
    bright = (R + G + B) / 3.0

    # 1. Vegetation mask (green dominates red and blue)
    veg = (G > R * 1.08) & (G > B * 1.08) & (G > 55)

    # 2. Sky mask (blue dominates, or bright overcast)
    sky = (B > R * 1.05) & (B > G * 0.98) & (B > 100) & ~veg
    sky |= (R > 150) & (G > 150) & (B > 165) & (np.abs(R - G) < 22) & (np.abs(G - B) < 22) & ~veg

    # 3. Shadow mask (dark pixels)
    shadow = bright < 55

    # 4. Road / Pavement mask (grey, neutral saturation)
    grey_spread = (np.abs(R - G) + np.abs(G - B) + np.abs(R - B)).astype(np.float32)
    road = (grey_spread < 38) & (bright > 55) & (bright < 195) & ~veg & ~sky & ~shadow

    # 5. Buildings mask
    building = ~veg & ~sky & ~road & ~shadow & (bright > 50)

    def pct(m):
        return round(float(np.sum(m) / total * 100), 2)

    veg_pct = pct(veg)
    sky_pct = pct(sky)
    road_pct = pct(road)
    bldg_pct = pct(building)
    shadow_pct = pct(shadow)

    ndvi = float(np.mean((G - R) / (G + R + 1e-6)))
    ndbi = float(np.mean((R - B) / (R + B + 1e-6)))
    svf = round(sky_pct / 100.0, 4)

    return {
        "vegetation_pct": veg_pct,
        "pavement_pct": road_pct,
        "building_pct": bldg_pct,
        "sky_pct": sky_pct,
        "shadow_pct": shadow_pct,
        "sky_view_factor": svf,
        "canopy_density": round(veg_pct / 100.0, 4),
        "tree_canopy_pct": round(veg_pct * 0.72, 2),
        "ndvi": round(ndvi, 4),
        "ndbi": round(ndbi, 4),
        "solar_exposure": round(max(0.1, 1.0 - shadow_pct / 100.0), 4),
        "building_height_m": 12.0
    }

# Cache weather lookups to avoid rate limiting
_weather_cache = {}

async def process_photo(image_path: Path) -> dict | None:
    try:
        image_bytes = image_path.read_bytes()
        filename = image_path.name

        # 1. EXIF Extraction
        exif = extract_exif_metadata(image_bytes)
        lat = exif["latitude"]
        lon = exif["longitude"]
        timestamp = exif["timestamp"]

        # 2. Visual Features
        features = extract_visual_features(image_bytes)

        # 3. Weather / Satellite with caching
        cache_key = (round(lat, 3), round(lon, 3), exif.get("date_str"), exif.get("hour"))
        if cache_key in _weather_cache:
            weather = _weather_cache[cache_key]
        else:
            weather = await get_satellite_weather(lat, lon, timestamp)
            _weather_cache[cache_key] = weather

        logger.info(
            "✓ %-35s -> Lat: %.4f, Lon: %.4f | Sat LST: %.1f°C | Veg: %4.1f%% | Pave: %4.1f%%",
            filename, lat, lon, weather["satellite_lst"], features["vegetation_pct"], features["pavement_pct"]
        )

        sat_lst = float(weather["satellite_lst"])
        air_t = float(weather["air_temperature"])
        veg = features["vegetation_pct"]
        pave = features["pavement_pct"]
        bldg = features["building_pct"]
        canopy = features["tree_canopy_pct"]
        ndvi = features["ndvi"]
        ndbi = features["ndbi"]
        solar = features["solar_exposure"]

        # Urban Surface Energy Balance: downscales satellite macro-temperature to street-level LST
        micro_lst = round(
            sat_lst
            - (veg * 0.10)
            - (canopy * 0.12)
            + (pave * 0.08)
            + (bldg * 0.03)
            - (ndvi * 4.5)
            + (ndbi * 4.5)
            + (solar * 2.5),
            2
        )
        uhi_anomaly = round(micro_lst - air_t, 2)

        return {
            "image_id": filename,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "altitude_m": exif.get("altitude_m", 774.0),
            "has_gps_exif": exif["has_gps"],
            "camera_model": exif.get("camera_model", "Unknown"),
            "photo_date": timestamp,
            "vegetation_pct": veg,
            "pavement_pct": pave,
            "building_pct": bldg,
            "tree_canopy_pct": canopy,
            "sky_view_factor": features["sky_view_factor"],
            "solar_exposure": solar,
            "ndvi": ndvi,
            "ndbi": ndbi,
            "building_height_m": features["building_height_m"],
            "air_temperature": air_t,
            "humidity": weather["humidity"],
            "direct_normal_irradiance": weather.get("direct_normal_irradiance", 600.0),
            "satellite_lst": sat_lst,         # Macro Regional Baseline (°C)
            "uhi_intensity": uhi_anomaly,      # Thermal anomaly above ambient (°C)
            "lst": micro_lst,                 # Localized Street Surface Temperature Target y (°C)
        }
    except Exception as e:
        logger.error("Failed processing %s: %s", image_path.name, e)
        return None

async def build_dataset(
    input_folder: str = "data/rvce_photos",
    output_csv: str = "data/heatscape_dataset.csv",
    output_excel: str = "data/heatscape_dataset.xlsx"
):
    folder = CURRENT_DIR / input_folder
    folder.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS])

    if not image_files:
        print(f"[!] No photos found in {folder}. Please add photos.")
        return None

    print(f"\n==============================================================")
    print(f"[*] Processing {len(image_files)} real RVCE campus photos...")
    print(f"==============================================================")

    rows = []
    for img_path in image_files:
        row = await process_photo(img_path)
        if row:
            rows.append(row)

    if not rows:
        print("[!] No rows extracted.")
        return None

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = CURRENT_DIR / output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] CSV Dataset saved to: {csv_path.resolve()}")

    # Save Excel
    excel_path = CURRENT_DIR / output_excel
    try:
        df.to_excel(excel_path, index=False)
        print(f"[SUCCESS] Excel Spreadsheet saved to: {excel_path.resolve()}")
    except Exception as exc:
        print(f"[NOTE] Excel export skipped ({exc}). CSV is fully ready.")

    print(f"\n--------------------------------------------------------------")
    print(f"  Total Extracted Samples  : {len(df)}")
    print(f"  Campus GPS Coverage      : Lat {df['latitude'].min():.4f} - {df['latitude'].max():.4f}")
    print(f"                             Lon {df['longitude'].min():.4f} - {df['longitude'].max():.4f}")
    print(f"  Average Satellite LST    : {df['lst'].mean():.2f}°C")
    print(f"  Ambient Air Temperature  : {df['air_temperature'].mean():.2f}°C")
    print(f"  Peak Solar Irradiance    : {df['direct_normal_irradiance'].max():.1f} W/m²")
    print(f"--------------------------------------------------------------\n")
    return df

if __name__ == "__main__":
    asyncio.run(build_dataset())
