"""
Automated RVCE Dataset Builder (Member 3 Pipeline)
Takes raw GPS-tagged photos from data/rvce_photos/, extracts:
  1. GPS & Timestamp via EXIF
  2. Ground surface features (vegetation %, pavement %, SVF, NDVI, NDBI) via Computer Vision
  3. Real satellite thermal LST and ambient weather via Satellite Reanalysis API
Saves the final unified dataset to:
  - data/heatscape_dataset.csv
  - data/heatscape_dataset.xlsx
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import pandas as pd

# Setup path so we can import backend services cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.exif_service import extract_exif_metadata
from backend.satellite_weather_service import get_satellite_weather
from backend.main import _colour_segment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

async def process_photo(image_path: Path) -> dict | None:
    """
    Processes a single raw photo and extracts its full feature row.
    """
    try:
        image_bytes = image_path.read_bytes()
        filename = image_path.name
        
        # 1. EXIF Extraction
        exif = extract_exif_metadata(image_bytes)
        lat = exif["latitude"]
        lon = exif["longitude"]
        timestamp = exif["timestamp"]
        has_gps = exif["has_gps"]
        
        # 2. Visual Feature Segmentation
        features, _, _ = _colour_segment(image_bytes)
        
        # 3. Satellite & Weather Retrieval at that coordinate & time
        weather = await get_satellite_weather(lat, lon, timestamp)
        
        logger.info(
            "Processed '%s' -> Lat: %.4f, Lon: %.4f, Sat LST: %.1f°C, Veg: %.1f%%",
            filename, lat, lon, weather["satellite_lst"], features["vegetation_pct"]
        )
        
        return {
            "image_id": filename,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "has_gps_exif": has_gps,
            "photo_date": timestamp,
            "vegetation_pct": features["vegetation_pct"],
            "pavement_pct": features["pavement_pct"],
            "building_pct": features["building_pct"],
            "tree_canopy_pct": round(features["vegetation_pct"] * 0.7, 2),
            "ndvi": features["ndvi"],
            "ndbi": features["ndbi"],
            "building_height_m": 12.0,
            "sky_view_factor": features["sky_view_factor"],
            "solar_exposure": features["solar_exposure"],
            "air_temperature": weather["air_temperature"],
            "humidity": weather["humidity"],
            "direct_normal_irradiance": weather.get("direct_normal_irradiance", 800.0),
            "lst": weather["satellite_lst"],  # Ground-truth Target y
        }
    except Exception as e:
        logger.error("Failed processing '%s': %s", image_path.name, e)
        return None

async def build_dataset(
    input_folder: str = "data/rvce_photos",
    output_csv: str = "data/heatscape_dataset.csv",
    output_excel: str = "data/heatscape_dataset.xlsx"
):
    """
    Loops through all photos in input_folder, extracts all features, and saves heatscape_dataset.csv.
    """
    folder = Path(__file__).resolve().parent / input_folder
    folder.mkdir(parents=True, exist_ok=True)
    
    image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
    
    if not image_files:
        print("==================================================================")
        print(f"[!] No photos found in folder: {folder}")
        print("    Please drop your campus photos (.jpg, .png) into this folder.")
        print(f"    Folder location: {folder.resolve()}")
        print("==================================================================")
        return None
    
    print("==================================================================")
    print(f"[*] Found {len(image_files)} photos in {folder.name}. Processing...")
    print("==================================================================")
    
    rows = []
    for img_path in image_files:
        row = await process_photo(img_path)
        if row:
            rows.append(row)
            
    if not rows:
        print("[!] No rows could be extracted.")
        return None
        
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = Path(__file__).resolve().parent / output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] CSV Dataset saved to: {csv_path.resolve()}")
    
    # Save Excel
    excel_path = Path(__file__).resolve().parent / output_excel
    try:
        df.to_excel(excel_path, index=False)
        print(f"[SUCCESS] Excel Spreadsheet saved to: {excel_path.resolve()}")
    except Exception as exc:
        print(f"[NOTE] Excel export skipped ({exc}). CSV is fully ready.")
        
    print(f"\n[SUMMARY] Total Samples: {len(df)}")
    print(f"          Average Satellite LST: {df['lst'].mean():.2f}°C")
    print(f"          Max Hotspot: {df['lst'].max():.2f}°C")
    print(f"          Min Temp: {df['lst'].min():.2f}°C")
    print("\nNext step: Run 'python train_model.py' to train on this real dataset!")
    return df

if __name__ == "__main__":
    asyncio.run(build_dataset())
