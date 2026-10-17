"""
EXIF Metadata Extraction Service for HeatScape
Extracts GPS Coordinates, Capture Timestamp, and Device info directly from image headers.
"""

import io
import logging
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

logger = logging.getLogger(__name__)

# Default fallback coordinates (e.g. Urban University Campus in Bengaluru)
DEFAULT_LAT = 12.9344
DEFAULT_LON = 77.5345

def _convert_dms_to_decimal(dms_values, ref: str) -> float | None:
    """
    Converts GPS coordinates stored in Degrees, Minutes, Seconds to decimal format.
    Handles rational tuples / floats returned by PIL.
    """
    try:
        def to_float(val):
            if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                return float(val.numerator) / float(val.denominator) if val.denominator != 0 else 0.0
            if isinstance(val, (tuple, list)) and len(val) == 2:
                return float(val[0]) / float(val[1]) if val[1] != 0 else 0.0
            return float(val)

        d = to_float(dms_values[0])
        m = to_float(dms_values[1])
        s = to_float(dms_values[2])
        dec = d + (m / 60.0) + (s / 3600.0)

        if ref.upper() in ["S", "W"]:
            dec = -dec
        return round(dec, 6)
    except Exception as e:
        logger.warning("Error converting DMS to decimal: %s", e)
        return None

def extract_exif_metadata(image_bytes: bytes) -> dict:
    """
    Parses EXIF binary data from raw image bytes.
    Extracts GPS latitude, longitude, altitude, and capture timestamp.
    Provides graceful fallback if EXIF is missing or stripped by messaging apps.
    """
    metadata = {
        "has_gps": False,
        "latitude": DEFAULT_LAT,
        "longitude": DEFAULT_LON,
        "altitude_m": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date_str": datetime.now().strftime("%Y-%m-%d"),
        "hour": datetime.now().hour,
        "camera_make": None,
        "camera_model": None,
        "source": "fallback",
        "notes": "No EXIF GPS tags found. Using default campus coordinates."
    }

    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif = img._getexif()

        if not exif:
            return metadata

        labeled_exif = {}
        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, tag_id)
            labeled_exif[tag_name] = value

        # 1. Device Info
        metadata["camera_make"] = labeled_exif.get("Make")
        metadata["camera_model"] = labeled_exif.get("Model")

        # 2. Timestamp Extraction
        raw_datetime = (
            labeled_exif.get("DateTimeOriginal")
            or labeled_exif.get("DateTimeDigitized")
            or labeled_exif.get("DateTime")
        )

        if raw_datetime and isinstance(raw_datetime, str):
            try:
                # Format: "YYYY:MM:DD HH:MM:SS"
                clean_dt = raw_datetime.replace("-", ":")
                dt = datetime.strptime(clean_dt.strip(), "%Y:%m:%d %H:%M:%S")
                metadata["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                metadata["date_str"] = dt.strftime("%Y-%m-%d")
                metadata["hour"] = dt.hour
            except Exception as dt_err:
                logger.debug("Failed parsing date '%s': %s", raw_datetime, dt_err)

        # 3. GPS Coordinates Extraction
        gps_info = labeled_exif.get("GPSInfo")
        if gps_info:
            gps_tags = {}
            for t, val in gps_info.items():
                tag_name = GPSTAGS.get(t, t)
                gps_tags[tag_name] = val

            lat_val = gps_tags.get("GPSLatitude")
            lat_ref = gps_tags.get("GPSLatitudeRef", "N")
            lon_val = gps_tags.get("GPSLongitude")
            lon_ref = gps_tags.get("GPSLongitudeRef", "E")

            if lat_val and lon_val:
                lat = _convert_dms_to_decimal(lat_val, lat_ref)
                lon = _convert_dms_to_decimal(lon_val, lon_ref)

                if lat is not None and lon is not None:
                    metadata["has_gps"] = True
                    metadata["latitude"] = lat
                    metadata["longitude"] = lon
                    metadata["source"] = "exif"
                    metadata["notes"] = "Exact coordinates extracted from photo EXIF tags."

            # Altitude
            alt_val = gps_tags.get("GPSAltitude")
            if alt_val is not None:
                try:
                    if hasattr(alt_val, 'numerator') and hasattr(alt_val, 'denominator'):
                        metadata["altitude_m"] = round(float(alt_val.numerator) / float(alt_val.denominator), 1)
                    else:
                        metadata["altitude_m"] = round(float(alt_val), 1)
                except Exception:
                    pass

    except Exception as exc:
        logger.warning("EXIF extraction encountered error: %s", exc)

    return metadata
