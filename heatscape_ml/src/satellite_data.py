
import numpy as np
import rasterio


def load_satellite_lst(raster_path):
    """Load a land-surface-temperature GeoTIFF in Celsius."""
    with rasterio.open(raster_path) as src:
        lst = src.read(1, masked=True).astype("float32")
        metadata = {
            "crs": str(src.crs),
            "resolution_m": src.res[0],
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
        }

    return lst, metadata


def get_lst_statistics(lst):
    """Calculate statistics using valid raster pixels only."""
    values = np.asarray(lst.compressed())

    if values.size == 0:
        raise ValueError("The raster contains no valid temperature pixels.")

    return {
        "valid_pixels": int(values.size),
        "mean_c": float(np.mean(values)),
        "min_c": float(np.min(values)),
        "max_c": float(np.max(values)),
    }

