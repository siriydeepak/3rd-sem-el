
import rasterio
import numpy as np

path = "data/HeatScape_Cloud_Mask.tif"

with rasterio.open(path) as raster:
    data = raster.read(1, masked=True)
    values, counts = np.unique(data.compressed(), return_counts=True)

    print("Cloud mask unique values and counts:")
    for value, count in zip(values, counts):
        print(value, ":", count)

    print("\nRaster dimensions:", raster.width, "x", raster.height)
