
import os
import numpy as np
import rasterio

files = [
    "HeatScape_Cloud_Mask.tif",
    "HeatScape_LST_CloudMasked.tif",
    "HeatScape_LST_CloudMasked_Final.tif",
]

for filename in files:
    path = os.path.join("data", filename)
    print(f"\n--- {filename} ---")

    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        values = np.asarray(arr.compressed(), dtype=np.float64)
        values = values[np.isfinite(values)]

        print("Valid pixels:", len(values))

        if len(values) == 0:
            print("No valid pixels found.")
            continue

        print("Minimum:", np.min(values))
        print("Maximum:", np.max(values))
        print("Mean:", np.mean(values))
        print("Percentiles:", np.percentile(values, [1, 25, 50, 75, 99]))

        if "Cloud_Mask" in filename:
            print("Sample unique values:", np.unique(values[:100000])[:20])
