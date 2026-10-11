# HeatScape Feature Extraction Pipeline — Setup & Usage Guide

## 1. What This Folder Contains

```
feature_extraction/
├── sam2_segmenter.py       ← Step 1: SAM 2 automatic segmentation
├── clip_labeller.py        ← Step 2: CLIP semantic labelling of masks
├── feature_calculator.py   ← Step 3: Numerical feature extraction
├── run_pipeline.py         ← Step 4: Batch orchestrator + dataset merger
├── requirements.txt        ← Python dependencies
├── checkpoints/            ← (you create this) Place SAM2 model weights here
│   └── sam2_hiera_tiny.pt
└── README.md               ← This file
```

---

## 2. First-Time Setup

### 2a. Create a virtual environment
```powershell
cd "D:\RVCE\Sem 3\Main EL\HeatScape\backend\feature_extraction"
python -m venv venv
.\venv\Scripts\activate
```

### 2b. Install PyTorch (CPU version — works on any machine)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **If you have an NVIDIA GPU (recommended for speed):**
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2c. Install SAM 2
```powershell
pip install git+https://github.com/facebookresearch/sam2.git
```

### 2d. Install remaining requirements
```powershell
pip install -r requirements.txt
```

### 2e. Download the SAM 2 checkpoint
Create a `checkpoints/` folder and download the tiny model (fastest):

```powershell
mkdir checkpoints
# Download from:
# https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
# Place it at: checkpoints/sam2_hiera_tiny.pt
```

---

## 3. Running the Pipeline

### Single image (testing/development)
```powershell
# Step 1: Segment the image
python sam2_segmenter.py --image data/test_image.jpg --output output/masks

# Step 2: Label the segments
python clip_labeller.py --image data/test_image.jpg --masks output/masks/masks.npz --output output/labels

# Step 3: Extract features
python feature_calculator.py --image data/test_image.jpg --labels output/labels/labelled_masks.npz --output output/features
```

### All three steps at once (single image)
```powershell
python run_pipeline.py --image data/test_image.jpg --output output/
```

### Batch mode (for pre-training dataset)
```powershell
python run_pipeline.py --batch data/images/ --output output/batch/
```

### Batch with thermal merge (final integrated dataset)
```powershell
python run_pipeline.py \
    --batch   data/images/ \
    --thermal data/thermal.csv \
    --output  output/batch/
```

---

## 4. What the Output Looks Like

After running on 1 image, you get a `features.json` like this:

```json
{
  "vegetation_pct": 23.4,
  "pavement_pct": 45.1,
  "building_pct": 18.7,
  "tree_canopy_pct": 23.4,
  "ndvi": 0.12,
  "ndbi": 0.08,
  "building_height_m": 0.82,
  "sky_view_factor": 0.74,
  "solar_exposure": 0.91,
  "water_pct": 0.5,
  "bare_soil_pct": 8.2,
  "shadow_pct": 4.1,
  "lst": null,
  "air_temperature": null,
  "humidity": null
}
```

The `lst`, `air_temperature`, and `humidity` fields are null until
Member 2 provides their thermal CSV.

---

## 5. Member 2's Thermal CSV Format

Member 2 should provide a CSV with these columns so that
`run_pipeline.py --thermal` can merge everything:

| image_name | lat | lon | lst | air_temperature | humidity | ndvi | ndbi |
|---|---|---|---|---|---|---|---|
| test_image | 12.9716 | 77.5946 | 38.2 | 32.1 | 65 | 0.18 | 0.11 |
| image_2    | 12.9750 | 77.5980 | 41.5 | 33.4 | 60 | 0.09 | 0.22 |

The `image_name` column must match the filename (without extension).

---

## 6. Recommended Open-Source Datasets for Pre-Training

| Dataset | What it gives you | Where to get it |
|---|---|---|
| **So2Sat LCZ42** | 400k urban image patches, 42 cities | [IEEE DataPort](https://ieee-dataport.org/open-access/so2sat-lcz42) |
| **SpaceNet Buildings** | Sub-meter RGB + building footprints | [AWS Open Data](https://spacenet.ai/datasets/) |
| **NAIP** (US) | 60cm aerial imagery | Google Earth Engine |
| **Mapillary Vistas** | Street-level segmented images | [mapillary.com](https://www.mapillary.com/dataset/vistas) |

For thermal ground truth, use **Google Earth Engine** to pull
Landsat 8/9 Band 10 LST for the matching coordinates.
