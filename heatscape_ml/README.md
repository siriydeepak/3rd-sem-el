# HeatScape ML Prototype

HeatScape is an AI/ML-based urban heat mitigation system. This prototype demonstrates a proof-of-concept machine learning pipeline that uses urban morphological features to estimate localized Land Surface Temperature (LST) and simulate the effects of cooling interventions.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Generate data and train the model**:
   ```bash
   python train_model.py
   ```

2. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

## Future Integration
This prototype uses synthetic data for demonstration. In the complete HeatScape pipeline, the workflow will be:
- **Data Collection**: RGB images will be processed using TRELLIS for 3D reconstruction and semantic segmentation.
- **Feature Extraction**: Real urban morphology features (vegetation %, building height, sky view factor) will be extracted.
- **Thermal Data Alignment**: Features will be spatially aligned with Landsat/ECOSTRESS/Bhuvan/IMD thermal data.
- **Model Application**: The trained ML model will generate high-resolution heat exposure maps.
- **Optimization**: Intervention simulations will be run at scale to optimize cooling strategies.
- **Visualization**: 3D before/after visualizations of interventions.
