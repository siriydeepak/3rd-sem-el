import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=350, output_path='data/synthetic_heat_data.csv'):
    np.random.seed(42)
    
    sample_id = np.arange(1, num_samples + 1)
    
    # Generate base features with realistic distributions
    vegetation_pct = np.random.uniform(0, 60, num_samples)
    pavement_pct = np.random.uniform(10, 80, num_samples)
    
    # Ensure they don't exceed 100% easily
    total_cover = vegetation_pct + pavement_pct
    correction = np.where(total_cover > 95, 95 / total_cover, 1.0)
    vegetation_pct *= correction
    pavement_pct *= correction
    
    building_pct = 100 - (vegetation_pct + pavement_pct)
    building_pct = np.clip(building_pct, 5, 80)
    
    tree_canopy_pct = vegetation_pct * np.random.uniform(0.3, 0.9, num_samples)
    
    ndvi = (vegetation_pct / 100) * 0.8 + np.random.normal(0, 0.05, num_samples)
    ndvi = np.clip(ndvi, -0.2, 0.9)
    
    ndbi = (building_pct / 100) * 0.6 + (pavement_pct / 100) * 0.4 + np.random.normal(0, 0.1, num_samples)
    ndbi = np.clip(ndbi, -0.5, 0.8)
    
    building_height_m = np.random.exponential(10, num_samples) + 3
    
    sky_view_factor = 1.0 - (building_pct / 100) * 0.6 - (tree_canopy_pct / 100) * 0.3
    sky_view_factor = np.clip(sky_view_factor, 0.1, 1.0)
    
    solar_exposure = sky_view_factor * np.random.uniform(0.7, 1.0, num_samples)
    
    air_temperature = np.random.normal(30, 3, num_samples) # Base temp 30C
    humidity = np.random.normal(50, 10, num_samples)
    
    # Generate LST based on relationships
    # Higher vegetation/tree canopy -> lower LST
    # Higher pavement/ndbi -> higher LST
    # Higher solar exposure -> higher LST
    # Higher air temp -> higher LST
    lst = (
        air_temperature * 1.1
        - vegetation_pct * 0.1
        - tree_canopy_pct * 0.15
        + pavement_pct * 0.08
        + building_pct * 0.05
        - ndvi * 5.0
        + ndbi * 5.0
        + solar_exposure * 3.0
        + np.random.normal(0, 1.5, num_samples) # Noise
    )
    
    df = pd.DataFrame({
        'sample_id': sample_id,
        'vegetation_pct': vegetation_pct,
        'pavement_pct': pavement_pct,
        'building_pct': building_pct,
        'tree_canopy_pct': tree_canopy_pct,
        'ndvi': ndvi,
        'ndbi': ndbi,
        'building_height_m': building_height_m,
        'sky_view_factor': sky_view_factor,
        'solar_exposure': solar_exposure,
        'air_temperature': air_temperature,
        'humidity': humidity,
        'lst': lst
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset generated with {num_samples} samples at {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
