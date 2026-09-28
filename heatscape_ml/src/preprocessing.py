import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath='data/synthetic_heat_data.csv'):
    df = pd.read_csv(filepath)
    return df

def prepare_features_target(df):
    features = [
        'vegetation_pct', 'pavement_pct', 'building_pct', 
        'tree_canopy_pct', 'ndvi', 'ndbi', 'building_height_m', 
        'sky_view_factor', 'solar_exposure', 'air_temperature', 'humidity'
    ]
    X = df[features]
    y = df['lst']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
