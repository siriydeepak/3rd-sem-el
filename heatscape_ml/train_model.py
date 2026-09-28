import os
from src.data_generator import generate_synthetic_data
from src.preprocessing import load_data, prepare_features_target, split_data
from src.model import train_model, save_model

def main():
    print("--- HeatScape ML Pipeline ---")
    
    # 1. Generate Data
    data_path = 'data/synthetic_heat_data.csv'
    generate_synthetic_data(num_samples=400, output_path=data_path)
    
    # 2. Preprocess Data
    df = load_data(data_path)
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Data split into train ({len(X_train)}) and test ({len(X_test)}) sets.")
    
    # 3. Train Model
    print("Training Random Forest model (200 estimators)...")
    model = train_model(X_train, y_train)
    
    # 4. Save Model
    save_model(model, 'models/rf_model.pkl')
    print("Pipeline complete. You can now run the Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    main()
