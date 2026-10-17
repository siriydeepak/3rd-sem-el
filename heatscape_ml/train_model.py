import os
from src.data_generator import generate_synthetic_data
from src.preprocessing import load_data, prepare_features_target, split_data
from src.model import train_model, save_model
from src.evaluation import evaluate_model

def main():
    print("========================================")
    print("   HeatScape ML Training Pipeline       ")
    print("   Member 3: Brain & Thermal Modeling   ")
    print("========================================")
    
    # 1. Dataset Selection (Real vs Synthetic Prototype)
    real_dataset_path = 'data/heatscape_dataset.csv'
    synthetic_dataset_path = 'data/synthetic_heat_data.csv'
    
    if os.path.exists(real_dataset_path):
        data_path = real_dataset_path
        print(f"[DATASET] Found real field dataset: {real_dataset_path}")
    else:
        data_path = synthetic_dataset_path
        print(f"[DATASET] Real dataset not found at {real_dataset_path}.")
        print(f"[DATASET] Generating calibrated baseline dataset at {synthetic_dataset_path}...")
        generate_synthetic_data(num_samples=500, output_path=data_path)
    
    # 2. Preprocess Data
    df = load_data(data_path)
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"[PREPROCESS] Split {len(df)} samples into Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 3. Train Model
    print("[TRAINING] Training Random Forest Regressor (200 trees)...")
    model = train_model(X_train, y_train)
    
    # 4. Evaluate Model
    metrics = evaluate_model(model, X_test, y_test)
    print("----------------------------------------")
    print(f"  Model R^2 Score : {metrics['R2']:.4f} (Variance explained)")
    print(f"  Mean Absolute Err: {metrics['MAE']:.3f} °C")
    print(f"  Root Mean Sq Err : {metrics['RMSE']:.3f} °C")
    print("----------------------------------------")
    
    # Feature Importance Rankings
    importances = model.feature_importances_
    features = list(X.columns)
    ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print("[FEATURES] Top Driving Thermal Features:")
    for feat, imp in ranked[:5]:
        print(f"   • {feat:20s}: {imp * 100:.1f}% importance")
    
    # 5. Save Model
    save_model(model, 'models/rf_model.pkl')
    print("[SUCCESS] Model trained & saved to models/rf_model.pkl")

if __name__ == "__main__":
    main()
