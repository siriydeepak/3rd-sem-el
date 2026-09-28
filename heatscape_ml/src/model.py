from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath='models/rf_model.pkl'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath='models/rf_model.pkl'):
    return joblib.load(filepath)
