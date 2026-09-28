import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'predictions': predictions
    }

def plot_actual_vs_predicted(y_test, predictions):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions, alpha=0.7, color='b')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual LST')
    ax.set_ylabel('Predicted LST')
    ax.set_title('Actual vs Predicted LST')
    return fig

def plot_residuals(y_test, predictions):
    residuals = y_test - predictions
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel('Residuals')
    ax.set_title('Residuals Distribution')
    return fig

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], align="center")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title("Feature Importances (Random Forest)")
    fig.tight_layout()
    return fig
