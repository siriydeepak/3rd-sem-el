import shap
import matplotlib.pyplot as plt

def get_shap_values(model, X):
    # Using TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

def plot_shap_summary(shap_values, X):
    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    return fig
