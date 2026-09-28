import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from src.preprocessing import load_data, prepare_features_target, split_data
from src.model import load_model
from src.evaluation import evaluate_model, plot_actual_vs_predicted, plot_feature_importance
from src.explainability import get_shap_values, plot_shap_summary
from src.interventions import predict_intervention

st.set_page_config(page_title="HeatScape Prototype", layout="wide")

st.title("🌡️ HeatScape – ML Heat Exposure Prototype")

st.header("1. Project Overview")
st.write("""
HeatScape is an AI/ML-based urban heat mitigation system. This prototype demonstrates a proof-of-concept machine learning pipeline that uses urban morphological features, vegetation indices, and environmental factors to estimate localized Land Surface Temperature (LST) and simulate the effects of cooling interventions.
""")

@st.cache_data
def get_data():
    if not os.path.exists('data/synthetic_heat_data.csv'):
        st.error("Dataset not found. Please run `python train_model.py` first.")
        st.stop()
    df = load_data('data/synthetic_heat_data.csv')
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return df, X, y, X_train, X_test, y_train, y_test

@st.cache_resource
def get_trained_model():
    if not os.path.exists('models/rf_model.pkl'):
        st.error("Model not found. Please run `python train_model.py` first.")
        st.stop()
    return load_model('models/rf_model.pkl')

try:
    df, X, y, X_train, X_test, y_train, y_test = get_data()
    model = get_trained_model()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

st.header("2. Dataset")
st.warning("⚠️ PROTOTYPE DATA: The dataset currently displayed is SYNTHETIC and generated for proof-of-concept modeling. It demonstrates expected physical relationships but is not real satellite data.")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Samples", len(df))
with col2:
    st.write("**Features Used:**", ", ".join(X.columns))

st.write("Data Preview:")
st.dataframe(df.head())

st.header("3. Model Architecture")
st.write("- **Algorithm**: Random Forest Regressor")
st.write("- **Estimators (Trees)**: 200")
st.write(f"- **Train/Test Split**: {len(X_train)} train / {len(X_test)} test")

st.header("4. Model Performance")
metrics = evaluate_model(model, X_test, y_test)

col1, col2, col3 = st.columns(3)
col1.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:.2f}")
col2.metric("RMSE (Root Mean Squared Error)", f"{metrics['RMSE']:.2f}")
col3.metric("R² Score", f"{metrics['R2']:.3f}")

col4, col5 = st.columns(2)
with col4:
    st.write("**Actual vs Predicted Estimated LST**")
    fig_scatter = plot_actual_vs_predicted(y_test, metrics['predictions'])
    st.pyplot(fig_scatter)
    
st.header("5. Feature Importance")
with st.spinner("Generating feature importance plot..."):
    fig_importance = plot_feature_importance(model, X.columns)
    st.pyplot(fig_importance)

st.header("6. Explainability (SHAP)")
with st.expander("View SHAP Analysis"):
    st.write("SHAP (SHapley Additive exPlanations) shows how each feature contributes to pushing the model output from the base value to the final prediction.")
    with st.spinner("Calculating SHAP values..."):
        try:
            X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
            explainer, shap_values = get_shap_values(model, X_sample)
            fig_shap = plot_shap_summary(shap_values, X_sample)
            st.pyplot(fig_shap)
        except Exception as e:
            st.error(f"Could not generate SHAP plots. Error: {e}")

st.header("7. Intervention Simulator (What-If Scenarios)")
st.info("Select a baseline scenario (a sample from the test set) and apply an urban cooling intervention to see the estimated change in heat exposure.")

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    sample_idx = st.selectbox("Select a Sample Baseline (Index)", X_test.index[:20])
    baseline_features = X_test.loc[sample_idx]
    
    intervention_choice = st.radio(
        "Choose Intervention:",
        ["Baseline (None)", "Add Trees", "Cool Pavement", "Shade Structure", "Combined"]
    )

with col_sim2:
    if intervention_choice == "Baseline (None)":
        pred_lst = model.predict(baseline_features.to_frame().T)[0]
        st.metric("Estimated Baseline LST", f"{pred_lst:.2f} °C")
        st.write("**Baseline Features:**")
        st.dataframe(baseline_features.to_frame(name="Value"))
    else:
        results = predict_intervention(model, baseline_features, intervention_choice)
        
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Baseline LST", f"{results['baseline_pred']:.2f} °C")
        mcol2.metric("Intervention LST", f"{results['intervention_pred']:.2f} °C", 
                     delta=f"{results['difference']:.2f} °C", delta_color="inverse")
        mcol3.metric("Reduction %", f"{abs(results['pct_change']):.1f} %")
        
        st.write("⚠️ *MODEL ESTIMATES ONLY: This demonstrates relative heat reduction potential, not guaranteed real-world temperature drops.*")
        
        comp_df = pd.DataFrame({
            'Baseline': baseline_features,
            'Post-Intervention': results['modified_features']
        })
        comp_df['Change'] = comp_df['Post-Intervention'] - comp_df['Baseline']
        
        st.write("**Feature Modifications:**")
        st.dataframe(comp_df[comp_df['Change'] != 0])
