import pandas as pd

def apply_intervention(baseline_features, intervention_type):
    """
    Apply urban cooling interventions to a baseline feature set.
    """
    modified = baseline_features.copy()
    
    if intervention_type == "Add Trees":
        modified['vegetation_pct'] = min(modified['vegetation_pct'] + 15, 100)
        modified['tree_canopy_pct'] = min(modified['tree_canopy_pct'] + 13, modified['vegetation_pct'])
        modified['ndvi'] = min(modified['ndvi'] + 0.2, 0.9)
        modified['pavement_pct'] = max(modified['pavement_pct'] - 15, 0)
        
    elif intervention_type == "Cool Pavement":
        modified['ndbi'] = max(modified['ndbi'] - 0.15, -0.5)
        
    elif intervention_type == "Shade Structure":
        modified['solar_exposure'] = max(modified['solar_exposure'] - 0.3, 0.1)
        modified['sky_view_factor'] = max(modified['sky_view_factor'] - 0.2, 0.1)
        
    elif intervention_type == "Combined":
        modified['vegetation_pct'] = min(modified['vegetation_pct'] + 15, 100)
        modified['tree_canopy_pct'] = min(modified['tree_canopy_pct'] + 13, modified['vegetation_pct'])
        modified['ndvi'] = min(modified['ndvi'] + 0.2, 0.9)
        modified['pavement_pct'] = max(modified['pavement_pct'] - 15, 0)
        modified['ndbi'] = max(modified['ndbi'] - 0.15, -0.5)
        modified['solar_exposure'] = max(modified['solar_exposure'] - 0.2, 0.1)
        modified['sky_view_factor'] = max(modified['sky_view_factor'] - 0.1, 0.1)
        
    return modified

def predict_intervention(model, baseline_features, intervention_type):
    if isinstance(baseline_features, pd.Series):
        baseline_features = baseline_features.to_frame().T
        
    modified_features = apply_intervention(baseline_features.iloc[0], intervention_type).to_frame().T
    
    baseline_pred = model.predict(baseline_features)[0]
    intervention_pred = model.predict(modified_features)[0]
    
    return {
        'baseline_pred': baseline_pred,
        'intervention_pred': intervention_pred,
        'difference': intervention_pred - baseline_pred,
        'pct_change': ((intervention_pred - baseline_pred) / baseline_pred) * 100,
        'modified_features': modified_features.iloc[0]
    }
