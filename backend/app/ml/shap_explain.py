import shap
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from . import unsw_model, cicids_model

# Global explainer caches to avoid rebuilding them repeatedly
unsw_explainer = None
cicids_explainer = None

def get_unsw_explainer():
    """Initializes and returns the TreeExplainer for the UNSW-NB15 model."""
    global unsw_explainer
    if unsw_explainer is None:
        if unsw_model.booster is None:
            raise ValueError("UNSW booster is not loaded.")
        unsw_explainer = shap.TreeExplainer(unsw_model.booster)
    return unsw_explainer

def get_cicids_explainer():
    """Initializes and returns the TreeExplainer for the CICIDS2017 model."""
    global cicids_explainer
    if cicids_explainer is None:
        if cicids_model.booster is None:
            raise ValueError("CICIDS2017 booster is not loaded/trained.")
        cicids_explainer = shap.TreeExplainer(cicids_model.booster)
    return cicids_explainer

def extract_shap_1d(shap_vals: Any, class_idx: int) -> np.ndarray:
    """
    Extracts a 1D array of feature SHAP values for a single flow sample.
    Handles different shapes returned by shap.TreeExplainer across different SHAP versions.
    """
    if isinstance(shap_vals, list):
        # List of numpy arrays, e.g., one array per class
        # Each array has shape: (samples, features)
        arr = shap_vals[class_idx]
        if len(arr.shape) == 2:
            return arr[0]
        return arr
        
    elif isinstance(shap_vals, np.ndarray):
        if len(shap_vals.shape) == 3:
            # Shape is either (samples, features, classes) or (classes, samples, features)
            # Typically (samples, features, classes)
            if shap_vals.shape[0] == 1:
                return shap_vals[0, :, class_idx]
            else:
                return shap_vals[:, 0, class_idx]
        elif len(shap_vals.shape) == 2:
            # For binary classification, sometimes shape is (samples, features) representing class 1
            return shap_vals[0]
            
    raise ValueError(f"Unknown SHAP values shape/type: {type(shap_vals)}")

def explain_unsw_flow(flow_dict: Dict[str, Any], predicted_class_name: str) -> List[Dict[str, Any]]:
    """
    Computes SHAP explanation values for a single UNSW-NB15 flow.
    Returns list of dicts with feature names, raw values, and SHAP values.
    """
    explainer = get_unsw_explainer()
    
    # Preprocess the raw features into model feature order
    df_raw = pd.DataFrame([flow_dict])
    df_preprocessed = unsw_model.preprocess_unsw_df(df_raw)
    
    # Compute SHAP values
    shap_vals = explainer.shap_values(df_preprocessed)
    
    # Get index of predicted class
    class_idx = int(unsw_model.label_encoder.transform([predicted_class_name])[0])
    
    # Extract 1D shap values
    shap_1d = extract_shap_1d(shap_vals, class_idx)
    
    # Pair with features and values
    explanation = []
    for feat in unsw_model.ALL_MODEL_FEATURES:
        feat_idx = unsw_model.ALL_MODEL_FEATURES.index(feat)
        explanation.append({
            "feature_name": feat,
            "feature_value": float(df_preprocessed.iloc[0][feat]),
            "shap_value": float(shap_1d[feat_idx])
        })
        
    # Sort by absolute SHAP value descending
    explanation = sorted(explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
    return explanation

def explain_cicids_flow(flow_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Computes SHAP explanation values for a single CICIDS2017 flow.
    Returns list of dicts with feature names, raw values, and SHAP values.
    """
    explainer = get_cicids_explainer()
    
    # Preprocess
    df_raw = pd.DataFrame([flow_dict])
    df_preprocessed = cicids_model.preprocess_cicids_df(df_raw)
    
    # Compute SHAP values
    shap_vals = explainer.shap_values(df_preprocessed)
    
    # Binary model: class 1 is Attack, class 0 is Benign.
    # We explain relative to class 1 (Attack) so positive SHAP means pushing towards Attack.
    shap_1d = extract_shap_1d(shap_vals, 1)
    
    explanation = []
    for feat in cicids_model.CICIDS_FEATURES_REQUIRED:
        feat_idx = cicids_model.CICIDS_FEATURES_REQUIRED.index(feat)
        explanation.append({
            "feature_name": feat,
            "feature_value": float(df_preprocessed.iloc[0][feat]),
            "shap_value": float(shap_1d[feat_idx])
        })
        
    # Sort by absolute SHAP value descending
    explanation = sorted(explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
    return explanation
