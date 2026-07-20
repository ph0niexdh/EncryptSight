import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from .schema_detect import CICIDS_FEATURES_REQUIRED

# Global booster reference
booster = None

def _model_path_for_lightgbm(model_path: str) -> str:
    """Stage model files under an ASCII-safe path for LightGBM on Windows."""
    staged_dir = os.path.join(tempfile.gettempdir(), "encryptsight-models")
    os.makedirs(staged_dir, exist_ok=True)
    staged_path = os.path.join(staged_dir, os.path.basename(model_path))
    shutil.copy2(model_path, staged_path)
    return staged_path

def init_cicids_model(data_dir: str) -> bool:
    """Loads the CICIDS LightGBM booster from disk if it exists."""
    global booster
    model_path = os.path.join(data_dir, "models", "cicids_lgbm_model.txt")
    if os.path.exists(model_path):
        try:
            booster = lgb.Booster(model_file=_model_path_for_lightgbm(model_path))
            return True
        except Exception:
            booster = None
            return False
    return False

def is_cicids_model_loaded() -> bool:
    """Returns True if the CICIDS booster is loaded and ready for inference."""
    return booster is not None

def preprocess_cicids_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a raw pandas DataFrame into the shape and encoding the CICIDS booster expects."""
    df = df.copy()
    
    # Strip column names whitespace
    df.columns = [col.strip() for col in df.columns]
    
    # Convert features to numeric, mapping non-numeric values to NaN
    for col in CICIDS_FEATURES_REQUIRED:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0.0
            
    # Select only the required columns in the exact order
    df = df[CICIDS_FEATURES_REQUIRED]
    
    # Handle NaN and Infinity values defensively (important for CICIDS data)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    
    return df

def predict_cicids_batch(df: pd.DataFrame):
    """
    Runs batch inference on a DataFrame of CICIDS network flows.
    Returns:
        predicted_labels: List[int] (0 = Benign, 1 = Attack)
        predicted_cats: List[str] ("Benign" or "Attack")
        confidences: List[float] (probability of predicted class, 0.0 to 1.0)
        probs_2d: np.ndarray (shape: N x 2, class probabilities)
    """
    global booster
    if booster is None:
        raise ValueError("CICIDS2017 model is not loaded. Please train the model first.")
        
    preprocessed_df = preprocess_cicids_df(df)
    
    # LightGBM binary prediction yields the probability of the positive class (class 1, Attack)
    probs_class_1 = booster.predict(preprocessed_df)
    
    predicted_labels = (probs_class_1 > 0.5).astype(int).tolist()
    predicted_cats = ["Attack" if label == 1 else "Benign" for label in predicted_labels]
    
    # Calculate confidence as the probability of the predicted class
    confidences = [
        float(p) if label == 1 else float(1.0 - p) 
        for label, p in zip(predicted_labels, probs_class_1)
    ]
    
    # Construct a 2D array of class probabilities (class 0: Benign, class 1: Attack) for SHAP explainability
    probs_2d = np.vstack([1.0 - probs_class_1, probs_class_1]).T
    
    return predicted_labels, predicted_cats, confidences, probs_2d
