import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Numerical feature columns used by the LightGBM booster
FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
    'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'dwin', 'tcprtt', 'synack',
    'ackdat', 'smean', 'dmean', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_src_ltm', 'ct_srv_dst'
]

# Categorical feature columns used by the LightGBM booster
CAT_COLS = ['proto', 'service', 'state']

# Combine columns in the exact order the booster expects
ALL_MODEL_FEATURES = FEATURES + CAT_COLS

# Global booster and encoders
booster = None
label_encoder = None
cat_encoders = {}
cached_metrics = {}

def _model_path_for_lightgbm(model_path: str) -> str:
    """Stage the model under an ASCII-safe temporary path on Windows.

    LightGBM's native loader cannot open model paths containing some Unicode
    characters (for example the Korean OneDrive folder used in development).
    Python can read the source file normally; the staged copy lets the native
    loader work in both local and packaged deployments.
    """
    staged_dir = os.path.join(tempfile.gettempdir(), "encryptsight-models")
    os.makedirs(staged_dir, exist_ok=True)
    staged_path = os.path.join(staged_dir, os.path.basename(model_path))
    shutil.copy2(model_path, staged_path)
    return staged_path

def init_unsw_model(data_dir: str):
    """Loads the LightGBM booster and encoders, and precomputes test-set metrics."""
    global booster, label_encoder, cat_encoders, cached_metrics
    
    model_path = os.path.join(data_dir, "models", "etf_lgbm_model.txt")
    
    # Check that assets exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"UNSW booster model file not found at {model_path}")
        
    # Load LightGBM booster
    booster = lgb.Booster(model_file=_model_path_for_lightgbm(model_path))
    
    # Fit encoders on the shipped train/test data. Rebuilding the target
    # LabelEncoder keeps its deterministic lexical class ordering while avoiding
    # pickle incompatibilities between Python/scikit-learn versions.
    train_path = os.path.join(data_dir, "unsw", "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "unsw", "UNSW_NB15_testing-set.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Raw UNSW train/test CSV datasets are missing under {data_dir}/unsw/")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Strip spaces from columns
    train_df.columns = [col.strip() for col in train_df.columns]
    test_df.columns = [col.strip() for col in test_df.columns]

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train_df["attack_cat"].astype(str), test_df["attack_cat"].astype(str)]))
    
    for col in CAT_COLS:
        le_col = LabelEncoder()
        # Convert to string to avoid mixed-type comparison issues
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
        le_col.fit(combined)
        cat_encoders[col] = le_col
        
    # Compute live metrics on the test set for the Model Info dashboard page
    compute_test_metrics(test_df)

def preprocess_unsw_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a raw pandas DataFrame into the shape and encoding the UNSW model expects."""
    df = df.copy()
    # Strip column names whitespace
    df.columns = [col.strip() for col in df.columns]
    
    # Encode categorical columns
    for col in CAT_COLS:
        if col in df.columns:
            known_classes = set(cat_encoders[col].classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: cat_encoders[col].transform([x])[0] if x in known_classes else 0
            )
        else:
            df[col] = 0
            
    # Ensure numerical columns are present and numeric
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
            
    # Reindex and select only the 34 features in the correct order
    return df[ALL_MODEL_FEATURES]

def predict_unsw_batch(df: pd.DataFrame):
    """
    Runs batch inference on a DataFrame of network flows.
    Returns:
        predicted_labels: List[int] (0 = Benign, 1 = Attack)
        predicted_cats: List[str] (e.g. 'DoS', 'Normal', etc.)
        confidences: List[float] (probability of prediction, 0.0 to 1.0)
        probs: np.ndarray (shape: N x 10, raw probabilities for all classes)
    """
    preprocessed_df = preprocess_unsw_df(df)
    
    # Run predictions
    probs = booster.predict(preprocessed_df)
    pred_classes = np.argmax(probs, axis=1)
    
    predicted_cats = label_encoder.inverse_transform(pred_classes)
    predicted_cats = [cat.strip() for cat in predicted_cats]
    
    # Map predictions to binary label (0 = Normal/Benign, 1 = Attack category)
    predicted_labels = [0 if cat == "Normal" else 1 for cat in predicted_cats]
    confidences = [float(probs[i][pred_classes[i]]) for i in range(len(df))]
    
    return predicted_labels, predicted_cats, confidences, probs

def compute_test_metrics(test_df: pd.DataFrame):
    """Computes confusion matrix, classification report, and feature importances live."""
    global cached_metrics
    
    test_preprocessed = preprocess_unsw_df(test_df)
    
    # Get predictions
    probs = booster.predict(test_preprocessed)
    pred_classes = np.argmax(probs, axis=1)
    
    # Ground truth
    true_cats = test_df['attack_cat'].str.strip()
    true_classes = label_encoder.transform(true_cats)
    
    # Metrics calculations
    accuracy = float(np.mean(pred_classes == true_classes))
    classes_list = list(label_encoder.classes_)
    
    cm = confusion_matrix(true_classes, pred_classes, labels=range(len(classes_list)))
    
    report = classification_report(
        true_classes,
        pred_classes,
        target_names=classes_list,
        output_dict=True,
        labels=range(len(classes_list)),
        zero_division=0
    )
    
    # Feature importance based on information gain
    importance_gain = booster.feature_importance(importance_type='gain')
    feature_importance_dict = dict(zip(ALL_MODEL_FEATURES, importance_gain.tolist()))
    
    # Sort feature importance descending
    sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    cached_metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classes": classes_list,
        "feature_importance": sorted_importance
    }
