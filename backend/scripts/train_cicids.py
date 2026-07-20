import os
import sys
import json
import logging
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.database import Base, SessionLocal, engine
from app.db.models import ModelVersion
from app.ml.schema_detect import CICIDS_FEATURES_REQUIRED

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_cicids")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CICIDS_DIR = os.path.join(DATA_DIR, "cicids")
MODEL_OUT_DIR = os.path.join(DATA_DIR, "models")
REPORT_PATH = os.path.join(DATA_DIR, "models", "cicids_sampling_report.json")

def train_cicids_model():
    logger.info("Starting CICIDS2017 model training pipeline...")
    
    # 1. Identify files
    if not os.path.exists(CICIDS_DIR):
        logger.error(f"CICIDS data directory does not exist: {CICIDS_DIR}")
        return
        
    csv_files = [f for f in os.listdir(CICIDS_DIR) if f.endswith(".csv")]
    if not csv_files:
        logger.error(f"No CSV files found in {CICIDS_DIR}")
        return
        
    logger.info(f"Found {len(csv_files)} day-files to process.")
    
    # We will sample equally from each file to reach ~200k rows in total
    sample_per_file = 200000 // len(csv_files)
    
    raw_class_counts = {}
    sampled_class_counts = {}
    
    sampled_dfs = []
    
    # Process each file individually to be memory-safe
    for idx, filename in enumerate(csv_files):
        filepath = os.path.join(CICIDS_DIR, filename)
        logger.info(f"Processing file {idx+1}/{len(csv_files)}: {filename}")
        
        # Load only the required columns to save memory
        try:
            # Try loading with utf-8 first, fallback to latin1 for non-standard chars
            try:
                df_temp = pd.read_csv(filepath, nrows=0, encoding='utf-8')
                encoding_used = 'utf-8'
            except UnicodeDecodeError:
                df_temp = pd.read_csv(filepath, nrows=0, encoding='latin1')
                encoding_used = 'latin1'
                
            cols_map = {col: col.strip() for col in df_temp.columns}
            
            # Find the true column name for 'Label'
            label_col = None
            for col in df_temp.columns:
                if col.strip().lower() == 'label':
                    label_col = col
                    break
            
            if not label_col:
                logger.warning(f"Label column not found in {filename}, skipping.")
                continue
                
            # Map required features to their actual raw column names
            usecols_raw = []
            for feat in CICIDS_FEATURES_REQUIRED:
                for col in df_temp.columns:
                    if col.strip().lower() == feat.lower():
                        usecols_raw.append(col)
                        break
            
            usecols_raw.append(label_col)
            
            # Read only these columns
            try:
                df = pd.read_csv(filepath, usecols=usecols_raw, encoding='utf-8')
            except UnicodeDecodeError:
                logger.info(f"UTF-8 decode failed for body of {filename}, falling back to Latin-1")
                df = pd.read_csv(filepath, usecols=usecols_raw, encoding='latin1')
            # Rename columns to their normalized trimmed names
            df = df.rename(columns=cols_map)
            # Standardize label column name to 'Label'
            df = df.rename(columns={label_col.strip(): 'Label'})
            
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            continue
            
        # Clean label values
        df['Label'] = df['Label'].astype(str).str.strip()
        
        # Log before-counts
        for label, count in df['Label'].value_counts().items():
            raw_class_counts[label] = raw_class_counts.get(label, 0) + int(count)
            
        # Map label to binary (BENIGN = 0, others = 1)
        df['target'] = df['Label'].apply(lambda x: 0 if x.upper() == 'BENIGN' else 1)
        
        # Handle NaN and Infinity values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=CICIDS_FEATURES_REQUIRED + ['target'], inplace=True)
        
        if len(df) == 0:
            logger.warning(f"No valid rows in {filename} after cleaning.")
            continue
            
        # Perform stratified sampling
        file_sample_size = min(len(df), sample_per_file)
        
        # Check class distribution
        target_counts = df['target'].value_counts()
        if len(target_counts) > 1 and file_sample_size < len(df):
            # Stratified sample
            df_sampled, _ = train_test_split(
                df, 
                train_size=file_sample_size, 
                stratify=df['target'], 
                random_state=42
            )
        else:
            # Random sample if only one class is present or no need to split
            df_sampled = df.sample(n=file_sample_size, random_state=42)
            
        # Record sampled counts (by original label)
        for label, count in df_sampled['Label'].value_counts().items():
            sampled_class_counts[label] = sampled_class_counts.get(label, 0) + int(count)
            
        # Convert numeric columns to float32 to reduce memory footprint
        for col in CICIDS_FEATURES_REQUIRED:
            df_sampled[col] = df_sampled[col].astype(np.float32)
            
        sampled_dfs.append(df_sampled[CICIDS_FEATURES_REQUIRED + ['target']])
        logger.info(f"Sampled {len(df_sampled)} rows from {filename}")
        
    if not sampled_dfs:
        logger.error("No data could be processed. Training aborted.")
        return
        
    # 2. Concatenate all samples
    logger.info("Concatenating sampled datasets...")
    full_df = pd.concat(sampled_dfs, ignore_index=True)
    logger.info(f"Final training set size: {full_df.shape}")
    
    # Log counts report
    logger.info("Class distribution BEFORE sampling:")
    for label, count in sorted(raw_class_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label:<40}: {count:,}")
        
    logger.info("Class distribution AFTER sampling:")
    for label, count in sorted(sampled_class_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label:<40}: {count:,}")
        
    # Save sampling report
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    report_data = {
        "raw_counts": raw_class_counts,
        "sampled_counts": sampled_class_counts,
        "total_raw_rows": sum(raw_class_counts.values()),
        "total_sampled_rows": len(full_df)
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"Saved sampling report to {REPORT_PATH}")
    
    # 3. Train-test split for verification
    X = full_df[CICIDS_FEATURES_REQUIRED]
    y = full_df['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 4. Train LightGBM Booster
    logger.info("Training LightGBM booster...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train booster
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = float(accuracy_score(y_val, y_pred))
    report = classification_report(y_val, y_pred, target_names=["Benign", "Attack"], output_dict=True)
    cm = confusion_matrix(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Validation Confusion Matrix:\n{cm}")
    
    # Compute feature importance based on gain
    importance_gain = model.feature_importance(importance_type='gain')
    feature_importance_dict = dict(zip(CICIDS_FEATURES_REQUIRED, importance_gain.tolist()))
    sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Pack metrics
    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classes": ["Benign", "Attack"],
        "feature_importance": sorted_importance
    }
    
    # Save booster model file
    model_output_path = os.path.join(MODEL_OUT_DIR, "cicids_lgbm_model.txt")
    model.save_model(model_output_path)
    logger.info(f"Model saved successfully to {model_output_path}")
    
    # 5. Save to database
    logger.info("Registering model version in database...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Mark all other active CICIDS models as inactive
        db.query(ModelVersion).filter(
            ModelVersion.dataset_source == 'cicids2017'
        ).update({ModelVersion.is_active: False})
        
        # Insert new version
        new_version = ModelVersion(
            dataset_source='cicids2017',
            version_label=f"v1.0-trained-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
            artifact_path=os.path.relpath(model_output_path, os.getcwd()),
            metrics_json=metrics,
            is_active=True
        )
        db.add(new_version)
        db.commit()
        logger.info("Successfully registered model in DB.")
    except Exception as db_err:
        db.rollback()
        logger.error(f"Failed to save model to database: {db_err}")
    finally:
        db.close()

if __name__ == "__main__":
    train_cicids_model()
