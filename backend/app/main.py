import os
import io
import asyncio
import datetime
import logging
import uuid
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.database import Base, SessionLocal, engine, get_db
from app.db.models import Dataset, Flow, ModelVersion, ShapExplanation
from app.ml import schema_detect, unsw_model, cicids_model, shap_explain
from scripts.train_cicids import train_cicids_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("encryptsight.api")

app = FastAPI(title="EncryptSight API", version="1.0.0")

# Enable CORS for React frontend (Vite defaults to port 5173, Tauri to tauri://localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve packaged assets relative to this module rather than the process working
# directory.  The latter differs between pytest, Docker, and the Tauri sidecar.
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MAX_ANALYSIS_ROWS = int(os.getenv("MAX_ANALYSIS_ROWS", "2000"))
analysis_jobs = {}

def initialize_application():
    """Initialize storage and model lanes for startup and lazy ASGI clients."""
    logger.info("Starting EncryptSight backend application...")
    # Create DB tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Initialize UNSW model and metrics
    logger.info("Initializing UNSW-NB15 model...")
    try:
        if unsw_model.booster is None or not unsw_model.cached_metrics:
            unsw_model.init_unsw_model(DATA_DIR)
            logger.info("UNSW-NB15 model initialized successfully.")
        
        # Register the pretrained UNSW model in database if not already present
        db = SessionLocal()
        try:
            exists = db.query(ModelVersion).filter(
                ModelVersion.dataset_source == 'unsw_nb15',
                ModelVersion.version_label == 'v1.0-pretrained'
            ).first()
            if not exists:
                new_version = ModelVersion(
                    dataset_source='unsw_nb15',
                    version_label='v1.0-pretrained',
                    artifact_path='backend/data/models/etf_lgbm_model.txt',
                    metrics_json=unsw_model.cached_metrics,
                    is_active=True
                )
                db.add(new_version)
                db.commit()
                logger.info("Registered pretrained UNSW-NB15 model in the database.")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error initializing UNSW model: {e}")
        
    # Initialize CICIDS model
    logger.info("Initializing CICIDS2017 model...")
    loaded = cicids_model.booster is not None or cicids_model.init_cicids_model(DATA_DIR)
    if loaded:
        logger.info("CICIDS2017 model loaded successfully.")
    else:
        logger.warning("CICIDS2017 model not found. Needs to be trained first.")

@app.on_event("startup")
def startup_event():
    initialize_application()

@app.get("/health")
def health_check():
    # TestClient instances not used as context managers do not run ASGI lifespan
    # events. Lazy initialization keeps those clients and embedded deployments
    # consistent with a normal Uvicorn/Tauri startup.
    initialize_application()
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "models": {
            "unsw_nb15_loaded": unsw_model.booster is not None,
            "cicids2017_loaded": cicids_model.booster is not None
        }
    }

async def process_upload(file: UploadFile, db: Session):
    """
    Uploads a dataset CSV file, detects the schema, validates columns,
    scores all flows using the appropriate booster model, and stores predictions.
    """
    logger.info(f"Received file upload: {file.filename}")
    
    # Read entire file into memory safely (CSV parsing)
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")
        
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    source_row_count = len(df)
    # Full CICIDS/UNSW exports can contain hundreds of thousands of rows.  The
    # interactive analyst path deliberately samples them so a single upload
    # cannot monopolize the API and browser for minutes.
    truncated = source_row_count > MAX_ANALYSIS_ROWS
    if truncated:
        df = df.head(MAX_ANALYSIS_ROWS).copy()
        
    # Auto-detect schema
    try:
        buffer = io.BytesIO(contents)
        schema_type = schema_detect.detect_schema(buffer)
        logger.info(f"Auto-detected schema: {schema_type}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Validate columns
    buffer.seek(0)
    is_valid, missing_cols = schema_detect.validate_schema(buffer, schema_type)
    if not is_valid:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required columns for schema '{schema_type}': {', '.join(missing_cols)}"
        )
        
    # Standardize column headers by stripping whitespace
    df.columns = [col.strip() for col in df.columns]
    
    # Handle ground truth if available in the CSV
    true_labels = [None] * len(df)
    
    if schema_type == "unsw":
        # Look for 'label' or 'label' equivalent column
        if "label" in df.columns:
            true_labels = pd.to_numeric(df["label"], errors='coerce').fillna(-1).astype(int).tolist()
            # Map -1 to None
            true_labels = [tl if tl != -1 else None for tl in true_labels]
            
        # Perform UNSW inference
        try:
            pred_labels, pred_cats, confidences, _ = unsw_model.predict_unsw_batch(df)
        except Exception as e:
            logger.error(f"UNSW inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"UNSW model inference failed: {e}")
            
    elif schema_type == "cicids":
        # Check if CICIDS booster is trained and loaded
        if not cicids_model.is_cicids_model_loaded():
            # Try to load it one more time just in case it was trained recently
            loaded = cicids_model.init_cicids_model(DATA_DIR)
            if not loaded:
                raise HTTPException(
                    status_code=400, 
                    detail="The CICIDS2017 model is not trained yet. Please run training first."
                )
                
        # Look for 'Label' column
        if "Label" in df.columns:
            # Map benign to 0, attacks to 1
            true_labels = df["Label"].astype(str).str.strip().apply(
                lambda x: 0 if x.upper() == "BENIGN" else 1
            ).tolist()
            
        # Perform CICIDS inference
        try:
            pred_labels, pred_cats, confidences, _ = cicids_model.predict_cicids_batch(df)
        except Exception as e:
            logger.error(f"CICIDS inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"CICIDS model inference failed: {e}")
            
    # Create Dataset record
    dataset_name = "unsw_nb15" if schema_type == "unsw" else "cicids2017"
    db_dataset = Dataset(
        name=dataset_name,
        source_file=file.filename,
        row_count=len(df)
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    # Save Flows in chunked bulk mappings for extreme performance
    logger.info(f"Storing {len(df)} flows into database for dataset {db_dataset.id}...")
    flow_mappings = []
    
    for i in range(len(df)):
        # Convert raw row features to dict
        raw_row = df.iloc[i].to_dict()
        # Ensure we convert numpy types to standard python types for JSON serialization
        raw_row_serializable = {
            k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
                float(v) if isinstance(v, (np.floating, np.float64)) else 
                v) 
            for k, v in raw_row.items()
        }
        
        flow_mappings.append({
            "dataset_id": db_dataset.id,
            "schema_type": schema_type,
            "raw_features_json": raw_row_serializable,
            "predicted_label": pred_labels[i],
            "predicted_attack_cat": pred_cats[i] if schema_type == "unsw" else None,
            "confidence": confidences[i],
            "true_label": true_labels[i]
        })
        
    # Bulk insert in chunks of 1000 rows
    chunk_size = 1000
    for chunk_start in range(0, len(flow_mappings), chunk_size):
        chunk = flow_mappings[chunk_start:chunk_start + chunk_size]
        db.bulk_insert_mappings(Flow, chunk)
        
    db.commit()
    logger.info("Dataset and flows successfully saved.")
    
    return {
        "dataset_id": db_dataset.id,
        "name": db_dataset.name,
        "source_file": db_dataset.source_file,
        "row_count": db_dataset.row_count,
        "schema_type": schema_type,
        "source_row_count": source_row_count,
        "truncated": truncated
    }

def run_upload_job(job_id: str, contents: bytes, filename: str):
    """Run CPU/database-heavy scoring outside the request lifecycle."""
    db = SessionLocal()
    try:
        upload = UploadFile(filename=filename, file=io.BytesIO(contents))
        analysis_jobs[job_id] = {"status": "processing"}
        result = asyncio.run(process_upload(upload, db))
        analysis_jobs[job_id] = {"status": "complete", "result": result}
    except HTTPException as exc:
        analysis_jobs[job_id] = {"status": "failed", "detail": exc.detail}
    except Exception as exc:
        logger.exception("Dataset analysis job failed")
        analysis_jobs[job_id] = {"status": "failed", "detail": str(exc)}
    finally:
        db.close()

@app.post("/api/datasets/upload", status_code=202)
async def upload_dataset(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Queue an upload so the API remains responsive during flow scoring."""
    initialize_application()
    contents = await file.read()
    job_id = str(uuid.uuid4())
    analysis_jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(run_upload_job, job_id, contents, file.filename or "upload.csv")
    return {"job_id": job_id, "status": "queued", "max_analysis_rows": MAX_ANALYSIS_ROWS}

@app.get("/api/jobs/{job_id}")
def get_analysis_job(job_id: str):
    job = analysis_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return job

@app.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    return datasets

@app.get("/api/datasets/{id}")
def get_dataset(id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@app.get("/api/datasets/{id}/flows")
def get_dataset_flows(
    id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    predicted_label: Optional[int] = None,
    predicted_attack_cat: Optional[str] = None,
    true_label: Optional[int] = None,
    min_confidence: Optional[float] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Flow).filter(Flow.dataset_id == id)
    
    # Apply filters
    if predicted_label is not None:
        query = query.filter(Flow.predicted_label == predicted_label)
    if predicted_attack_cat is not None:
        query = query.filter(Flow.predicted_attack_cat == predicted_attack_cat)
    if true_label is not None:
        query = query.filter(Flow.true_label == true_label)
    if min_confidence is not None:
        query = query.filter(Flow.confidence >= min_confidence)
        
    # Get total count before pagination
    total_count = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    flows = query.order_by(Flow.confidence.desc()).offset(offset).limit(limit).all()
    
    return {
        "flows": flows,
        "total_count": total_count,
        "page": page,
        "limit": limit
    }

@app.get("/api/flows/{id}")
def get_flow(id: str, db: Session = Depends(get_db)):
    """Return one stored flow for a direct, reload-safe detail view."""
    flow = db.query(Flow).filter(Flow.id == id).first()
    if not flow:
        raise HTTPException(status_code=404, detail="Flow record not found")
    return flow

@app.get("/api/flows/{id}/explanation")
def get_flow_explanation(id: str, db: Session = Depends(get_db)):
    """
    Computes SHAP explanations for one flow on-demand and caches results in database.
    """
    flow = db.query(Flow).filter(Flow.id == id).first()
    if not flow:
        raise HTTPException(status_code=404, detail="Flow record not found")
        
    # Check cache first
    cached = db.query(ShapExplanation).filter(ShapExplanation.flow_id == id).all()
    if cached:
        logger.info(f"Returning cached SHAP explanation for flow {id}")
        return [
            {
                "feature_name": exp.feature_name,
                "shap_value": exp.shap_value,
                "feature_value": exp.feature_value
            }
            for exp in cached
        ]
        
    # Compute on-demand
    logger.info(f"Computing on-demand SHAP explanation for flow {id}...")
    try:
        raw_features = flow.raw_features_json
        if flow.schema_type == "unsw":
            explanation = shap_explain.explain_unsw_flow(raw_features, flow.predicted_attack_cat)
        elif flow.schema_type == "cicids":
            explanation = shap_explain.explain_cicids_flow(raw_features)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid flow schema type: {flow.schema_type}")
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"SHAP explanation calculation failed: {e}")
        
    # Save to cache DB
    cache_objects = [
        ShapExplanation(
            flow_id=id,
            feature_name=item["feature_name"],
            shap_value=item["shap_value"],
            feature_value=item["feature_value"]
        )
        for item in explanation
    ]
    db.bulk_save_objects(cache_objects)
    db.commit()
    
    return explanation

@app.get("/api/datasets/{id}/summary")
def get_dataset_summary(id: str, db: Session = Depends(get_db)):
    """Returns dataset class distribution and accuracy metrics if ground truth was present."""
    dataset = db.query(Dataset).filter(Dataset.id == id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    # Query flow counts
    flows = db.query(Flow).filter(Flow.dataset_id == id).all()
    if not flows:
        return {
            "row_count": 0,
            "predictions_count": {},
            "accuracy": None
        }
        
    # Predictions distribution
    predictions_count = {}
    correct = 0
    total_with_truth = 0
    
    for f in flows:
        if f.schema_type == "unsw":
            cat = f.predicted_attack_cat or "Unknown"
            predictions_count[cat] = predictions_count.get(cat, 0) + 1
        else:
            cat = "Attack" if f.predicted_label == 1 else "Benign"
            predictions_count[cat] = predictions_count.get(cat, 0) + 1
            
        if f.true_label is not None:
            total_with_truth += 1
            if f.predicted_label == f.true_label:
                correct += 1
                
    accuracy = (correct / total_with_truth) if total_with_truth > 0 else None
    
    return {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "source_file": dataset.source_file,
        "row_count": dataset.row_count,
        "uploaded_at": dataset.uploaded_at,
        "predictions_count": predictions_count,
        "accuracy": accuracy,
        "labeled_count": total_with_truth
    }

@app.get("/api/models/{dataset_source}/active")
def get_active_model(dataset_source: str, db: Session = Depends(get_db)):
    """Returns active model details and performance metrics."""
    initialize_application()
    if dataset_source == "unsw_nb15":
        # UNSW metrics are pre-computed live on startup
        if not unsw_model.cached_metrics:
            raise HTTPException(status_code=500, detail="UNSW model was not initialized successfully.")
            
        active_model = db.query(ModelVersion).filter(
            ModelVersion.dataset_source == 'unsw_nb15',
            ModelVersion.is_active == True
        ).first()
        
        return {
            "version_label": active_model.version_label if active_model else "v1.0-pretrained",
            "dataset_source": "unsw_nb15",
            "metrics": unsw_model.cached_metrics
        }
        
    elif dataset_source == "cicids2017":
        active_model = db.query(ModelVersion).filter(
            ModelVersion.dataset_source == 'cicids2017',
            ModelVersion.is_active == True
        ).first()
        
        if not active_model:
            raise HTTPException(
                status_code=404, 
                detail="No active model found for CICIDS2017. Please run training first."
            )
            
        return {
            "version_label": active_model.version_label,
            "dataset_source": "cicids2017",
            "trained_at": active_model.trained_at,
            "metrics": active_model.metrics_json
        }
        
    else:
        raise HTTPException(status_code=400, detail=f"Invalid dataset source: {dataset_source}")

@app.get("/api/models/{dataset_source}/confusion-matrix")
def get_model_confusion_matrix(dataset_source: str, db: Session = Depends(get_db)):
    """Returns the confusion matrix for the active model."""
    initialize_application()
    if dataset_source == "unsw_nb15":
        if not unsw_model.cached_metrics:
            raise HTTPException(status_code=500, detail="UNSW model was not initialized successfully.")
        return {
            "classes": unsw_model.cached_metrics["classes"],
            "matrix": unsw_model.cached_metrics["confusion_matrix"]
        }
    elif dataset_source == "cicids2017":
        active_model = db.query(ModelVersion).filter(
            ModelVersion.dataset_source == 'cicids2017',
            ModelVersion.is_active == True
        ).first()
        if not active_model:
            raise HTTPException(
                status_code=404, 
                detail="No active model found for CICIDS2017. Please run training first."
            )
        return {
            "classes": active_model.metrics_json["classes"],
            "matrix": active_model.metrics_json["confusion_matrix"]
        }
    else:
        raise HTTPException(status_code=400, detail=f"Invalid dataset source: {dataset_source}")

@app.post("/api/models/cicids2017/train")
def train_cicids_endpoint(background_tasks: BackgroundTasks):
    """Launches the CICIDS2017 training pipeline in the background."""
    background_tasks.add_task(train_cicids_model)
    return {"status": "training_started", "message": "CICIDS2017 training has been started in the background."}
