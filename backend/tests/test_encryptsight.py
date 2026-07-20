import os
import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.ml import schema_detect, unsw_model

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert data["models"]["unsw_nb15_loaded"] is True

def test_schema_detection_unsw():
    # Construct a valid UNSW header schema
    header_cols = schema_detect.UNSW_FEATURES + ["attack_cat", "label"]
    df = pd.DataFrame(columns=header_cols)
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    
    schema = schema_detect.detect_schema(csv_buf)
    assert schema == "unsw"
    
    csv_buf.seek(0)
    is_valid, missing = schema_detect.validate_schema(csv_buf, "unsw")
    assert is_valid
    assert len(missing) == 0

def test_schema_detection_cicids():
    # Construct a valid CICIDS header schema (stripping leading spaces from required list)
    header_cols = schema_detect.CICIDS_FEATURES_REQUIRED + ["Label"]
    df = pd.DataFrame(columns=header_cols)
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    
    schema = schema_detect.detect_schema(csv_buf)
    assert schema == "cicids"
    
    csv_buf.seek(0)
    is_valid, missing = schema_detect.validate_schema(csv_buf, "cicids")
    assert is_valid
    assert len(missing) == 0

def test_schema_detection_invalid():
    df = pd.DataFrame(columns=["InvalidCol1", "InvalidCol2", "label_not_matching"])
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    
    with pytest.raises(ValueError):
        schema_detect.detect_schema(csv_buf)

def test_unsw_active_model():
    response = client.get("/api/models/unsw_nb15/active")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_source"] == "unsw_nb15"
    assert "metrics" in data
    assert "accuracy" in data["metrics"]
    assert "confusion_matrix" in data["metrics"]
    assert "feature_importance" in data["metrics"]
