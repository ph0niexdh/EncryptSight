import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(50), nullable=False)  # 'unsw_nb15' | 'cicids2017'
    source_file = Column(String(255), nullable=False)
    row_count = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    flows = relationship("Flow", back_populates="dataset", cascade="all, delete-orphan")

class Flow(Base):
    __tablename__ = "flows"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    schema_type = Column(String(20), nullable=False)  # 'unsw' | 'cicids'
    raw_features_json = Column(JSON, nullable=False)
    predicted_label = Column(Integer, nullable=False)  # 0=benign, 1=attack
    predicted_attack_cat = Column(String(50), nullable=True)  # e.g. 'DoS' for UNSW, null for CICIDS
    confidence = Column(Float, nullable=False)
    true_label = Column(Integer, nullable=True)  # ground truth if provided
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="flows")
    shap_explanations = relationship("ShapExplanation", back_populates="flow", cascade="all, delete-orphan")

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_source = Column(String(50), nullable=False)  # 'unsw_nb15' | 'cicids2017'
    version_label = Column(String(100), nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    artifact_path = Column(String(255), nullable=False)
    metrics_json = Column(JSON, nullable=False)  # stores model performance metrics
    is_active = Column(Boolean, default=True)

class ShapExplanation(Base):
    __tablename__ = "shap_explanations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    flow_id = Column(String(36), ForeignKey("flows.id"), nullable=False)
    feature_name = Column(String(100), nullable=False)
    shap_value = Column(Float, nullable=False)
    feature_value = Column(Float, nullable=False)

    # Relationships
    flow = relationship("Flow", back_populates="shap_explanations")
