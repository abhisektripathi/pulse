"""
Prediction data models for the Predictive System Health Platform.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PredictionType(str, Enum):
    """Types of predictions that can be made."""
    FAILURE = "failure"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ANOMALY = "anomaly"
    CASCADE = "cascade"


class RiskLevel(str, Enum):
    """Risk levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionRequest(BaseModel):
    """Request for generating predictions."""
    entity_id: str = Field(..., description="Entity to predict for")
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    time_horizon_hours: int = Field(..., ge=1, le=168, description="Prediction horizon in hours")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")
    include_factors: bool = Field(default=True, description="Include contributing factors")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class PredictionResponse(BaseModel):
    """Response containing prediction results."""
    entity_id: str = Field(..., description="Entity identifier")
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    time_horizon_hours: int = Field(..., description="Prediction horizon")
    
    # Prediction results
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    risk_level: RiskLevel = Field(..., description="Risk level")
    
    # Time information
    predicted_time: Optional[datetime] = Field(None, description="Predicted occurrence time")
    time_to_event_hours: Optional[float] = Field(None, description="Hours until predicted event")
    
    # Contributing factors
    contributing_factors: List[Dict[str, Any]] = Field(default_factory=list, description="Contributing factors")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    
    # Model information
    model_version: str = Field(..., description="Model version used")
    model_accuracy: float = Field(..., description="Model accuracy score")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnomalyAlert(BaseModel):
    """Alert for detected anomalies."""
    alert_id: str = Field(..., description="Unique alert identifier")
    entity_id: str = Field(..., description="Affected entity")
    timestamp: datetime = Field(..., description="Alert timestamp")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score")
    severity: str = Field(..., description="Anomaly severity")
    metric_name: str = Field(..., description="Anomalous metric")
    metric_value: float = Field(..., description="Current metric value")
    expected_range: Dict[str, float] = Field(..., description="Expected value range")
    description: str = Field(..., description="Anomaly description")
    
    # Context
    historical_context: Dict[str, Any] = Field(default_factory=dict, description="Historical context")
    business_impact: Optional[str] = Field(None, description="Business impact")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FailurePrediction(BaseModel):
    """Detailed failure prediction."""
    entity_id: str = Field(..., description="Entity identifier")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Failure probability")
    time_to_failure_hours: Optional[float] = Field(None, description="Hours until failure")
    failure_type: str = Field(..., description="Type of failure")
    failure_mode: str = Field(..., description="Failure mode")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    
    # Risk assessment
    risk_level: RiskLevel = Field(..., description="Risk level")
    business_impact: float = Field(..., description="Business impact score")
    customer_impact: int = Field(..., ge=1, le=5, description="Customer impact level")
    
    # Mitigation
    mitigation_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    preventive_measures: List[str] = Field(default_factory=list, description="Preventive measures")
    
    # Model details
    model_used: str = Field(..., description="Model used for prediction")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CascadeRiskAssessment(BaseModel):
    """Assessment of cascade failure risk."""
    root_entity_id: str = Field(..., description="Root entity")
    timestamp: datetime = Field(..., description="Assessment timestamp")
    cascade_probability: float = Field(..., ge=0.0, le=1.0, description="Cascade probability")
    affected_entities: List[str] = Field(..., description="Potentially affected entities")
    impact_chain: List[Dict[str, Any]] = Field(..., description="Impact chain analysis")
    
    # Risk metrics
    total_business_impact: float = Field(..., description="Total business impact")
    affected_customers: int = Field(..., description="Number of affected customers")
    revenue_risk: float = Field(..., description="Revenue risk amount")
    
    # Mitigation
    isolation_points: List[str] = Field(default_factory=list, description="Recommended isolation points")
    recovery_time_hours: float = Field(..., description="Estimated recovery time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PredictionSummary(BaseModel):
    """Summary of predictions across entities."""
    timestamp: datetime = Field(..., description="Summary timestamp")
    total_predictions: int = Field(..., description="Total number of predictions")
    high_risk_predictions: int = Field(..., description="High risk predictions")
    critical_predictions: int = Field(..., description="Critical predictions")
    
    # Aggregated metrics
    average_failure_probability: float = Field(..., description="Average failure probability")
    total_business_risk: float = Field(..., description="Total business risk")
    
    # Top predictions
    top_predictions: List[PredictionResponse] = Field(..., description="Top predictions by risk")
    critical_entities: List[str] = Field(..., description="Entities with critical predictions")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 