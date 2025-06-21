"""
Health scoring data models for the Predictive System Health Platform.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthCategory(str, Enum):
    """Health categories based on score ranges."""
    CRITICAL = "critical"      # 0-49%
    WARNING = "warning"        # 50-69%
    DEGRADED = "degraded"      # 70-89%
    HEALTHY = "healthy"        # 90-100%


class HealthFactor(BaseModel):
    """Individual factor contributing to health score."""
    name: str = Field(..., description="Factor name")
    value: float = Field(..., description="Factor value")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in health calculation")
    impact: float = Field(..., description="Impact on overall health score")
    threshold_warning: Optional[float] = Field(None, description="Warning threshold")
    threshold_critical: Optional[float] = Field(None, description="Critical threshold")
    trend: str = Field(default="stable", description="Trend direction (improving, degrading, stable)")
    recommendation: Optional[str] = Field(None, description="Recommendation for improvement")


class HealthTrend(BaseModel):
    """Health trend information."""
    direction: str = Field(..., description="Trend direction (up, down, stable)")
    slope: float = Field(..., description="Rate of change")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in trend")
    duration_hours: float = Field(..., description="Duration of trend in hours")
    prediction_hours: Optional[float] = Field(None, description="Predicted hours until threshold")


class HealthScore(BaseModel):
    """Comprehensive health score for a system entity."""
    entity_id: str = Field(..., description="Entity identifier")
    timestamp: datetime = Field(..., description="Timestamp of health assessment")
    score: float = Field(..., ge=0.0, le=100.0, description="Overall health score (0-100)")
    category: HealthCategory = Field(..., description="Health category")
    factors: List[HealthFactor] = Field(..., description="Contributing factors")
    trend: HealthTrend = Field(..., description="Health trend information")
    
    # Additional context
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in health assessment")
    last_updated: datetime = Field(..., description="Last update timestamp")
    data_freshness_minutes: int = Field(..., description="Age of data in minutes")
    
    # Business impact
    business_impact_score: float = Field(..., ge=0.0, le=100.0, description="Business impact score")
    revenue_risk: float = Field(..., ge=0.0, le=100.0, description="Revenue risk percentage")
    customer_impact: int = Field(..., ge=1, le=5, description="Customer impact level")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthSummary(BaseModel):
    """Summary of health across multiple entities."""
    timestamp: datetime = Field(..., description="Summary timestamp")
    total_entities: int = Field(..., description="Total number of entities")
    healthy_count: int = Field(..., description="Number of healthy entities")
    degraded_count: int = Field(..., description="Number of degraded entities")
    warning_count: int = Field(..., description="Number of warning entities")
    critical_count: int = Field(..., description="Number of critical entities")
    
    # Aggregated scores
    average_health_score: float = Field(..., description="Average health score")
    weighted_health_score: float = Field(..., description="Weighted by criticality")
    
    # Business impact
    total_revenue_risk: float = Field(..., description="Total revenue risk")
    affected_customers: int = Field(..., description="Number of affected customers")
    
    # Top issues
    top_issues: List[str] = Field(..., description="Top health issues")
    critical_entities: List[str] = Field(..., description="Entities in critical state")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthAlert(BaseModel):
    """Health alert for significant changes."""
    alert_id: str = Field(..., description="Unique alert identifier")
    entity_id: str = Field(..., description="Affected entity")
    timestamp: datetime = Field(..., description="Alert timestamp")
    severity: str = Field(..., description="Alert severity (info, warning, critical)")
    category: str = Field(..., description="Alert category")
    message: str = Field(..., description="Alert message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    
    # Health context
    previous_score: Optional[float] = Field(None, description="Previous health score")
    current_score: float = Field(..., description="Current health score")
    score_change: float = Field(..., description="Change in health score")
    
    # Business context
    business_impact: str = Field(..., description="Business impact description")
    revenue_impact: float = Field(..., description="Revenue impact amount")
    
    # Resolution
    acknowledged: bool = Field(default=False, description="Alert acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    resolved: bool = Field(default=False, description="Alert resolved")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthQuery(BaseModel):
    """Query parameters for health data."""
    entity_ids: Optional[List[str]] = Field(None, description="Filter by entity IDs")
    categories: Optional[List[HealthCategory]] = Field(None, description="Filter by health categories")
    start_time: datetime = Field(..., description="Start time for query")
    end_time: datetime = Field(..., description="End time for query")
    include_factors: bool = Field(default=True, description="Include health factors")
    include_trends: bool = Field(default=True, description="Include trend information")
    limit: Optional[int] = Field(1000, description="Maximum number of results")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 