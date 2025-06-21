"""
Metrics data models for the Predictive System Health Platform.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    BUSINESS = "business"
    CUSTOM = "custom"


class EntityInfo(BaseModel):
    """Information about a system entity."""
    id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Human-readable name")
    type: str = Field(..., description="Type of entity (service, host, database, etc.)")
    environment: str = Field(..., description="Environment (prod, staging, dev)")
    criticality: int = Field(..., ge=1, le=5, description="Criticality level 1-5")
    dependencies: List[str] = Field(default_factory=list, description="List of dependent entity IDs")
    team: Optional[str] = Field(None, description="Team responsible for this entity")
    tags: Dict[str, str] = Field(default_factory=dict, description="Additional tags")


class BusinessContext(BaseModel):
    """Business context information for metrics."""
    service_line: str = Field(..., description="Business service line")
    revenue_impact: float = Field(..., description="Revenue impact score (0-100)")
    customer_impact: int = Field(..., ge=1, le=5, description="Customer impact level 1-5")
    compliance_tag: Optional[str] = Field(None, description="Compliance requirement tag")
    sla_target: Optional[float] = Field(None, description="SLA target in milliseconds")
    business_priority: str = Field(default="medium", description="Business priority level")


class StandardMetric(BaseModel):
    """Standardized metric format for all data sources."""
    timestamp: datetime = Field(..., description="Timestamp of the metric")
    source: str = Field(..., description="Data source (appdynamics, prometheus, etc.)")
    metric_type: MetricType = Field(..., description="Type of metric")
    entity: EntityInfo = Field(..., description="Entity information")
    value: Union[float, int, str, bool] = Field(..., description="Metric value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    tags: Dict[str, str] = Field(default_factory=dict, description="Additional tags")
    business_info: BusinessContext = Field(..., description="Business context")
    
    # Quality and validation fields
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality confidence")
    is_anomalous: Optional[bool] = Field(None, description="Anomaly detection result")
    threshold_warning: Optional[float] = Field(None, description="Warning threshold")
    threshold_critical: Optional[float] = Field(None, description="Critical threshold")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricBatch(BaseModel):
    """Batch of metrics for bulk processing."""
    metrics: List[StandardMetric] = Field(..., description="List of metrics")
    batch_id: str = Field(..., description="Unique batch identifier")
    source: str = Field(..., description="Source system")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricQuery(BaseModel):
    """Query parameters for retrieving metrics."""
    entity_ids: Optional[List[str]] = Field(None, description="Filter by entity IDs")
    metric_types: Optional[List[MetricType]] = Field(None, description="Filter by metric types")
    sources: Optional[List[str]] = Field(None, description="Filter by data sources")
    start_time: datetime = Field(..., description="Start time for query")
    end_time: datetime = Field(..., description="End time for query")
    aggregation: Optional[str] = Field(None, description="Aggregation function (avg, sum, max, min)")
    interval: Optional[str] = Field(None, description="Time interval for aggregation")
    limit: Optional[int] = Field(1000, description="Maximum number of results")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricResponse(BaseModel):
    """Response containing metrics data."""
    metrics: List[StandardMetric] = Field(..., description="List of metrics")
    total_count: int = Field(..., description="Total number of metrics")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    next_cursor: Optional[str] = Field(None, description="Cursor for pagination")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 