"""
Entity data models for the Predictive System Health Platform.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of system entities."""
    SERVICE = "service"
    DATABASE = "database"
    HOST = "host"
    QUEUE = "queue"
    API = "api"
    CONTAINER = "container"
    POD = "pod"
    NAMESPACE = "namespace"


class Environment(str, Enum):
    """Environment types."""
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TESTING = "testing"


class SystemEntity(BaseModel):
    """Base system entity."""
    id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    environment: Environment = Field(..., description="Environment")
    criticality: int = Field(..., ge=1, le=5, description="Criticality level 1-5")
    team: Optional[str] = Field(None, description="Responsible team")
    description: Optional[str] = Field(None, description="Entity description")
    tags: Dict[str, str] = Field(default_factory=dict, description="Entity tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ServiceEntity(SystemEntity):
    """Service-specific entity."""
    service_type: str = Field(..., description="Type of service")
    version: str = Field(..., description="Service version")
    endpoints: List[str] = Field(default_factory=list, description="Service endpoints")
    health_check_url: Optional[str] = Field(None, description="Health check endpoint")
    sla_target_ms: Optional[int] = Field(None, description="SLA target in milliseconds")
    max_instances: Optional[int] = Field(None, description="Maximum instances")
    min_instances: Optional[int] = Field(None, description="Minimum instances")
    scaling_policy: Optional[str] = Field(None, description="Scaling policy")


class DatabaseEntity(SystemEntity):
    """Database-specific entity."""
    database_type: str = Field(..., description="Database type")
    version: str = Field(..., description="Database version")
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    database_name: str = Field(..., description="Database name")
    connection_pool_size: Optional[int] = Field(None, description="Connection pool size")
    max_connections: Optional[int] = Field(None, description="Maximum connections")
    backup_schedule: Optional[str] = Field(None, description="Backup schedule")
    replication_factor: Optional[int] = Field(None, description="Replication factor")


class HostEntity(SystemEntity):
    """Host-specific entity."""
    hostname: str = Field(..., description="Host hostname")
    ip_address: str = Field(..., description="IP address")
    os_type: str = Field(..., description="Operating system type")
    os_version: str = Field(..., description="Operating system version")
    cpu_cores: int = Field(..., description="Number of CPU cores")
    memory_gb: float = Field(..., description="Memory in GB")
    disk_gb: float = Field(..., description="Disk space in GB")
    location: Optional[str] = Field(None, description="Physical location")
    datacenter: Optional[str] = Field(None, description="Datacenter")


class Dependency(BaseModel):
    """Dependency relationship between entities."""
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    dependency_type: str = Field(..., description="Type of dependency")
    strength: float = Field(..., ge=0.0, le=1.0, description="Dependency strength")
    latency_ms: Optional[float] = Field(None, description="Expected latency")
    timeout_ms: Optional[float] = Field(None, description="Timeout value")
    retry_policy: Optional[str] = Field(None, description="Retry policy")
    circuit_breaker: Optional[bool] = Field(None, description="Circuit breaker enabled")
    created_at: datetime = Field(..., description="Dependency creation time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EntityHealth(BaseModel):
    """Health information for an entity."""
    entity_id: str = Field(..., description="Entity ID")
    timestamp: datetime = Field(..., description="Health timestamp")
    status: str = Field(..., description="Current status")
    health_score: float = Field(..., ge=0.0, le=100.0, description="Health score")
    last_check: datetime = Field(..., description="Last health check")
    uptime_percentage: float = Field(..., ge=0.0, le=100.0, description="Uptime percentage")
    response_time_ms: Optional[float] = Field(None, description="Response time")
    error_rate: Optional[float] = Field(None, description="Error rate")
    throughput: Optional[float] = Field(None, description="Throughput")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EntityMetrics(BaseModel):
    """Metrics for an entity."""
    entity_id: str = Field(..., description="Entity ID")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, description="Memory usage percentage")
    disk_usage: Optional[float] = Field(None, description="Disk usage percentage")
    network_io: Optional[float] = Field(None, description="Network I/O")
    active_connections: Optional[int] = Field(None, description="Active connections")
    queue_depth: Optional[int] = Field(None, description="Queue depth")
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Custom metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EntitySummary(BaseModel):
    """Summary of entity information."""
    total_entities: int = Field(..., description="Total number of entities")
    entities_by_type: Dict[str, int] = Field(..., description="Entities by type")
    entities_by_environment: Dict[str, int] = Field(..., description="Entities by environment")
    entities_by_criticality: Dict[str, int] = Field(..., description="Entities by criticality")
    healthy_entities: int = Field(..., description="Number of healthy entities")
    unhealthy_entities: int = Field(..., description="Number of unhealthy entities")
    critical_entities: List[str] = Field(..., description="Critical entities")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 