#!/usr/bin/env python3
"""
Health Scoring Service for Predictive System Health Platform

Calculates real-time health scores for system entities using:
- Infrastructure metrics (CPU, memory, disk, network)
- Application metrics (response time, throughput, error rates)
- Business metrics (success rates, SLA compliance)
- Predictive risk factors
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel
import redis
from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxTable

from src.models.health import (
    HealthScore, HealthCategory, HealthFactor, HealthTrend, HealthSummary
)
from src.models.metrics import StandardMetric, MetricType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Health Scoring Service", version="1.0.0")


class HealthScoringService:
    """Service for calculating health scores for system entities."""
    
    def __init__(self, kafka_brokers: str, influxdb_url: str, redis_url: str):
        self.kafka_brokers = kafka_brokers
        self.influxdb_url = influxdb_url
        self.redis_url = redis_url
        
        # Initialize clients
        self.kafka_consumer = KafkaConsumer(
            'normalized-metrics',
            bootstrap_servers=kafka_brokers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='health-scoring-group',
            auto_offset_reset='latest'
        )
        
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_brokers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        
        self.influxdb_client = InfluxDBClient(
            url=influxdb_url,
            token="system-health-token",
            org="system-health"
        )
        
        self.redis_client = redis.from_url(redis_url)
        
        # Health scoring configuration
        self.health_weights = {
            'availability': 0.25,
            'performance': 0.20,
            'error_rate': 0.20,
            'resource_utilization': 0.15,
            'predictive_risk': 0.20
        }
        
        # Thresholds for different metrics
        self.thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95},
            'disk_usage': {'warning': 85, 'critical': 95},
            'response_time': {'warning': 500, 'critical': 1000},
            'error_rate': {'warning': 2.0, 'critical': 5.0},
            'success_rate': {'warning': 98.0, 'critical': 95.0}
        }
        
        # Entity health cache
        self.health_cache = {}
        
    def _calculate_availability_score(self, metrics: List[StandardMetric]) -> Tuple[float, List[HealthFactor]]:
        """Calculate availability score based on uptime and service status."""
        factors = []
        
        # Check for service health endpoints
        health_metrics = [m for m in metrics if 'health' in m.tags.get('metric', '')]
        if health_metrics:
            avg_health = np.mean([float(m.value) for m in health_metrics])
            availability_score = min(100, max(0, avg_health))
        else:
            # Estimate availability based on error rates
            error_metrics = [m for m in metrics if m.tags.get('metric') == 'error_rate']
            if error_metrics:
                avg_error_rate = np.mean([float(m.value) for m in error_metrics])
                availability_score = max(0, 100 - avg_error_rate)
            else:
                availability_score = 95.0  # Default assumption
        
        factors.append(HealthFactor(
            name="availability",
            value=availability_score,
            weight=self.health_weights['availability'],
            impact=availability_score * self.health_weights['availability'],
            threshold_warning=95.0,
            threshold_critical=90.0,
            trend="stable"
        ))
        
        return availability_score, factors
    
    def _calculate_performance_score(self, metrics: List[StandardMetric]) -> Tuple[float, List[HealthFactor]]:
        """Calculate performance score based on response time and throughput."""
        factors = []
        
        # Response time score
        response_time_metrics = [m for m in metrics if m.tags.get('metric') == 'response_time']
        if response_time_metrics:
            avg_response_time = np.mean([float(m.value) for m in response_time_metrics])
            # Normalize response time (lower is better)
            response_time_score = max(0, 100 - (avg_response_time / 10))
        else:
            response_time_score = 85.0  # Default assumption
        
        # Throughput score
        throughput_metrics = [m for m in metrics if m.tags.get('metric') == 'throughput']
        if throughput_metrics:
            avg_throughput = np.mean([float(m.value) for m in throughput_metrics])
            # Normalize throughput (higher is better, but with diminishing returns)
            throughput_score = min(100, avg_throughput / 20)
        else:
            throughput_score = 80.0  # Default assumption
        
        # Combined performance score
        performance_score = (response_time_score * 0.6) + (throughput_score * 0.4)
        
        factors.extend([
            HealthFactor(
                name="response_time",
                value=response_time_score,
                weight=0.6,
                impact=response_time_score * 0.6,
                threshold_warning=self.thresholds['response_time']['warning'],
                threshold_critical=self.thresholds['response_time']['critical'],
                trend="stable"
            ),
            HealthFactor(
                name="throughput",
                value=throughput_score,
                weight=0.4,
                impact=throughput_score * 0.4,
                trend="stable"
            )
        ])
        
        return performance_score, factors
    
    def _calculate_error_rate_score(self, metrics: List[StandardMetric]) -> Tuple[float, List[HealthFactor]]:
        """Calculate error rate score (inverse of error rate)."""
        factors = []
        
        # Error rate score
        error_rate_metrics = [m for m in metrics if m.tags.get('metric') == 'error_rate']
        if error_rate_metrics:
            avg_error_rate = np.mean([float(m.value) for m in error_rate_metrics])
            # Inverse error rate (lower error rate = higher score)
            error_rate_score = max(0, 100 - avg_error_rate)
        else:
            error_rate_score = 95.0  # Default assumption
        
        # Success rate score
        success_rate_metrics = [m for m in metrics if m.tags.get('metric') == 'success_rate']
        if success_rate_metrics:
            avg_success_rate = np.mean([float(m.value) for m in success_rate_metrics])
            success_rate_score = avg_success_rate
        else:
            success_rate_score = 95.0  # Default assumption
        
        # Combined error rate score
        error_rate_score = (error_rate_score * 0.5) + (success_rate_score * 0.5)
        
        factors.append(HealthFactor(
            name="error_rate",
            value=error_rate_score,
            weight=self.health_weights['error_rate'],
            impact=error_rate_score * self.health_weights['error_rate'],
            threshold_warning=self.thresholds['error_rate']['warning'],
            threshold_critical=self.thresholds['error_rate']['critical'],
            trend="stable"
        ))
        
        return error_rate_score, factors
    
    def _calculate_resource_utilization_score(self, metrics: List[StandardMetric]) -> Tuple[float, List[HealthFactor]]:
        """Calculate resource utilization score (lower utilization is better)."""
        factors = []
        
        # CPU utilization
        cpu_metrics = [m for m in metrics if m.tags.get('metric') == 'cpu_usage']
        if cpu_metrics:
            avg_cpu = np.mean([float(m.value) for m in cpu_metrics])
            cpu_score = max(0, 100 - avg_cpu)  # Lower CPU = higher score
        else:
            cpu_score = 60.0  # Default assumption
        
        # Memory utilization
        memory_metrics = [m for m in metrics if m.tags.get('metric') == 'memory_usage']
        if memory_metrics:
            avg_memory = np.mean([float(m.value) for m in memory_metrics])
            memory_score = max(0, 100 - avg_memory)  # Lower memory = higher score
        else:
            memory_score = 50.0  # Default assumption
        
        # Disk utilization
        disk_metrics = [m for m in metrics if m.tags.get('metric') == 'disk_usage']
        if disk_metrics:
            avg_disk = np.mean([float(m.value) for m in disk_metrics])
            disk_score = max(0, 100 - avg_disk)  # Lower disk = higher score
        else:
            disk_score = 70.0  # Default assumption
        
        # Combined resource utilization score
        resource_score = (cpu_score * 0.4) + (memory_score * 0.4) + (disk_score * 0.2)
        
        factors.extend([
            HealthFactor(
                name="cpu_utilization",
                value=cpu_score,
                weight=0.4,
                impact=cpu_score * 0.4,
                threshold_warning=self.thresholds['cpu_usage']['warning'],
                threshold_critical=self.thresholds['cpu_usage']['critical'],
                trend="stable"
            ),
            HealthFactor(
                name="memory_utilization",
                value=memory_score,
                weight=0.4,
                impact=memory_score * 0.4,
                threshold_warning=self.thresholds['memory_usage']['warning'],
                threshold_critical=self.thresholds['memory_usage']['critical'],
                trend="stable"
            ),
            HealthFactor(
                name="disk_utilization",
                value=disk_score,
                weight=0.2,
                impact=disk_score * 0.2,
                threshold_warning=self.thresholds['disk_usage']['warning'],
                threshold_critical=self.thresholds['disk_usage']['critical'],
                trend="stable"
            )
        ])
        
        return resource_score, factors
    
    def _calculate_predictive_risk_score(self, entity_id: str, metrics: List[StandardMetric]) -> Tuple[float, List[HealthFactor]]:
        """Calculate predictive risk score based on trends and patterns."""
        factors = []
        
        # Simple trend analysis (in a real system, this would use ML models)
        risk_score = 85.0  # Default low risk
        
        # Check for concerning trends
        for metric in metrics:
            metric_name = metric.tags.get('metric', '')
            
            if metric_name == 'cpu_usage' and float(metric.value) > 80:
                risk_score -= 10
            elif metric_name == 'memory_usage' and float(metric.value) > 85:
                risk_score -= 10
            elif metric_name == 'error_rate' and float(metric.value) > 3:
                risk_score -= 15
            elif metric_name == 'response_time' and float(metric.value) > 800:
                risk_score -= 10
        
        # Ensure risk score is within bounds
        risk_score = max(0, min(100, risk_score))
        
        factors.append(HealthFactor(
            name="predictive_risk",
            value=risk_score,
            weight=self.health_weights['predictive_risk'],
            impact=risk_score * self.health_weights['predictive_risk'],
            trend="stable"
        ))
        
        return risk_score, factors
    
    def _determine_health_category(self, score: float) -> HealthCategory:
        """Determine health category based on score."""
        if score >= 90:
            return HealthCategory.HEALTHY
        elif score >= 70:
            return HealthCategory.DEGRADED
        elif score >= 50:
            return HealthCategory.WARNING
        else:
            return HealthCategory.CRITICAL
    
    def _calculate_health_trend(self, entity_id: str) -> HealthTrend:
        """Calculate health trend based on historical data."""
        # In a real system, this would analyze historical health scores
        # For now, return a stable trend
        return HealthTrend(
            direction="stable",
            slope=0.0,
            confidence=0.8,
            duration_hours=1.0
        )
    
    def calculate_health_score(self, entity_id: str, metrics: List[StandardMetric]) -> HealthScore:
        """Calculate comprehensive health score for an entity."""
        
        # Calculate individual component scores
        availability_score, availability_factors = self._calculate_availability_score(metrics)
        performance_score, performance_factors = self._calculate_performance_score(metrics)
        error_rate_score, error_rate_factors = self._calculate_error_rate_score(metrics)
        resource_score, resource_factors = self._calculate_resource_utilization_score(metrics)
        predictive_risk_score, risk_factors = self._calculate_predictive_risk_score(entity_id, metrics)
        
        # Calculate weighted composite score
        composite_score = (
            availability_score * self.health_weights['availability'] +
            performance_score * self.health_weights['performance'] +
            error_rate_score * self.health_weights['error_rate'] +
            resource_score * self.health_weights['resource_utilization'] +
            predictive_risk_score * self.health_weights['predictive_risk']
        )
        
        # Combine all factors
        all_factors = availability_factors + performance_factors + error_rate_factors + resource_factors + risk_factors
        
        # Calculate trend
        trend = self._calculate_health_trend(entity_id)
        
        # Determine category
        category = self._determine_health_category(composite_score)
        
        # Calculate business impact
        business_impact_score = composite_score
        revenue_risk = max(0, 100 - composite_score)
        customer_impact = 5 if composite_score < 50 else 4 if composite_score < 70 else 3 if composite_score < 90 else 1
        
        # Create health score
        health_score = HealthScore(
            entity_id=entity_id,
            timestamp=datetime.utcnow(),
            score=composite_score,
            category=category,
            factors=all_factors,
            trend=trend,
            confidence=0.85,
            last_updated=datetime.utcnow(),
            data_freshness_minutes=5,
            business_impact_score=business_impact_score,
            revenue_risk=revenue_risk,
            customer_impact=customer_impact
        )
        
        return health_score
    
    def _get_metrics_from_influxdb(self, entity_id: str, time_range_minutes: int = 15) -> List[StandardMetric]:
        """Retrieve metrics from InfluxDB for health calculation."""
        try:
            query_api = self.influxdb_client.query_api()
            
            # Query for recent metrics
            query = f'''
            from(bucket: "metrics")
                |> range(start: -{time_range_minutes}m)
                |> filter(fn: (r) => r["entity_id"] == "{entity_id}")
                |> filter(fn: (r) => r["_measurement"] == "system_metrics")
            '''
            
            result = query_api.query(query)
            
            metrics = []
            for table in result:
                for record in table.records:
                    # Convert InfluxDB record to StandardMetric
                    metric = StandardMetric(
                        timestamp=record.get_time(),
                        source=record.values.get("source", "unknown"),
                        metric_type=MetricType(record.values.get("metric_type", "infrastructure")),
                        entity=record.values.get("entity_id", entity_id),  # Simplified
                        value=record.get_value(),
                        unit=record.values.get("unit", ""),
                        tags={"metric": record.values.get("metric", "")},
                        business_info=record.values.get("business_info", {})  # Simplified
                    )
                    metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to retrieve metrics from InfluxDB: {e}")
            return []
    
    async def process_metrics(self):
        """Process incoming metrics and calculate health scores."""
        try:
            for message in self.kafka_consumer:
                try:
                    # Parse metrics from message
                    metrics_data = message.value
                    if isinstance(metrics_data, list):
                        metrics = [StandardMetric(**m) for m in metrics_data]
                    else:
                        metrics = [StandardMetric(**metrics_data)]
                    
                    # Group metrics by entity
                    entity_metrics = {}
                    for metric in metrics:
                        entity_id = metric.entity.id if hasattr(metric.entity, 'id') else metric.entity
                        if entity_id not in entity_metrics:
                            entity_metrics[entity_id] = []
                        entity_metrics[entity_id].append(metric)
                    
                    # Calculate health scores for each entity
                    for entity_id, entity_metrics_list in entity_metrics.items():
                        health_score = self.calculate_health_score(entity_id, entity_metrics_list)
                        
                        # Cache health score
                        self.health_cache[entity_id] = health_score
                        
                        # Send to Kafka
                        self.kafka_producer.send('health-scores', health_score.dict())
                        
                        # Cache in Redis
                        self.redis_client.setex(
                            f"health:{entity_id}",
                            300,  # 5 minutes TTL
                            json.dumps(health_score.dict(), default=str)
                        )
                        
                        logger.info(f"Calculated health score for {entity_id}: {health_score.score:.1f}")
                
                except Exception as e:
                    logger.error(f"Error processing metrics: {e}")
        
        except Exception as e:
            logger.error(f"Error in metrics processing loop: {e}")
    
    async def get_health_score(self, entity_id: str) -> Optional[HealthScore]:
        """Get health score for an entity."""
        # Try cache first
        cached_data = self.redis_client.get(f"health:{entity_id}")
        if cached_data:
            return HealthScore(**json.loads(cached_data))
        
        # Get from InfluxDB and calculate
        metrics = self._get_metrics_from_influxdb(entity_id)
        if metrics:
            health_score = self.calculate_health_score(entity_id, metrics)
            return health_score
        
        return None
    
    async def get_health_summary(self) -> HealthSummary:
        """Get health summary across all entities."""
        # Get all cached health scores
        all_entities = []
        for key in self.redis_client.scan_iter("health:*"):
            entity_id = key.decode('utf-8').replace("health:", "")
            cached_data = self.redis_client.get(key)
            if cached_data:
                health_score = HealthScore(**json.loads(cached_data))
                all_entities.append(health_score)
        
        if not all_entities:
            return HealthSummary(
                timestamp=datetime.utcnow(),
                total_entities=0,
                healthy_count=0,
                degraded_count=0,
                warning_count=0,
                critical_count=0,
                average_health_score=0.0,
                weighted_health_score=0.0,
                total_revenue_risk=0.0,
                affected_customers=0,
                top_issues=[],
                critical_entities=[]
            )
        
        # Calculate summary statistics
        total_entities = len(all_entities)
        healthy_count = len([e for e in all_entities if e.category == HealthCategory.HEALTHY])
        degraded_count = len([e for e in all_entities if e.category == HealthCategory.DEGRADED])
        warning_count = len([e for e in all_entities if e.category == HealthCategory.WARNING])
        critical_count = len([e for e in all_entities if e.category == HealthCategory.CRITICAL])
        
        average_health_score = np.mean([e.score for e in all_entities])
        weighted_health_score = np.mean([e.score * e.entity.criticality for e in all_entities])
        
        total_revenue_risk = sum([e.revenue_risk for e in all_entities])
        affected_customers = sum([e.customer_impact for e in all_entities if e.customer_impact > 1])
        
        # Top issues
        top_issues = []
        for entity in sorted(all_entities, key=lambda x: x.score)[:5]:
            if entity.score < 90:
                top_issues.append(f"{entity.entity_id}: {entity.score:.1f}%")
        
        critical_entities = [e.entity_id for e in all_entities if e.category == HealthCategory.CRITICAL]
        
        return HealthSummary(
            timestamp=datetime.utcnow(),
            total_entities=total_entities,
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            warning_count=warning_count,
            critical_count=critical_count,
            average_health_score=average_health_score,
            weighted_health_score=weighted_health_score,
            total_revenue_risk=total_revenue_risk,
            affected_customers=affected_customers,
            top_issues=top_issues,
            critical_entities=critical_entities
        )


# Initialize service
import os

kafka_brokers = os.getenv("KAFKA_BROKERS", "localhost:9092")
influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

health_service = HealthScoringService(kafka_brokers, influxdb_url, redis_url)


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("🚀 Starting Health Scoring Service")
    asyncio.create_task(health_service.process_metrics())


@app.get("/health/{entity_id}")
async def get_entity_health(entity_id: str):
    """Get health score for a specific entity."""
    health_score = await health_service.get_health_score(entity_id)
    if health_score:
        return health_score
    else:
        raise HTTPException(status_code=404, detail="Entity not found")


@app.get("/health/summary")
async def get_health_summary():
    """Get health summary across all entities."""
    return await health_service.get_health_summary()


@app.get("/health")
async def service_health():
    """Service health check."""
    return {"status": "healthy", "service": "health-scoring"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 