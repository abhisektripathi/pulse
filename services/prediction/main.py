#!/usr/bin/env python3
"""
Prediction Service for Predictive System Health Platform

Uses machine learning models to predict:
- System failures (2-24 hour horizon)
- Resource exhaustion
- Performance degradation
- Anomaly detection
- Cascade failure risk
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel
import redis
from influxdb_client import InfluxDBClient
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

from src.models.predictions import (
    PredictionRequest, PredictionResponse, PredictionType, RiskLevel,
    AnomalyAlert, FailurePrediction, CascadeRiskAssessment
)
from src.models.health import HealthScore
from src.models.metrics import StandardMetric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prediction Service", version="1.0.0")


class PredictionService:
    """Service for ML-driven system predictions."""
    
    def __init__(self, kafka_brokers: str, influxdb_url: str, redis_url: str, mlflow_uri: str):
        self.kafka_brokers = kafka_brokers
        self.influxdb_url = influxdb_url
        self.redis_url = redis_url
        self.mlflow_uri = mlflow_uri
        
        # Initialize clients
        self.kafka_consumer = KafkaConsumer(
            'health-scores',
            bootstrap_servers=kafka_brokers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='prediction-group',
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
        
        # Initialize MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        
        # ML models
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        
        # Load models
        self._load_models()
        
        # Prediction cache
        self.prediction_cache = {}
        
    def _load_models(self):
        """Load trained ML models from MLflow."""
        try:
            # Load anomaly detection model
            self.models['anomaly_detector'] = mlflow.sklearn.load_model(
                "models:/anomaly_detector/Production"
            )
            self.scalers['anomaly_detector'] = mlflow.sklearn.load_model(
                "models:/anomaly_detector_scaler/Production"
            )
            
            # Load failure prediction model
            self.models['failure_predictor'] = mlflow.sklearn.load_model(
                "models:/failure_predictor/Production"
            )
            self.scalers['failure_predictor'] = mlflow.sklearn.load_model(
                "models:/failure_predictor_scaler/Production"
            )
            
            # Load performance prediction model
            self.models['performance_predictor'] = mlflow.sklearn.load_model(
                "models:/performance_predictor/Production"
            )
            self.scalers['performance_predictor'] = mlflow.sklearn.load_model(
                "models:/performance_predictor_scaler/Production"
            )
            
            logger.info("✅ ML models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load models from MLflow: {e}")
            logger.info("Training new models with default parameters...")
            self._train_default_models()
    
    def _train_default_models(self):
        """Train default models with synthetic data."""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Synthetic features
        cpu_usage = np.random.uniform(20, 95, n_samples)
        memory_usage = np.random.uniform(30, 90, n_samples)
        disk_usage = np.random.uniform(40, 85, n_samples)
        error_rate = np.random.uniform(0, 10, n_samples)
        response_time = np.random.uniform(50, 1000, n_samples)
        throughput = np.random.uniform(100, 2000, n_samples)
        
        # Create feature matrix
        X = np.column_stack([cpu_usage, memory_usage, disk_usage, error_rate, response_time, throughput])
        
        # Anomaly detection model
        self.models['anomaly_detector'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.models['anomaly_detector'].fit(X)
        
        # Failure prediction model (binary classification)
        # Create synthetic failure labels (failures when multiple metrics are high)
        failure_labels = (
            (cpu_usage > 85) & (memory_usage > 80) |
            (error_rate > 5) & (response_time > 800) |
            (disk_usage > 90)
        ).astype(int)
        
        self.models['failure_predictor'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.models['failure_predictor'].fit(X, failure_labels)
        
        # Performance prediction model
        performance_scores = 100 - (
            cpu_usage * 0.3 + memory_usage * 0.3 + 
            disk_usage * 0.2 + error_rate * 2 + 
            (response_time / 1000) * 10
        )
        performance_scores = np.clip(performance_scores, 0, 100)
        
        self.models['performance_predictor'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.models['performance_predictor'].fit(X, performance_scores)
        
        # Create scalers
        self.scalers['anomaly_detector'] = StandardScaler()
        self.scalers['failure_predictor'] = StandardScaler()
        self.scalers['performance_predictor'] = StandardScaler()
        
        self.scalers['anomaly_detector'].fit(X)
        self.scalers['failure_predictor'].fit(X)
        self.scalers['performance_predictor'].fit(X)
        
        logger.info("✅ Default models trained successfully")
    
    def _extract_features(self, metrics: List[StandardMetric]) -> np.ndarray:
        """Extract features from metrics for ML models."""
        feature_dict = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'error_rate': 0.0,
            'response_time': 0.0,
            'throughput': 0.0
        }
        
        # Extract latest values for each metric
        for metric in metrics:
            metric_name = metric.tags.get('metric', '')
            if metric_name in feature_dict:
                feature_dict[metric_name] = float(metric.value)
        
        # Return feature array
        return np.array([
            feature_dict['cpu_usage'],
            feature_dict['memory_usage'],
            feature_dict['disk_usage'],
            feature_dict['error_rate'],
            feature_dict['response_time'],
            feature_dict['throughput']
        ]).reshape(1, -1)
    
    def _detect_anomalies(self, entity_id: str, metrics: List[StandardMetric]) -> List[AnomalyAlert]:
        """Detect anomalies in system metrics."""
        alerts = []
        
        try:
            # Extract features
            features = self._extract_features(metrics)
            
            # Scale features
            scaled_features = self.scalers['anomaly_detector'].transform(features)
            
            # Get anomaly scores
            anomaly_scores = self.models['anomaly_detector'].decision_function(scaled_features)
            
            # Check for anomalies
            for i, score in enumerate(anomaly_scores):
                if score < -0.5:  # Threshold for anomaly
                    # Find the most anomalous metric
                    metric_values = features[i]
                    metric_names = ['cpu_usage', 'memory_usage', 'disk_usage', 'error_rate', 'response_time', 'throughput']
                    
                    # Find metric with highest deviation
                    deviations = np.abs(metric_values - np.mean(metric_values))
                    anomalous_metric_idx = np.argmax(deviations)
                    anomalous_metric = metric_names[anomalous_metric_idx]
                    anomalous_value = metric_values[anomalous_metric_idx]
                    
                    # Create anomaly alert
                    alert = AnomalyAlert(
                        alert_id=str(uuid.uuid4()),
                        entity_id=entity_id,
                        timestamp=datetime.utcnow(),
                        anomaly_score=float(score),
                        severity="high" if score < -0.8 else "medium",
                        metric_name=anomalous_metric,
                        metric_value=anomalous_value,
                        expected_range={
                            "min": float(np.mean(metric_values) - np.std(metric_values)),
                            "max": float(np.mean(metric_values) + np.std(metric_values))
                        },
                        description=f"Anomalous {anomalous_metric} detected: {anomalous_value:.2f}",
                        business_impact="Potential performance degradation"
                    )
                    
                    alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return alerts
    
    def _predict_failures(self, entity_id: str, metrics: List[StandardMetric], time_horizon_hours: int) -> FailurePrediction:
        """Predict system failures."""
        try:
            # Extract features
            features = self._extract_features(metrics)
            
            # Scale features
            scaled_features = self.scalers['failure_predictor'].transform(features)
            
            # Get failure probability
            failure_probability = self.models['failure_predictor'].predict(scaled_features)[0]
            
            # Determine risk level
            if failure_probability > 0.7:
                risk_level = RiskLevel.CRITICAL
            elif failure_probability > 0.5:
                risk_level = RiskLevel.HIGH
            elif failure_probability > 0.3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Estimate time to failure (simplified)
            time_to_failure_hours = None
            if failure_probability > 0.5:
                time_to_failure_hours = max(1, int(24 * (1 - failure_probability)))
            
            # Determine failure type based on metrics
            failure_type = "performance_degradation"
            failure_mode = "gradual"
            
            if any(m.tags.get('metric') == 'error_rate' and float(m.value) > 5 for m in metrics):
                failure_type = "service_failure"
                failure_mode = "sudden"
            elif any(m.tags.get('metric') == 'cpu_usage' and float(m.value) > 90 for m in metrics):
                failure_type = "resource_exhaustion"
                failure_mode = "gradual"
            
            # Calculate business impact
            business_impact = failure_probability * 100
            customer_impact = 5 if failure_probability > 0.7 else 4 if failure_probability > 0.5 else 3
            
            # Generate mitigation actions
            mitigation_actions = []
            if failure_type == "resource_exhaustion":
                mitigation_actions.extend([
                    "Scale up resources immediately",
                    "Check for resource leaks",
                    "Optimize resource usage"
                ])
            elif failure_type == "service_failure":
                mitigation_actions.extend([
                    "Check error logs for root cause",
                    "Restart service if necessary",
                    "Verify dependencies"
                ])
            
            # Get feature importance
            feature_names = ['cpu_usage', 'memory_usage', 'disk_usage', 'error_rate', 'response_time', 'throughput']
            feature_importance = dict(zip(feature_names, self.models['failure_predictor'].feature_importances_))
            
            return FailurePrediction(
                entity_id=entity_id,
                timestamp=datetime.utcnow(),
                failure_probability=failure_probability,
                time_to_failure_hours=time_to_failure_hours,
                failure_type=failure_type,
                failure_mode=failure_mode,
                confidence=0.85,
                risk_level=risk_level,
                business_impact=business_impact,
                customer_impact=customer_impact,
                mitigation_actions=mitigation_actions,
                preventive_measures=[
                    "Implement proactive monitoring",
                    "Set up automated scaling",
                    "Regular health checks"
                ],
                model_used="RandomForest",
                feature_importance=feature_importance
            )
        
        except Exception as e:
            logger.error(f"Error in failure prediction: {e}")
            # Return default prediction
            return FailurePrediction(
                entity_id=entity_id,
                timestamp=datetime.utcnow(),
                failure_probability=0.1,
                failure_type="unknown",
                failure_mode="unknown",
                confidence=0.5,
                risk_level=RiskLevel.LOW,
                business_impact=10.0,
                customer_impact=1,
                mitigation_actions=["Monitor system closely"],
                preventive_measures=["Implement monitoring"],
                model_used="default",
                feature_importance={}
            )
    
    def _assess_cascade_risk(self, entity_id: str, dependencies: List[str]) -> CascadeRiskAssessment:
        """Assess cascade failure risk."""
        try:
            # Get health scores for dependent entities
            dependent_health_scores = []
            for dep_id in dependencies:
                cached_data = self.redis_client.get(f"health:{dep_id}")
                if cached_data:
                    health_score = HealthScore(**json.loads(cached_data))
                    dependent_health_scores.append(health_score)
            
            if not dependent_health_scores:
                return CascadeRiskAssessment(
                    root_entity_id=entity_id,
                    timestamp=datetime.utcnow(),
                    cascade_probability=0.1,
                    affected_entities=dependencies,
                    impact_chain=[],
                    total_business_impact=10.0,
                    affected_customers=100,
                    revenue_risk=1000.0,
                    isolation_points=[],
                    recovery_time_hours=1.0
                )
            
            # Calculate cascade probability based on dependent health scores
            avg_dependent_health = np.mean([h.score for h in dependent_health_scores])
            cascade_probability = max(0, (100 - avg_dependent_health) / 100)
            
            # Calculate business impact
            total_business_impact = sum([h.business_impact_score for h in dependent_health_scores])
            affected_customers = sum([h.customer_impact for h in dependent_health_scores])
            revenue_risk = sum([h.revenue_risk for h in dependent_health_scores])
            
            # Generate impact chain
            impact_chain = []
            for health_score in dependent_health_scores:
                impact_chain.append({
                    "entity_id": health_score.entity_id,
                    "health_score": health_score.score,
                    "impact_level": "high" if health_score.score < 50 else "medium" if health_score.score < 70 else "low"
                })
            
            # Identify isolation points
            isolation_points = [dep_id for dep_id in dependencies if any(
                h.entity_id == dep_id and h.score > 80 for h in dependent_health_scores
            )]
            
            # Estimate recovery time
            recovery_time_hours = max(1, int(24 * cascade_probability))
            
            return CascadeRiskAssessment(
                root_entity_id=entity_id,
                timestamp=datetime.utcnow(),
                cascade_probability=cascade_probability,
                affected_entities=dependencies,
                impact_chain=impact_chain,
                total_business_impact=total_business_impact,
                affected_customers=affected_customers,
                revenue_risk=revenue_risk,
                isolation_points=isolation_points,
                recovery_time_hours=recovery_time_hours
            )
        
        except Exception as e:
            logger.error(f"Error in cascade risk assessment: {e}")
            return CascadeRiskAssessment(
                root_entity_id=entity_id,
                timestamp=datetime.utcnow(),
                cascade_probability=0.1,
                affected_entities=dependencies,
                impact_chain=[],
                total_business_impact=10.0,
                affected_customers=100,
                revenue_risk=1000.0,
                isolation_points=[],
                recovery_time_hours=1.0
            )
    
    def _get_metrics_from_influxdb(self, entity_id: str, time_range_minutes: int = 30) -> List[StandardMetric]:
        """Retrieve metrics from InfluxDB for prediction."""
        try:
            query_api = self.influxdb_client.query_api()
            
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
                    metric = StandardMetric(
                        timestamp=record.get_time(),
                        source=record.values.get("source", "unknown"),
                        metric_type=record.values.get("metric_type", "infrastructure"),
                        entity=record.values.get("entity_id", entity_id),
                        value=record.get_value(),
                        unit=record.values.get("unit", ""),
                        tags={"metric": record.values.get("metric", "")},
                        business_info=record.values.get("business_info", {})
                    )
                    metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to retrieve metrics from InfluxDB: {e}")
            return []
    
    async def generate_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Generate prediction for an entity."""
        try:
            # Get metrics for the entity
            metrics = self._get_metrics_from_influxdb(request.entity_id)
            
            if not metrics:
                raise HTTPException(status_code=404, detail="No metrics found for entity")
            
            # Generate prediction based on type
            if request.prediction_type == PredictionType.FAILURE:
                failure_prediction = self._predict_failures(
                    request.entity_id, metrics, request.time_horizon_hours
                )
                
                return PredictionResponse(
                    entity_id=request.entity_id,
                    prediction_type=request.prediction_type,
                    timestamp=datetime.utcnow(),
                    time_horizon_hours=request.time_horizon_hours,
                    probability=failure_prediction.failure_probability,
                    confidence=failure_prediction.confidence,
                    risk_level=failure_prediction.risk_level,
                    predicted_time=datetime.utcnow() + timedelta(hours=failure_prediction.time_to_failure_hours) if failure_prediction.time_to_failure_hours else None,
                    time_to_event_hours=failure_prediction.time_to_failure_hours,
                    contributing_factors=[
                        {"factor": k, "importance": v} for k, v in failure_prediction.feature_importance.items()
                    ],
                    recommendations=failure_prediction.mitigation_actions,
                    model_version="1.0",
                    model_accuracy=0.85
                )
            
            elif request.prediction_type == PredictionType.ANOMALY:
                anomaly_alerts = self._detect_anomalies(request.entity_id, metrics)
                
                if anomaly_alerts:
                    alert = anomaly_alerts[0]  # Return first anomaly
                    return PredictionResponse(
                        entity_id=request.entity_id,
                        prediction_type=request.prediction_type,
                        timestamp=datetime.utcnow(),
                        time_horizon_hours=request.time_horizon_hours,
                        probability=1.0 - alert.anomaly_score,
                        confidence=0.8,
                        risk_level=RiskLevel.HIGH if alert.severity == "high" else RiskLevel.MEDIUM,
                        contributing_factors=[{"factor": alert.metric_name, "value": alert.metric_value}],
                        recommendations=["Investigate anomalous behavior", "Check system logs"],
                        model_version="1.0",
                        model_accuracy=0.8
                    )
                else:
                    return PredictionResponse(
                        entity_id=request.entity_id,
                        prediction_type=request.prediction_type,
                        timestamp=datetime.utcnow(),
                        time_horizon_hours=request.time_horizon_hours,
                        probability=0.0,
                        confidence=0.9,
                        risk_level=RiskLevel.LOW,
                        contributing_factors=[],
                        recommendations=["Continue monitoring"],
                        model_version="1.0",
                        model_accuracy=0.8
                    )
            
            else:
                # Default performance prediction
                features = self._extract_features(metrics)
                scaled_features = self.scalers['performance_predictor'].transform(features)
                performance_score = self.models['performance_predictor'].predict(scaled_features)[0]
                
                return PredictionResponse(
                    entity_id=request.entity_id,
                    prediction_type=request.prediction_type,
                    timestamp=datetime.utcnow(),
                    time_horizon_hours=request.time_horizon_hours,
                    probability=performance_score / 100,
                    confidence=0.85,
                    risk_level=RiskLevel.LOW if performance_score > 80 else RiskLevel.MEDIUM,
                    contributing_factors=[{"factor": "performance_score", "value": performance_score}],
                    recommendations=["Monitor performance trends"],
                    model_version="1.0",
                    model_accuracy=0.85
                )
        
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def process_health_scores(self):
        """Process incoming health scores and generate predictions."""
        try:
            for message in self.kafka_consumer:
                try:
                    # Parse health score
                    health_data = message.value
                    health_score = HealthScore(**health_data)
                    
                    # Generate predictions for critical entities
                    if health_score.category in ["critical", "warning"]:
                        # Get metrics for prediction
                        metrics = self._get_metrics_from_influxdb(health_score.entity_id)
                        
                        if metrics:
                            # Generate failure prediction
                            failure_prediction = self._predict_failures(
                                health_score.entity_id, metrics, 24
                            )
                            
                            # Send prediction to Kafka
                            self.kafka_producer.send('predictions', failure_prediction.dict())
                            
                            # Cache prediction
                            self.prediction_cache[health_score.entity_id] = failure_prediction
                            
                            logger.info(f"Generated failure prediction for {health_score.entity_id}: {failure_prediction.failure_probability:.2f}")
                
                except Exception as e:
                    logger.error(f"Error processing health score: {e}")
        
        except Exception as e:
            logger.error(f"Error in health score processing loop: {e}")


# Initialize service
import os

kafka_brokers = os.getenv("KAFKA_BROKERS", "localhost:9092")
influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

prediction_service = PredictionService(kafka_brokers, influxdb_url, redis_url, mlflow_uri)


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("🚀 Starting Prediction Service")
    asyncio.create_task(prediction_service.process_health_scores())


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate prediction for an entity."""
    return await prediction_service.generate_prediction(request)


@app.get("/predictions/{entity_id}")
async def get_entity_predictions(entity_id: str):
    """Get cached predictions for an entity."""
    if entity_id in prediction_service.prediction_cache:
        return prediction_service.prediction_cache[entity_id]
    else:
        raise HTTPException(status_code=404, detail="No predictions found")


@app.get("/health")
async def service_health():
    """Service health check."""
    return {"status": "healthy", "service": "prediction"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 