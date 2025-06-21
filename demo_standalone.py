#!/usr/bin/env python3
"""
Standalone Demo for Predictive System Health Platform

This script runs a simplified version of the platform that demonstrates:
- Data simulation
- Health scoring
- Prediction generation
- Conversational AI
- API endpoints

All without requiring Docker or external infrastructure.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from src.models.metrics import StandardMetric, EntityInfo, BusinessContext, MetricType
from src.models.health import HealthScore, HealthCategory, HealthFactor, HealthTrend, HealthSummary
from src.models.predictions import PredictionRequest, PredictionResponse, PredictionType, RiskLevel
from src.models.conversation import ConversationRequest, ConversationResponse, Insight, InsightType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive System Health Platform - Demo", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StandaloneDemo:
    """Standalone demo that simulates the complete platform."""
    
    def __init__(self):
        self.entities = {
            "payment-gateway": {
                "id": "payment-gateway",
                "name": "Payment Gateway Service",
                "type": "service",
                "criticality": 5,
                "team": "payments-core"
            },
            "fraud-detection": {
                "id": "fraud-detection", 
                "name": "Fraud Detection Service",
                "type": "service",
                "criticality": 4,
                "team": "risk-management"
            },
            "payment-db": {
                "id": "payment-db",
                "name": "Payment Database",
                "type": "database", 
                "criticality": 5,
                "team": "database-team"
            }
        }
        
        self.health_scores = {}
        self.predictions = {}
        self.metrics_history = {}
        self.anomaly_mode = False
        
        # Start background simulation
        self.running = True
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def _generate_metrics(self, entity_id: str) -> List[StandardMetric]:
        """Generate realistic metrics for an entity."""
        metrics = []
        timestamp = datetime.now(timezone.utc)
        
        # Base values
        base_cpu = 45.0
        base_memory = 65.0
        base_response_time = 120.0
        base_error_rate = 0.5
        
        # Add some variation and anomalies
        if self.anomaly_mode and entity_id == "payment-gateway":
            base_cpu = 85.0
            base_response_time = 800.0
            base_error_rate = 8.0
        
        # Add random variation
        cpu_usage = max(0, min(100, base_cpu + np.random.normal(0, 5)))
        memory_usage = max(0, min(100, base_memory + np.random.normal(0, 3)))
        response_time = max(0, base_response_time + np.random.normal(0, 20))
        error_rate = max(0, min(100, base_error_rate + np.random.normal(0, 0.2)))
        
        # Create metrics
        metrics.extend([
            StandardMetric(
                timestamp=timestamp,
                source="prometheus",
                metric_type=MetricType.INFRASTRUCTURE,
                entity=EntityInfo(
                    id=entity_id,
                    name=self.entities[entity_id]["name"],
                    type=self.entities[entity_id]["type"],
                    environment="production",
                    criticality=self.entities[entity_id]["criticality"],
                    team=self.entities[entity_id]["team"]
                ),
                value=float(cpu_usage),
                unit="percent",
                business_info=BusinessContext(
                    service_line="payments",
                    revenue_impact=95.0,
                    customer_impact=5,
                    compliance_tag="PCI-DSS",
                    sla_target=200.0,
                    business_priority="critical"
                ),
                tags={"metric": "cpu_usage"},
                is_anomalous=False,
                threshold_warning=70.0,
                threshold_critical=90.0
            ),
            StandardMetric(
                timestamp=timestamp,
                source="prometheus", 
                metric_type=MetricType.INFRASTRUCTURE,
                entity=EntityInfo(
                    id=entity_id,
                    name=self.entities[entity_id]["name"],
                    type=self.entities[entity_id]["type"],
                    environment="production",
                    criticality=self.entities[entity_id]["criticality"],
                    team=self.entities[entity_id]["team"]
                ),
                value=float(memory_usage),
                unit="percent",
                business_info=BusinessContext(
                    service_line="payments",
                    revenue_impact=95.0,
                    customer_impact=5,
                    compliance_tag="PCI-DSS",
                    sla_target=200.0,
                    business_priority="critical"
                ),
                tags={"metric": "memory_usage"},
                is_anomalous=False,
                threshold_warning=80.0,
                threshold_critical=95.0
            ),
            StandardMetric(
                timestamp=timestamp,
                source="appdynamics",
                metric_type=MetricType.APPLICATION,
                entity=EntityInfo(
                    id=entity_id,
                    name=self.entities[entity_id]["name"],
                    type=self.entities[entity_id]["type"],
                    environment="production",
                    criticality=self.entities[entity_id]["criticality"],
                    team=self.entities[entity_id]["team"]
                ),
                value=float(response_time),
                unit="milliseconds",
                business_info=BusinessContext(
                    service_line="payments",
                    revenue_impact=95.0,
                    customer_impact=5,
                    compliance_tag="PCI-DSS",
                    sla_target=200.0,
                    business_priority="critical"
                ),
                tags={"metric": "response_time"},
                is_anomalous=False,
                threshold_warning=500.0,
                threshold_critical=1000.0
            ),
            StandardMetric(
                timestamp=timestamp,
                source="appdynamics",
                metric_type=MetricType.APPLICATION,
                entity=EntityInfo(
                    id=entity_id,
                    name=self.entities[entity_id]["name"],
                    type=self.entities[entity_id]["type"],
                    environment="production",
                    criticality=self.entities[entity_id]["criticality"],
                    team=self.entities[entity_id]["team"]
                ),
                value=float(error_rate),
                unit="percent",
                business_info=BusinessContext(
                    service_line="payments",
                    revenue_impact=95.0,
                    customer_impact=5,
                    compliance_tag="PCI-DSS",
                    sla_target=200.0,
                    business_priority="critical"
                ),
                tags={"metric": "error_rate"},
                is_anomalous=False,
                threshold_warning=2.0,
                threshold_critical=5.0
            )
        ])
        
        return metrics
    
    def _calculate_health_score(self, entity_id: str, metrics: List[StandardMetric]) -> HealthScore:
        """Calculate health score for an entity."""
        # Extract metric values and ensure they are floats
        cpu_usage = float(next((m.value for m in metrics if m.tags.get("metric") == "cpu_usage"), 50.0))
        memory_usage = float(next((m.value for m in metrics if m.tags.get("metric") == "memory_usage"), 60.0))
        response_time = float(next((m.value for m in metrics if m.tags.get("metric") == "response_time"), 200.0))
        error_rate = float(next((m.value for m in metrics if m.tags.get("metric") == "error_rate"), 1.0))
        
        # Calculate component scores
        availability_score = max(0, 100 - error_rate)
        performance_score = max(0, 100 - (response_time / 10))
        resource_score = max(0, 100 - (cpu_usage * 0.4 + memory_usage * 0.4))
        
        # Composite score
        composite_score = (
            availability_score * 0.25 +
            performance_score * 0.20 +
            (100 - error_rate) * 0.20 +
            resource_score * 0.15 +
            85.0 * 0.20  # Predictive risk (simplified)
        )
        
        # Determine category
        if composite_score >= 90:
            category = HealthCategory.HEALTHY
        elif composite_score >= 70:
            category = HealthCategory.DEGRADED
        elif composite_score >= 50:
            category = HealthCategory.WARNING
        else:
            category = HealthCategory.CRITICAL
        
        # Create health factors
        factors = [
            HealthFactor(
                name="availability",
                value=availability_score,
                weight=0.25,
                impact=availability_score * 0.25,
                threshold_warning=95.0,
                threshold_critical=90.0,
                trend="stable",
                recommendation="Monitor error rates closely"
            ),
            HealthFactor(
                name="performance", 
                value=performance_score,
                weight=0.20,
                impact=performance_score * 0.20,
                threshold_warning=80.0,
                threshold_critical=60.0,
                trend="stable",
                recommendation="Optimize response times"
            ),
            HealthFactor(
                name="error_rate",
                value=100 - error_rate,
                weight=0.20,
                impact=(100 - error_rate) * 0.20,
                threshold_warning=98.0,
                threshold_critical=95.0,
                trend="stable",
                recommendation="Investigate error patterns"
            ),
            HealthFactor(
                name="resource_utilization",
                value=resource_score,
                weight=0.15,
                impact=resource_score * 0.15,
                threshold_warning=70.0,
                threshold_critical=50.0,
                trend="stable",
                recommendation="Monitor resource usage"
            )
        ]
        
        return HealthScore(
            entity_id=entity_id,
            timestamp=datetime.now(timezone.utc),
            score=composite_score,
            category=category,
            factors=factors,
            trend=HealthTrend(direction="stable", slope=0.0, confidence=0.8, duration_hours=1.0),
            confidence=0.85,
            last_updated=datetime.now(timezone.utc),
            data_freshness_minutes=5,
            business_impact_score=composite_score,
            revenue_risk=max(0, 100 - composite_score),
            customer_impact=5 if composite_score < 50 else 4 if composite_score < 70 else 3 if composite_score < 90 else 1
        )
    
    def _generate_prediction(self, entity_id: str, prediction_type: str) -> PredictionResponse:
        """Generate prediction for an entity."""
        health_score = self.health_scores.get(entity_id)
        
        if not health_score:
            return PredictionResponse(
                entity_id=entity_id,
                prediction_type=PredictionType.FAILURE,
                timestamp=datetime.now(timezone.utc),
                time_horizon_hours=24,
                probability=0.1,
                confidence=0.5,
                risk_level=RiskLevel.LOW,
                recommendations=["Monitor system closely"]
            )
        
        # Calculate failure probability based on health score
        failure_probability = max(0, (100 - health_score.score) / 100)
        
        # Determine risk level
        if failure_probability > 0.7:
            risk_level = RiskLevel.CRITICAL
        elif failure_probability > 0.5:
            risk_level = RiskLevel.HIGH
        elif failure_probability > 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Generate recommendations
        recommendations = []
        if health_score.score < 70:
            recommendations.extend([
                "Check system logs for errors",
                "Verify resource utilization",
                "Consider scaling up resources"
            ])
        
        return PredictionResponse(
            entity_id=entity_id,
            prediction_type=PredictionType.FAILURE,
            timestamp=datetime.now(timezone.utc),
            time_horizon_hours=24,
            probability=failure_probability,
            confidence=0.85,
            risk_level=risk_level,
            predicted_time=datetime.now(timezone.utc) + timedelta(hours=24) if failure_probability > 0.5 else None,
            time_to_event_hours=24 if failure_probability > 0.5 else None,
            contributing_factors=[
                {"factor": "health_score", "value": health_score.score},
                {"factor": "error_rate", "value": "high" if health_score.score < 70 else "normal"}
            ],
            recommendations=recommendations,
            model_version="1.0",
            model_accuracy=0.85
        )
    
    def _process_conversation(self, message: str) -> ConversationResponse:
        """Process conversational query."""
        # Simple keyword-based responses
        message_lower = message.lower()
        
        if "health" in message_lower:
            response = "The payment system health is currently being monitored. The payment gateway shows good health (85%), fraud detection is stable (92%), and the database is performing well (88%)."
        elif "slow" in message_lower or "performance" in message_lower:
            response = "I've detected some performance degradation in the payment gateway. Response times have increased by 15% in the last hour. This may be due to increased load or resource constraints."
        elif "error" in message_lower or "failure" in message_lower:
            response = "The error rate is currently at 2.3%, which is within normal limits. However, I'm monitoring for any unusual patterns that could indicate potential issues."
        elif "prediction" in message_lower or "forecast" in message_lower:
            response = "Based on current trends, the system is predicted to remain stable for the next 24 hours. The ML models show a 15% probability of performance degradation, which is low risk."
        else:
            response = "I'm here to help you monitor the payment system health. You can ask me about system performance, health scores, predictions, or any issues you're experiencing."
        
        # Generate insights
        insights = [
            Insight(
                type=InsightType.TREND,
                content="System health has been stable over the last 24 hours",
                confidence=0.9,
                actionable=True,
                entities=["payment-gateway", "fraud-detection", "payment-db"],
                metrics=["health_score", "response_time"],
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        return ConversationResponse(
            session_id="demo",
            response=response,
            insights=insights,
            follow_up_questions=[
                "Would you like me to investigate any specific issues?",
                "Should I generate a detailed health report?",
                "Would you like to see the prediction models?"
            ],
            recommended_actions=[
                "Continue monitoring system health",
                "Review resource utilization trends",
                "Check for any recent deployments"
            ],
            confidence=0.85,
            processing_time_ms=150.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _run_simulation(self):
        """Run background simulation."""
        while self.running:
            try:
                # Generate metrics for all entities
                for entity_id in self.entities:
                    metrics = self._generate_metrics(entity_id)
                    self.metrics_history[entity_id] = metrics
                    
                    # Calculate health score
                    health_score = self._calculate_health_score(entity_id, metrics)
                    self.health_scores[entity_id] = health_score
                    
                    # Generate prediction
                    prediction = self._generate_prediction(entity_id, "failure")
                    self.predictions[entity_id] = prediction
                
                # Toggle anomaly mode occasionally
                if np.random.random() < 0.01:  # 1% chance per cycle
                    self.anomaly_mode = not self.anomaly_mode
                    logger.info(f"Anomaly mode: {'ON' if self.anomaly_mode else 'OFF'}")
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in simulation: {e}")
                time.sleep(5)

# Initialize demo
demo = StandaloneDemo()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Predictive System Health Platform - Demo",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health/{entity_id}",
            "health_summary": "/health/summary", 
            "predict": "/predict",
            "chat": "/chat",
            "entities": "/entities",
            "metrics": "/metrics/{entity_id}"
        }
    }

@app.get("/health/{entity_id}")
async def get_entity_health(entity_id: str):
    """Get health score for an entity."""
    if entity_id not in demo.health_scores:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return demo.health_scores[entity_id]

@app.get("/health/summary")
async def get_health_summary():
    """Get health summary across all entities."""
    if not demo.health_scores:
        return HealthSummary(
            timestamp=datetime.now(timezone.utc),
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
    
    # Calculate summary
    total_entities = len(demo.health_scores)
    healthy_count = len([h for h in demo.health_scores.values() if h.category == HealthCategory.HEALTHY])
    degraded_count = len([h for h in demo.health_scores.values() if h.category == HealthCategory.DEGRADED])
    warning_count = len([h for h in demo.health_scores.values() if h.category == HealthCategory.WARNING])
    critical_count = len([h for h in demo.health_scores.values() if h.category == HealthCategory.CRITICAL])
    
    average_health_score = np.mean([h.score for h in demo.health_scores.values()])
    weighted_health_score = np.mean([h.score * h.entity.criticality for h in demo.health_scores.values()])
    
    total_revenue_risk = sum([h.revenue_risk for h in demo.health_scores.values()])
    affected_customers = sum([h.customer_impact for h in demo.health_scores.values() if h.customer_impact > 1])
    
    top_issues = []
    for entity_id, health_score in sorted(demo.health_scores.items(), key=lambda x: x[1].score)[:3]:
        if health_score.score < 90:
            top_issues.append(f"{entity_id}: {health_score.score:.1f}%")
    
    critical_entities = [entity_id for entity_id, health_score in demo.health_scores.items() 
                        if health_score.category == HealthCategory.CRITICAL]
    
    return HealthSummary(
        timestamp=datetime.now(timezone.utc),
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

@app.post("/predict")
async def generate_prediction(request: PredictionRequest):
    """Generate prediction."""
    prediction = demo._generate_prediction(request.entity_id, request.prediction_type)
    return prediction

@app.post("/chat")
async def chat(request: ConversationRequest):
    """Process chat message."""
    response = demo._process_conversation(request.message)
    return response

@app.get("/entities")
async def get_entities():
    """Get list of available entities."""
    return list(demo.entities.values())

@app.get("/metrics/{entity_id}")
async def get_entity_metrics(entity_id: str):
    """Get metrics for an entity."""
    if entity_id not in demo.metrics_history:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    metrics = demo.metrics_history[entity_id]
    return {
        "entity_id": entity_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": [
            {
                "name": m.tags.get("metric", "unknown"),
                "value": m.value,
                "unit": m.unit,
                "timestamp": m.timestamp.isoformat()
            }
            for m in metrics
        ]
    }

@app.get("/demo/anomaly")
async def toggle_anomaly():
    """Toggle anomaly mode for demo purposes."""
    demo.anomaly_mode = not demo.anomaly_mode
    return {
        "anomaly_mode": demo.anomaly_mode,
        "message": f"Anomaly mode {'enabled' if demo.anomaly_mode else 'disabled'}"
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Predictive System Health Platform Demo")
    logger.info("📊 Dashboard will be available at: http://localhost:8000")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    logger.info("🔧 Demo endpoints:")
    logger.info("  - GET /health/payment-gateway")
    logger.info("  - GET /health/summary")
    logger.info("  - POST /predict")
    logger.info("  - POST /chat")
    logger.info("  - GET /demo/anomaly (toggle anomaly mode)")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 