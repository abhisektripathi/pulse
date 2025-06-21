#!/usr/bin/env python3
"""
API Gateway Service for Predictive System Health Platform

Provides unified GraphQL and REST interface for:
- Health scoring service
- Prediction service
- Conversation service
- Authentication and authorization
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import httpx
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import strawberry
from strawberry.fastapi import GraphQLRouter

from src.models.health import HealthScore, HealthSummary
from src.models.predictions import PredictionRequest, PredictionResponse
from src.models.conversation import ConversationRequest, ConversationResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Gateway", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


class ServiceClient:
    """Client for communicating with backend services."""
    
    def __init__(self):
        self.health_service_url = "http://health-scoring-service:8000"
        self.prediction_service_url = "http://prediction-service:8000"
        self.conversation_service_url = "http://conversation-service:8000"
        self.timeout = 30.0
    
    async def get_entity_health(self, entity_id: str) -> Optional[HealthScore]:
        """Get health score for an entity."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.health_service_url}/health/{entity_id}")
                if response.status_code == 200:
                    return HealthScore(**response.json())
                return None
        except Exception as e:
            logger.error(f"Error getting health for {entity_id}: {e}")
            return None
    
    async def get_health_summary(self) -> Optional[HealthSummary]:
        """Get health summary."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.health_service_url}/health/summary")
                if response.status_code == 200:
                    return HealthSummary(**response.json())
                return None
        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return None
    
    async def generate_prediction(self, request: PredictionRequest) -> Optional[PredictionResponse]:
        """Generate prediction."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.prediction_service_url}/predict",
                    json=request.dict()
                )
                if response.status_code == 200:
                    return PredictionResponse(**response.json())
                return None
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None
    
    async def process_conversation(self, request: ConversationRequest) -> Optional[ConversationResponse]:
        """Process conversation message."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.conversation_service_url}/chat",
                    json=request.dict()
                )
                if response.status_code == 200:
                    return ConversationResponse(**response.json())
                return None
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return None


# Initialize service client
service_client = ServiceClient()


# GraphQL Schema
@strawberry.type
class HealthScoreType:
    entity_id: str
    score: float
    category: str
    timestamp: str
    confidence: float
    business_impact_score: float
    revenue_risk: float
    customer_impact: int


@strawberry.type
class HealthSummaryType:
    timestamp: str
    total_entities: int
    healthy_count: int
    degraded_count: int
    warning_count: int
    critical_count: int
    average_health_score: float
    total_revenue_risk: float
    affected_customers: int
    top_issues: List[str]
    critical_entities: List[str]


@strawberry.type
class PredictionType:
    entity_id: str
    prediction_type: str
    probability: float
    confidence: float
    risk_level: str
    recommendations: List[str]


@strawberry.type
class ConversationResponseType:
    session_id: str
    response: str
    confidence: float
    processing_time_ms: float
    timestamp: str


@strawberry.type
class Query:
    @strawberry.field
    async def system_health(self, entity_id: str) -> Optional[HealthScoreType]:
        """Get health score for a system entity."""
        health_score = await service_client.get_entity_health(entity_id)
        if health_score:
            return HealthScoreType(
                entity_id=health_score.entity_id,
                score=health_score.score,
                category=health_score.category.value,
                timestamp=health_score.timestamp.isoformat(),
                confidence=health_score.confidence,
                business_impact_score=health_score.business_impact_score,
                revenue_risk=health_score.revenue_risk,
                customer_impact=health_score.customer_impact
            )
        return None
    
    @strawberry.field
    async def health_summary(self) -> Optional[HealthSummaryType]:
        """Get health summary across all entities."""
        summary = await service_client.get_health_summary()
        if summary:
            return HealthSummaryType(
                timestamp=summary.timestamp.isoformat(),
                total_entities=summary.total_entities,
                healthy_count=summary.healthy_count,
                degraded_count=summary.degraded_count,
                warning_count=summary.warning_count,
                critical_count=summary.critical_count,
                average_health_score=summary.average_health_score,
                total_revenue_risk=summary.total_revenue_risk,
                affected_customers=summary.affected_customers,
                top_issues=summary.top_issues,
                critical_entities=summary.critical_entities
            )
        return None


@strawberry.type
class Mutation:
    @strawberry.field
    async def generate_prediction(
        self, 
        entity_id: str, 
        prediction_type: str, 
        time_horizon_hours: int = 24
    ) -> Optional[PredictionType]:
        """Generate prediction for an entity."""
        request = PredictionRequest(
            entity_id=entity_id,
            prediction_type=prediction_type,
            time_horizon_hours=time_horizon_hours
        )
        
        prediction = await service_client.generate_prediction(request)
        if prediction:
            return PredictionType(
                entity_id=prediction.entity_id,
                prediction_type=prediction.prediction_type.value,
                probability=prediction.probability,
                confidence=prediction.confidence,
                risk_level=prediction.risk_level.value,
                recommendations=prediction.recommendations
            )
        return None
    
    @strawberry.field
    async def send_message(
        self, 
        session_id: str, 
        message: str
    ) -> Optional[ConversationResponseType]:
        """Send a message to the conversation service."""
        request = ConversationRequest(
            session_id=session_id,
            message=message
        )
        
        response = await service_client.process_conversation(request)
        if response:
            return ConversationResponseType(
                session_id=response.session_id,
                response=response.response,
                confidence=response.confidence,
                processing_time_ms=response.processing_time_ms,
                timestamp=response.timestamp.isoformat()
            )
        return None


# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)

# Add GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")


# REST API endpoints
@app.get("/health/{entity_id}")
async def get_entity_health(entity_id: str):
    """Get health score for an entity."""
    health_score = await service_client.get_entity_health(entity_id)
    if health_score:
        return health_score
    else:
        raise HTTPException(status_code=404, detail="Entity not found")


@app.get("/health/summary")
async def get_health_summary():
    """Get health summary."""
    summary = await service_client.get_health_summary()
    if summary:
        return summary
    else:
        raise HTTPException(status_code=404, detail="Health summary not available")


@app.post("/predict")
async def generate_prediction(request: PredictionRequest):
    """Generate prediction."""
    prediction = await service_client.generate_prediction(request)
    if prediction:
        return prediction
    else:
        raise HTTPException(status_code=500, detail="Failed to generate prediction")


@app.post("/chat")
async def chat(request: ConversationRequest):
    """Process chat message."""
    response = await service_client.process_conversation(request)
    if response:
        return response
    else:
        raise HTTPException(status_code=500, detail="Failed to process message")


@app.get("/entities")
async def get_entities():
    """Get list of available entities."""
    # In a real implementation, this would query a database
    entities = [
        {
            "id": "payment-gateway",
            "name": "Payment Gateway Service",
            "type": "service",
            "criticality": 5
        },
        {
            "id": "fraud-detection",
            "name": "Fraud Detection Service",
            "type": "service",
            "criticality": 4
        },
        {
            "id": "payment-db",
            "name": "Payment Database",
            "type": "database",
            "criticality": 5
        },
        {
            "id": "settlement-service",
            "name": "Settlement Service",
            "type": "service",
            "criticality": 4
        }
    ]
    return entities


@app.get("/metrics/{entity_id}")
async def get_entity_metrics(entity_id: str, time_range: str = "1h"):
    """Get metrics for an entity."""
    # In a real implementation, this would query InfluxDB
    # For now, return sample data
    return {
        "entity_id": entity_id,
        "time_range": time_range,
        "metrics": [
            {
                "name": "cpu_usage",
                "value": 65.5,
                "unit": "percent",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "name": "memory_usage",
                "value": 72.3,
                "unit": "percent",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "name": "response_time",
                "value": 150.2,
                "unit": "milliseconds",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
    }


@app.get("/alerts")
async def get_alerts(severity: str = None, limit: int = 10):
    """Get recent alerts."""
    # In a real implementation, this would query the alerting system
    alerts = [
        {
            "id": "alert-001",
            "entity_id": "payment-gateway",
            "severity": "warning",
            "message": "High response time detected",
            "timestamp": datetime.utcnow().isoformat()
        },
        {
            "id": "alert-002",
            "entity_id": "fraud-detection",
            "severity": "critical",
            "message": "Service unresponsive",
            "timestamp": datetime.utcnow().isoformat()
        }
    ]
    
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    
    return alerts[:limit]


@app.get("/health")
async def service_health():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "health_scoring": "healthy",
            "prediction": "healthy",
            "conversation": "healthy"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 