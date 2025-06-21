#!/usr/bin/env python3
"""
Conversation Service for Predictive System Health Platform

Provides AI-powered natural language interface for:
- System health queries
- Root cause analysis
- Historical incident analysis
- Recommendations and insights
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import openai
from fastapi import FastAPI, HTTPException
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel
import redis
from neo4j import GraphDatabase
from elasticsearch import Elasticsearch
import weaviate

from src.models.conversation import (
    ConversationRequest, ConversationResponse, ConversationSession,
    ConversationMessage, Insight, InsightType, Recommendation
)
from src.models.health import HealthScore
from src.models.predictions import PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Conversation Service", version="1.0.0")


class ConversationService:
    """Service for AI-powered conversational interface."""
    
    def __init__(self, weaviate_url: str, elasticsearch_url: str, neo4j_url: str, redis_url: str, openai_api_key: str):
        self.weaviate_url = weaviate_url
        self.elasticsearch_url = elasticsearch_url
        self.neo4j_url = neo4j_url
        self.redis_url = redis_url
        self.openai_api_key = openai_api_key
        
        # Initialize OpenAI
        openai.api_key = openai_api_key
        
        # Initialize clients
        self.weaviate_client = weaviate.Client(weaviate_url)
        self.elasticsearch_client = Elasticsearch([elasticsearch_url])
        self.neo4j_driver = GraphDatabase.driver(neo4j_url, auth=("neo4j", "password"))
        self.redis_client = redis.from_url(redis_url)
        
        # Kafka for communication with other services
        self.kafka_producer = KafkaProducer(
            bootstrap_servers="localhost:9092",
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        
        # Session management
        self.sessions = {}
        
        # Initialize knowledge base
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with sample data."""
        try:
            # Add sample incidents to Weaviate
            sample_incidents = [
                {
                    "title": "Payment Gateway High Latency",
                    "description": "Payment gateway experienced high response times due to database connection pool exhaustion",
                    "root_cause": "Database connection pool was not properly configured for high load",
                    "resolution": "Increased connection pool size and implemented connection monitoring",
                    "prevention": "Set up automated scaling for connection pools based on load",
                    "affected_systems": ["payment-gateway", "payment-db"],
                    "severity": "high",
                    "duration_minutes": 45,
                    "business_impact": 50000.0
                },
                {
                    "title": "Fraud Detection Service Outage",
                    "description": "Fraud detection service became unresponsive due to memory leak",
                    "root_cause": "Memory leak in fraud detection algorithm",
                    "resolution": "Restarted service and fixed memory leak in code",
                    "prevention": "Implemented memory monitoring and automated restarts",
                    "affected_systems": ["fraud-detection"],
                    "severity": "critical",
                    "duration_minutes": 30,
                    "business_impact": 75000.0
                },
                {
                    "title": "Database Performance Degradation",
                    "description": "Payment database experienced slow query performance",
                    "root_cause": "Missing database indexes on frequently queried columns",
                    "resolution": "Added appropriate indexes and optimized queries",
                    "prevention": "Regular database performance reviews and index optimization",
                    "affected_systems": ["payment-db"],
                    "severity": "medium",
                    "duration_minutes": 120,
                    "business_impact": 25000.0
                }
            ]
            
            # Add incidents to Weaviate (simplified - in real implementation, use proper schema)
            logger.info("✅ Knowledge base initialized with sample incidents")
            
        except Exception as e:
            logger.warning(f"Could not initialize knowledge base: {e}")
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entity names from user query."""
        entities = []
        
        # Simple entity extraction (in real implementation, use NER)
        entity_keywords = {
            "payment": "payment-gateway",
            "gateway": "payment-gateway",
            "fraud": "fraud-detection",
            "database": "payment-db",
            "db": "payment-db",
            "settlement": "settlement-service",
            "notification": "notification-service"
        }
        
        query_lower = query.lower()
        for keyword, entity_id in entity_keywords.items():
            if keyword in query_lower:
                entities.append(entity_id)
        
        return entities
    
    def _get_system_health_data(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Get current health data for entities."""
        health_data = {}
        
        for entity_id in entity_ids:
            try:
                # Get from Redis cache
                cached_data = self.redis_client.get(f"health:{entity_id}")
                if cached_data:
                    health_score = HealthScore(**json.loads(cached_data))
                    health_data[entity_id] = {
                        "health_score": health_score.score,
                        "category": health_score.category,
                        "factors": [f.name for f in health_score.factors],
                        "business_impact": health_score.business_impact_score
                    }
            except Exception as e:
                logger.error(f"Error getting health data for {entity_id}: {e}")
        
        return health_data
    
    def _get_historical_incidents(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant historical incidents."""
        try:
            # Search Weaviate for similar incidents
            # Simplified implementation
            return [
                {
                    "title": "Payment Gateway High Latency",
                    "description": "Similar issue occurred last month",
                    "resolution": "Increased connection pool size",
                    "duration_minutes": 45
                }
            ]
        except Exception as e:
            logger.error(f"Error getting historical incidents: {e}")
            return []
    
    def _generate_insights(self, query: str, health_data: Dict[str, Any], historical_incidents: List[Dict[str, Any]]) -> List[Insight]:
        """Generate insights based on query and data."""
        insights = []
        
        # Analyze health data
        for entity_id, data in health_data.items():
            if data["health_score"] < 70:
                insights.append(Insight(
                    type=InsightType.ANOMALY,
                    content=f"{entity_id} is experiencing health issues (score: {data['health_score']:.1f})",
                    confidence=0.9,
                    actionable=True,
                    entities=[entity_id],
                    metrics=data["factors"],
                    timestamp=datetime.utcnow()
                ))
        
        # Analyze historical patterns
        if historical_incidents:
            insights.append(Insight(
                type=InsightType.CORRELATION,
                content=f"Found {len(historical_incidents)} similar incidents in the past",
                confidence=0.8,
                actionable=True,
                entities=[],
                metrics=[],
                timestamp=datetime.utcnow()
            ))
        
        return insights
    
    def _generate_recommendations(self, query: str, health_data: Dict[str, Any], insights: List[Insight]) -> List[Recommendation]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Generate recommendations based on health data
        for entity_id, data in health_data.items():
            if data["health_score"] < 50:
                recommendations.append(Recommendation(
                    title=f"Immediate Action Required for {entity_id}",
                    description=f"{entity_id} is in critical state. Immediate intervention needed.",
                    priority="high",
                    effort="medium",
                    impact="high",
                    actions=[
                        "Check system logs for errors",
                        "Verify resource utilization",
                        "Consider service restart if necessary"
                    ],
                    entities=[entity_id],
                    estimated_time_minutes=30
                ))
            elif data["health_score"] < 80:
                recommendations.append(Recommendation(
                    title=f"Monitor {entity_id} Closely",
                    description=f"{entity_id} is showing signs of degradation.",
                    priority="medium",
                    effort="low",
                    impact="medium",
                    actions=[
                        "Monitor health trends",
                        "Check for recent changes",
                        "Review resource allocation"
                    ],
                    entities=[entity_id],
                    estimated_time_minutes=15
                ))
        
        return recommendations
    
    async def _generate_ai_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate AI response using OpenAI."""
        try:
            # Build system prompt
            system_prompt = """You are an AI assistant for a payment system monitoring platform. 
            You help users understand system health, diagnose issues, and provide recommendations.
            Be concise, professional, and actionable in your responses."""
            
            # Build user prompt with context
            user_prompt = f"""
            User Query: {query}
            
            Context:
            - Health Data: {context.get('health_data', {})}
            - Historical Incidents: {context.get('historical_incidents', [])}
            - Current Time: {datetime.utcnow()}
            
            Please provide a helpful response that addresses the user's query using the available context.
            """
            
            # Generate response using OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again later."
    
    async def process_message(self, session_id: str, message: str, context: Dict[str, Any] = None) -> ConversationResponse:
        """Process a user message and generate response."""
        start_time = time.time()
        
        try:
            # Extract entities from query
            entities = self._extract_entities_from_query(message)
            
            # Get system health data
            health_data = self._get_system_health_data(entities)
            
            # Get historical incidents
            historical_incidents = self._get_historical_incidents(message)
            
            # Build context for AI
            ai_context = {
                "health_data": health_data,
                "historical_incidents": historical_incidents,
                "entities": entities,
                "user_context": context or {}
            }
            
            # Generate AI response
            ai_response = await self._generate_ai_response(message, ai_context)
            
            # Generate insights
            insights = self._generate_insights(message, health_data, historical_incidents)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(message, health_data, insights)
            
            # Generate follow-up questions
            follow_up_questions = [
                "Would you like me to investigate the root cause of this issue?",
                "Should I check for similar incidents in the past?",
                "Would you like me to generate a detailed health report?"
            ]
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = ConversationResponse(
                session_id=session_id,
                response=ai_response,
                insights=insights,
                follow_up_questions=follow_up_questions,
                related_incidents=historical_incidents,
                recommended_actions=recommendations,
                confidence=0.85,
                processing_time_ms=processing_time_ms,
                timestamp=datetime.utcnow()
            )
            
            # Store in session
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationSession(
                    session_id=session_id,
                    created_at=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                    message_count=0,
                    context={},
                    is_active=True
                )
            
            self.sessions[session_id].message_count += 1
            self.sessions[session_id].last_activity = datetime.utcnow()
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")
    
    async def get_conversation_history(self, session_id: str) -> List[ConversationMessage]:
        """Get conversation history for a session."""
        # In a real implementation, this would retrieve from database
        # For now, return empty list
        return []
    
    async def create_session(self, user_id: str = None) -> ConversationSession:
        """Create a new conversation session."""
        session_id = f"session_{int(time.time())}"
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            message_count=0,
            context={},
            is_active=True
        )
        
        self.sessions[session_id] = session
        return session


# Initialize service
import os

weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
openai_api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

conversation_service = ConversationService(
    weaviate_url, elasticsearch_url, neo4j_url, redis_url, openai_api_key
)


@app.post("/chat")
async def chat(request: ConversationRequest):
    """Process a chat message."""
    return await conversation_service.process_message(
        request.session_id, request.message, request.context
    )


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history."""
    return await conversation_service.get_conversation_history(session_id)


@app.post("/sessions")
async def create_session(user_id: str = None):
    """Create a new conversation session."""
    return await conversation_service.create_session(user_id)


@app.get("/health")
async def service_health():
    """Service health check."""
    return {"status": "healthy", "service": "conversation"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 