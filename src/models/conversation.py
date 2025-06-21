"""
Conversation data models for the Predictive System Health Platform.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages in conversations."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class InsightType(str, Enum):
    """Types of insights that can be generated."""
    ANOMALY = "anomaly"
    TREND = "trend"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"


class Insight(BaseModel):
    """Insight generated from conversation analysis."""
    type: InsightType = Field(..., description="Type of insight")
    content: str = Field(..., description="Insight content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in insight")
    actionable: bool = Field(..., description="Whether insight is actionable")
    entities: List[str] = Field(default_factory=list, description="Related entities")
    metrics: List[str] = Field(default_factory=list, description="Related metrics")
    timestamp: datetime = Field(..., description="Insight timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Recommendation(BaseModel):
    """Actionable recommendation."""
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    priority: str = Field(..., description="Priority level")
    effort: str = Field(..., description="Effort required")
    impact: str = Field(..., description="Expected impact")
    actions: List[str] = Field(..., description="Specific actions to take")
    entities: List[str] = Field(default_factory=list, description="Affected entities")
    estimated_time_minutes: Optional[int] = Field(None, description="Estimated time to implement")


class ConversationRequest(BaseModel):
    """Request for conversational interaction."""
    session_id: str = Field(..., description="Conversation session ID")
    message: str = Field(..., description="User message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    include_history: bool = Field(default=True, description="Include conversation history")
    max_tokens: Optional[int] = Field(None, description="Maximum response tokens")


class ConversationResponse(BaseModel):
    """Response from conversational AI."""
    session_id: str = Field(..., description="Conversation session ID")
    response: str = Field(..., description="AI response")
    insights: List[Insight] = Field(default_factory=list, description="Generated insights")
    follow_up_questions: List[str] = Field(default_factory=list, description="Follow-up questions")
    related_incidents: List[Dict[str, Any]] = Field(default_factory=list, description="Related incidents")
    recommended_actions: List[Recommendation] = Field(default_factory=list, description="Recommended actions")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationMessage(BaseModel):
    """Individual message in a conversation."""
    message_id: str = Field(..., description="Unique message ID")
    session_id: str = Field(..., description="Session ID")
    type: MessageType = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationSession(BaseModel):
    """Conversation session information."""
    session_id: str = Field(..., description="Unique session ID")
    user_id: Optional[str] = Field(None, description="User identifier")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    message_count: int = Field(..., description="Number of messages")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
    is_active: bool = Field(..., description="Whether session is active")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationHistory(BaseModel):
    """Complete conversation history."""
    session_id: str = Field(..., description="Session ID")
    messages: List[ConversationMessage] = Field(..., description="Message history")
    summary: Optional[str] = Field(None, description="Conversation summary")
    key_topics: List[str] = Field(default_factory=list, description="Key topics discussed")
    entities_mentioned: List[str] = Field(default_factory=list, description="Entities mentioned")
    actions_taken: List[str] = Field(default_factory=list, description="Actions taken")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 