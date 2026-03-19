from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field

# ================================
# STATE MODEL FOR LANGGRAPH
# ================================

@dataclass
class AgentState:
    """State object passed through LangGraph nodes."""
    query: str
    sanitized_query: str = ""
    category: str = ""
    sentiment: str = ""
    confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    all_intents: List[str] = field(default_factory=list)
    response: str = ""
    path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "query": self.query,
            "sanitized_query": self.sanitized_query,
            "category": self.category,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "entities": self.entities,
            "all_intents": self.all_intents,
            "response": self.response,
            "path": self.path,
            "metadata": self.metadata,
        }

# ================================
# API MODELS
# ================================

class QueryRequest(BaseModel):
    """API request model for customer queries."""
    query: str = Field(..., min_length=1, max_length=5000, description="Customer query")
    customer_id: Optional[str] = None
    session_id: Optional[str] = None

class IntentResult(BaseModel):
    """Result of intent classification."""
    intent: str
    confidence: float

class EntityResult(BaseModel):
    """Result of entity extraction."""
    entity_type: str
    value: str
    confidence: float

class SentimentResult(BaseModel):
    """Result of sentiment analysis."""
    sentiment: str  # positive, negative, neutral, critical
    confidence: float
    score: float  # -1 to 1

class QueryResponse(BaseModel):
    """API response model for queries."""
    query: str
    category: str
    sentiment: str
    confidence: float
    response: str
    path: str
    entities: Dict[str, List[Dict[str, Any]]]
    intents: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[float] = None

# ================================
# DATABASE MODELS
# ================================

class QueryLog(BaseModel):
    """Model for logged queries."""
    query: str
    sanitized_query: str
    category: str
    sentiment: str
    confidence: float
    response: str
    path: str
    entities: Dict[str, Any]
    intents: List[str]
    customer_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolved: bool = False
    escalated: bool = False
    feedback_score: Optional[float] = None

# ================================
# RAG MODELS
# ================================

class Document(BaseModel):
    """Model for documents in RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class RetrievalResult(BaseModel):
    """Result of document retrieval."""
    documents: List[Document]
    scores: List[float]
    query: str
