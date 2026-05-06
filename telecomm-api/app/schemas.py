from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from datetime import datetime


class ComplaintAnalysis(BaseModel):
    """Structured analysis of a customer complaint — enforced by function calling."""
    model_config = ConfigDict(extra='forbid')

    category: Literal[
        'billing', 'network_outage', 'slow_data',
        'roaming', 'device', 'customer_service', 'other'
    ] = Field(..., description='Primary complaint category')

    priority: Literal['low', 'medium', 'high', 'critical'] = Field(
        ..., description='Urgency — critical means customer may churn imminently'
    )

    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description='Sentiment: -1.0=very negative, 0=neutral, 1.0=very positive'
    )

    account_at_risk: bool = Field(
        ..., description='True if customer explicitly threatens to cancel or leave'
    )

    key_issue: str = Field(
        ..., max_length=200,
        description='One-sentence summary of the core complaint'
    )

    suggested_resolution: str = Field(
        ..., max_length=400,
        description='Recommended first action for customer service agent'
    )

    requires_escalation: bool = Field(
        ..., description='True if this needs a supervisor or technical team'
    )


class AnalyseRequest(BaseModel):
    complaint_text: str = Field(..., min_length=10, max_length=5000)
    customer_id:    str = Field(..., min_length=1)


class AnalyseResponse(BaseModel):
    customer_id:   str
    analysis:      ComplaintAnalysis
    model_used:    str
    latency_ms:    float
    timestamp:     datetime = Field(default_factory=datetime.utcnow)
