from pydantic import BaseModel
from typing import List

class RiskItem(BaseModel):
    rank: int
    category: str
    title: str
    severity: str
    confidence: float
    probability_next_30d: float
    evidence: List[str]
    affected_entities: List[str]
    affected_geo: List[str]
    causal_chain: str
    recommended_action: str

class SoftRisk(BaseModel):
    category: str
    title: str
    note: str

class RiskReport(BaseModel):
    analysed_at: str
    domain: str
    top_risks: List[RiskItem]
    soft_risks: List[SoftRisk]
    data_quality_note: str
    model_used: str
    layer1_completeness: float
    layer2_enriched_at: str
    aggregate_sentiment: str
    avg_reliability: float