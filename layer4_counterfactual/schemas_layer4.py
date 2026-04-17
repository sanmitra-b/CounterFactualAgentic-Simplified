from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ObservedRiskState(BaseModel):
    """Observed risk instance from Layer 3, augmented with causal-state features."""

    risk_id: str
    rank: int
    category: str
    title: str
    severity_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    probability_next_30d: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    affected_entities: List[str] = Field(default_factory=list)
    affected_geo: List[str] = Field(default_factory=list)
    causal_chain: str = ""
    recommended_action: str = ""

    # Normalized causal variables in [0, 1]
    port_congestion: float = Field(ge=0.0, le=1.0)
    weather_severity: float = Field(ge=0.0, le=1.0)
    geopolitical_tension: float = Field(ge=0.0, le=1.0)
    shipping_delay: float = Field(ge=0.0, le=1.0)
    supplier_reliability: float = Field(ge=0.0, le=1.0)
    inventory_shortage: float = Field(ge=0.0, le=1.0)
    demand_shock: float = Field(ge=0.0, le=1.0)
    risk_severity: float = Field(ge=0.0, le=1.0)


class InterventionParams(BaseModel):
    """A single do(.) attempt on one causal variable."""

    variable: str
    intervened_value: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    iteration: int = Field(ge=1)


class CounterfactualResult(BaseModel):
    """Result of one counterfactual intervention attempt."""

    risk_id: str
    risk_title: str
    category: str
    iteration: int
    intervention: InterventionParams
    observed_risk_severity: float
    cf_risk_severity_mean: float
    ite_mean: float
    ite_ci95_low: float
    ite_ci95_high: float
    probability_of_improvement: float
    explained_causal_path: str
    threshold_cleared: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentMemory(BaseModel):
    """Per-risk memory across iterative attempts."""

    risk_id: str
    risk_title: str
    tried_interventions: List[InterventionParams] = Field(default_factory=list)
    all_results: List[CounterfactualResult] = Field(default_factory=list)
    best_result: Optional[CounterfactualResult] = None
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
