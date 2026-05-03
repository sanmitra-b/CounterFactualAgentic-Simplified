from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED ENUMERATIONS  (typed literals enforced at validation time)
# ═══════════════════════════════════════════════════════════════════════════════

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class Feasibility(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class InterventionType(str, Enum):
    POLICY      = "POLICY"
    SUPPLY      = "SUPPLY"
    FINANCIAL   = "FINANCIAL"
    OPERATIONAL = "OPERATIONAL"
    REGULATORY  = "REGULATORY"


class SolutionType(str, Enum):
    FRAMEWORK   = "FRAMEWORK"
    TECHNOLOGY  = "TECHNOLOGY"
    PROCESS     = "PROCESS"
    REGULATION  = "REGULATION"
    PARTNERSHIP = "PARTNERSHIP"


class TimeHorizon(str, Enum):
    SHORT  = "SHORT"   # < 3 months
    MEDIUM = "MEDIUM"  # 3 – 12 months
    LONG   = "LONG"    # > 12 months


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 CONTRACTS — Analyst Agent
# ═══════════════════════════════════════════════════════════════════════════════

class AnalystWorkOrder(BaseModel):
    """
    Contract the Supervisor sends to the Analyst Agent.
    Carries exactly one risk item; the Agent must not touch anything else.
    """
    request_id:          str                  = Field(..., description="Unique ID for this work order, e.g. 'l4_risk_1'")
    domain:              str
    risk_rank:           int                  = Field(..., ge=1, le=10)
    risk_title:          str                  = Field(..., min_length=3)
    risk_category:       str
    severity:            Severity
    confidence:          float                = Field(..., ge=0.0, le=1.0)
    probability_next_30d: float               = Field(..., ge=0.0, le=1.0)
    causal_chain:        str
    recommended_action:  str
    affected_entities:   List[str]            = Field(default_factory=list)
    affected_geo:        List[str]            = Field(default_factory=list)
    evidence:            List[str]            = Field(default_factory=list)
    n_scenarios:         int                  = Field(default=2, ge=1, le=5)


class CounterfactualScenario(BaseModel):
    """
    A single simulation result produced by the Analyst Agent / Engine.
    All numeric fields are range-validated; enums enforce allowed literals.
    """
    scenario_id:                str
    risk_rank:                  int           = Field(..., ge=1)
    risk_title:                 str
    intervention:               str           = Field(..., min_length=10)
    intervention_type:          InterventionType
    baseline_probability:       float         = Field(..., ge=0.0, le=1.0)
    counterfactual_probability: float         = Field(..., ge=0.0, le=1.0)
    delta_probability:          float         = Field(..., ge=-1.0, le=1.0)
    baseline_severity:          Severity
    counterfactual_severity:    Severity
    feasibility:                Feasibility
    feasibility_rationale:      str           = Field(..., min_length=10)
    assumptions:                List[str]     = Field(..., min_length=2)
    second_order_effects:       List[str]     = Field(..., min_length=2)
    estimated_cost_usd:         Optional[str] = None
    time_to_impact_days:        int           = Field(..., ge=1, le=3650)
    confidence:                 float         = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_delta_consistency(self) -> "CounterfactualScenario":
        expected = round(self.counterfactual_probability - self.baseline_probability, 4)
        if abs(expected - self.delta_probability) > 0.01:
            # Auto-correct rather than reject; keeps pipeline alive
            self.delta_probability = expected
        return self


class AnalystWorkResult(BaseModel):
    """
    Contract the Analyst Agent returns to the Supervisor.
    The Supervisor rejects this object if validation fails.
    """
    request_id:  str
    risk_rank:   int
    scenarios:   List[CounterfactualScenario] = Field(..., min_length=1)
    model_used:  str
    error:       Optional[str] = None       # populated only on partial failure
    model_config = {"protected_namespaces": ()}


class CounterfactualBundle(BaseModel):
    """Final Layer 4 output written to disk by the Supervisor."""
    simulated_at:        str
    domain:              str
    model_used:          str
    layer3_analysed_at:  str
    scenarios:           List[CounterfactualScenario]
    summary_note:        str
    total_risks:         int
    total_scenarios:     int
    feasibility_dist:    dict[str, int]
    avg_delta:           float
    model_config = {"protected_namespaces": ()}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5 CONTRACTS — Librarian Agent + Solution Synthesis
# ═══════════════════════════════════════════════════════════════════════════════

class LibrarianWorkOrder(BaseModel):
    """
    Contract the Supervisor sends to the Librarian Agent.
    The Librarian has READ-ONLY access and receives only what it needs to retrieve.
    It must NOT receive the full risk report — only query text + k.
    """
    request_id:  str
    query:       str  = Field(..., min_length=5, description="Free-text retrieval query built by the Supervisor")
    top_k:       int  = Field(default=4, ge=1, le=20)


class KBChunk(BaseModel):
    """A single knowledge-base document returned by the Librarian."""
    chunk_id:         str   = Field(..., alias="id")
    title:            str
    tags:             List[str]            = Field(default_factory=list)
    body:             str
    references:       List[str]            = Field(default_factory=list)
    retrieval_score:  float                = Field(..., ge=0.0, le=1.0, alias="_retrieval_score")

    model_config = {"populate_by_name": True}


class LibrarianWorkResult(BaseModel):
    """
    Contract the Librarian Agent returns to the Supervisor.
    Contains ONLY retrieved KB chunks — no LLM generation, no mutations.
    """
    request_id: str
    chunks:     List[KBChunk]
    query_used: str
    error:      Optional[str] = None


class SolutionSynthesisOrder(BaseModel):
    """
    Internal contract the Supervisor assembles for the LLM synthesis call.
    Combines risk context + counterfactual + retrieved chunks into one typed payload.
    """
    request_id:          str
    domain:              str
    risk_rank:           int
    risk_title:          str
    risk_category:       str
    severity:            Severity
    probability_next_30d: float
    causal_chain:        str
    recommended_action:  str
    affected_entities:   List[str]
    affected_geo:        List[str]
    linked_scenario:     Optional[dict[str, Any]] = None   # CounterfactualScenario as dict
    kb_chunks:           List[KBChunk]
    solution_index:      int


class SolutionMatch(BaseModel):
    """A single synthesised solution; schema validated before Supervisor accepts it."""
    solution_id:              str
    risk_rank:                int           = Field(..., ge=1)
    scenario_id:              Optional[str] = None
    risk_title:               str
    solution_title:           str           = Field(..., min_length=3)
    solution_type:            SolutionType
    description:              str           = Field(..., min_length=20)
    source_chunks:            List[str]     = Field(..., min_length=1)
    relevance_score:          float         = Field(..., ge=0.0, le=1.0)
    implementation_steps:     List[str]     = Field(..., min_length=2)
    kpis:                     List[str]     = Field(..., min_length=1)
    time_horizon:             TimeHorizon
    estimated_cost_usd:       Optional[str] = None
    risk_reduction_estimate:  Optional[str] = None
    dependencies:             List[str]     = Field(default_factory=list)
    references:               List[str]     = Field(default_factory=list)

    @field_validator("source_chunks")
    @classmethod
    def _chunks_must_reference_kb(cls, v: List[str]) -> List[str]:
        """Chunk IDs must look like 'kb_NNN' — rejects hallucinated references."""
        import re
        pattern = re.compile(r"^kb_\d+$")
        bad = [c for c in v if not pattern.match(c)]
        if bad:
            raise ValueError(
                f"source_chunks contains IDs that don't match KB format 'kb_NNN': {bad}. "
                "Only cite chunks that were actually retrieved."
            )
        return v


class SolutionMappingReport(BaseModel):
    """Final Layer 5 output written to disk by the Supervisor."""
    mapped_at:      str
    domain:         str
    model_used:     str
    kb_size:        int
    rag_method:     str
    solutions:      List[SolutionMatch]
    coverage_note:  str
    risks_covered:  List[int]
    risks_missed:   List[int]
    model_config = {"protected_namespaces": ()}
