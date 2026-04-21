 
from __future__ import annotations
 
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
 
 
# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
 
class PlaybookChunk(BaseModel):
    """One 512-token chunk stored in the vector DB."""
    chunk_id:        str
    playbook_id:     str
    playbook_title:  str
    category:        str          # maps to RISK_CATEGORIES
    intervention_type: str        # reroute | stockpile | hedge | diversify | insure | escalate | monitor
    text:            str
    metadata:        Dict[str, Any] = Field(default_factory=dict)
 
 
class RetrievedChunk(BaseModel):
    """A chunk returned by the vector retriever, with its similarity score."""
    chunk:           PlaybookChunk
    cosine_score:    float = Field(ge=0.0, le=1.0)
    rank:            int   = 1
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MITIGATION SOLUTION
# ─────────────────────────────────────────────────────────────────────────────
 
class MitigationSolution(BaseModel):
    """
    A ranked mitigation solution assembled from retrieved chunks
    and scored by the Confidence Scorer.
    """
    solution_rank:        int
    risk_id:              str
    risk_title:           str
    risk_category:        str
    intervention_type:    str
 
    # Core content
    title:                str
    description:          str
    action_steps:         List[str] = Field(default_factory=list)
 
    # Scoring
    cosine_similarity:    float = Field(ge=0.0, le=1.0)
    confidence_score:     float = Field(ge=0.0, le=1.0)   # composite
    ite_alignment_score:  float = Field(ge=0.0, le=1.0)   # how well it addresses the ITE variable
    severity_weight:      float = Field(ge=0.0, le=1.0)   # boosted by CRITICAL/HIGH
 
    # Provenance
    source_chunks:        List[str] = Field(default_factory=list)   # chunk_ids
    playbook_title:       str = ""
 
    # Layer 4 linkage
    causal_variable:      str = ""    # the do(X=x) variable from Layer 4
    ite_mean:             float = 0.0
    probability_of_improvement: float = 0.0
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PER-RISK SOLUTION MAPPING
# ─────────────────────────────────────────────────────────────────────────────
 
class RiskSolutionMapping(BaseModel):
    """All ranked mitigations for one risk from Layer 4."""
    risk_id:              str
    risk_title:           str
    risk_category:        str
    severity_label:       str
    causal_variable:      str
    ite_mean:             float
    probability_of_improvement: float
    top_mitigations:      List[MitigationSolution] = Field(default_factory=list)
    retrieval_query:      str = ""    # the query sent to the vector DB
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL LAYER 5 BUNDLE
# ─────────────────────────────────────────────────────────────────────────────
 
class RiskSolutionBundle(BaseModel):
    """
    Output of Layer 5. Carries ranked mitigations for all top-5 risks.
    Consumed by Layer 6 (Human-readable cause-effect output).
    """
    domain:               str = "supply_chain"
    mapped_at:            str = ""
    layer4_source:        str = ""
    total_risks_mapped:   int = 0
    total_solutions:      int = 0
 
    mappings:             List[RiskSolutionMapping] = Field(default_factory=list)
 
    # Portfolio-level ranked list (flat, sorted by confidence_score desc)
    ranked_mitigations:   List[MitigationSolution] = Field(default_factory=list)
 
    vector_db_backend:    str = "chromadb"   # chromadb | faiss
    embedding_model:      str = "all-MiniLM-L6-v2"
    chunk_size_tokens:    int = 512
 