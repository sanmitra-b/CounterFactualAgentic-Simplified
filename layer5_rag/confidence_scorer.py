
from __future__ import annotations
 
from typing import List
 
from schemas_layer5 import MitigationSolution, RetrievedChunk
 
# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
 
W_COSINE    = 0.40
W_CATEGORY  = 0.25
W_ITE_ALIGN = 0.20
W_SEVERITY  = 0.15
 
# Severity → score mapping
SEVERITY_SCORE = {
    "CRITICAL": 1.00,
    "HIGH":     0.80,
    "MEDIUM":   0.55,
    "LOW":      0.30,
}
 
# Causal variable → compatible intervention types
VARIABLE_TO_INTERVENTIONS = {
    "port_congestion":      {"reroute", "monitor", "escalate"},
    "weather_severity":     {"stockpile", "reroute", "monitor"},
    "geopolitical_tension": {"diversify", "hedge", "escalate", "stockpile"},
    "shipping_delay":       {"stockpile", "reroute", "escalate"},
    "supplier_reliability": {"diversify", "insure", "escalate"},
    "inventory_shortage":   {"stockpile", "diversify", "escalate"},
    "demand_shock":         {"hedge", "monitor", "diversify"},
}
 
 
def _category_match_score(retrieved_category: str, risk_category: str) -> float:
    """1.0 if exact match; 0.6 if keyword overlap; 0.0 otherwise."""
    rc = retrieved_category.lower()
    rk = risk_category.lower()
    if rc == rk:
        return 1.0
    # Partial keyword match
    rc_words = set(rc.split())
    rk_words = set(rk.split())
    overlap  = rc_words & rk_words
    if overlap:
        return 0.6
    return 0.0
 
 
def _ite_alignment_score(
    intervention_type: str,
    causal_variable:   str,
    ite_mean:          float,
) -> float:
    """
    Rewards intervention types that are compatible with the Layer 4
    causal variable that showed the strongest ITE effect.
    """
    compatible = VARIABLE_TO_INTERVENTIONS.get(causal_variable, set())
    type_match  = 1.0 if intervention_type in compatible else 0.3
 
    # ITE magnitude bonus: stronger negative ITE → solution more urgently needed
    # ite_mean is negative for improvement; scale to [0, 1]
    ite_bonus = min(1.0, max(0.0, abs(ite_mean) * 3.0))
 
    return round(0.6 * type_match + 0.4 * ite_bonus, 4)
 
 
def score_chunk(
    chunk:          RetrievedChunk,
    risk_category:  str,
    severity_label: str,
    causal_variable: str,
    ite_mean:       float,
) -> float:
    """Compute composite confidence score for one retrieved chunk."""
    cosine   = chunk.cosine_score
    cat_score = _category_match_score(chunk.chunk.category, risk_category)
    ite_score = _ite_alignment_score(chunk.chunk.intervention_type, causal_variable, ite_mean)
    sev_score = SEVERITY_SCORE.get(severity_label.upper(), 0.5)
 
    composite = (
        W_COSINE    * cosine
        + W_CATEGORY  * cat_score
        + W_ITE_ALIGN * ite_score
        + W_SEVERITY  * sev_score
    )
    return round(min(1.0, composite), 4)
 
 
def build_mitigation_solutions(
    retrieved_chunks:      List[RetrievedChunk],
    risk_id:               str,
    risk_title:            str,
    risk_category:         str,
    severity_label:        str,
    causal_variable:       str,
    ite_mean:              float,
    probability_of_improvement: float,
    top_k:                 int = 3,
) -> List[MitigationSolution]:
    """
    Score all retrieved chunks and return top-k as MitigationSolution objects,
    sorted descending by confidence_score.
    """
    scored: List[tuple] = []
    for chunk in retrieved_chunks:
        conf = score_chunk(
            chunk           = chunk,
            risk_category   = risk_category,
            severity_label  = severity_label,
            causal_variable = causal_variable,
            ite_mean        = ite_mean,
        )
        scored.append((conf, chunk))
 
    # Sort descending by confidence
    scored.sort(key=lambda x: x[0], reverse=True)
 
    solutions: List[MitigationSolution] = []
    for rank, (conf, retrieved) in enumerate(scored[:top_k], 1):
        c = retrieved.chunk
        action_steps = c.metadata.get("action_steps", [])
        if isinstance(action_steps, str):
            import json
            try:
                action_steps = json.loads(action_steps)
            except Exception:
                action_steps = []
 
        ite_align = _ite_alignment_score(c.intervention_type, causal_variable, ite_mean)
        sev_w     = SEVERITY_SCORE.get(severity_label.upper(), 0.5)
 
        solutions.append(MitigationSolution(
            solution_rank            = rank,
            risk_id                  = risk_id,
            risk_title               = risk_title,
            risk_category            = risk_category,
            intervention_type        = c.intervention_type,
            title                    = c.playbook_title,
            description              = c.text[:500],
            action_steps             = action_steps,
            cosine_similarity        = retrieved.cosine_score,
            confidence_score         = conf,
            ite_alignment_score      = ite_align,
            severity_weight          = sev_w,
            source_chunks            = [c.chunk_id],
            playbook_title           = c.playbook_title,
            causal_variable          = causal_variable,
            ite_mean                 = ite_mean,
            probability_of_improvement = probability_of_improvement,
        ))
 
    return solutions
 