from __future__ import annotations

import hashlib
import math
from typing import Any

from contracts import (
    AnalystWorkOrder,
    InterventionType,
    Severity,
    Feasibility,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HEURISTIC TABLES
# These tables encode domain-agnostic simulation heuristics.
# Extend or override them for domain-specific calibration.
# ═══════════════════════════════════════════════════════════════════════════════

# Intervention type → expected probability reduction multiplier
# Represents the *fraction* of baseline probability that can be removed.
# E.g. 0.30 means a 30% relative reduction in P(30d).
_INTERVENTION_REDUCTION: dict[str, float] = {
    InterventionType.POLICY:      0.25,
    InterventionType.SUPPLY:      0.35,
    InterventionType.FINANCIAL:   0.20,
    InterventionType.OPERATIONAL: 0.30,
    InterventionType.REGULATORY:  0.22,
}

# Severity → feasibility mapping heuristic
# Higher severity risks often have more organisational will but harder logistics
_SEVERITY_FEASIBILITY: dict[str, Feasibility] = {
    Severity.CRITICAL: Feasibility.MEDIUM,
    Severity.HIGH:     Feasibility.HIGH,
    Severity.MEDIUM:   Feasibility.HIGH,
    Severity.LOW:      Feasibility.HIGH,
}

# Intervention type → time-to-impact range in days (min, max)
_TIME_TO_IMPACT: dict[str, tuple[int, int]] = {
    InterventionType.POLICY:      (60, 180),
    InterventionType.SUPPLY:      (14, 60),
    InterventionType.FINANCIAL:   (7,  30),
    InterventionType.OPERATIONAL: (7,  45),
    InterventionType.REGULATORY:  (90, 365),
}

# Intervention type → rough cost order of magnitude
_COST_RANGE: dict[str, str] = {
    InterventionType.POLICY:      "$50K – $500K (staffing + legal)",
    InterventionType.SUPPLY:      "$500K – $10M (inventory / sourcing)",
    InterventionType.FINANCIAL:   "$100K – $5M (hedging / reserves)",
    InterventionType.OPERATIONAL: "$50K – $2M (process + tooling)",
    InterventionType.REGULATORY:  "$200K – $1M (legal + compliance)",
}

# Severity downgrade table after successful intervention
_SEVERITY_DOWNGRADE: dict[Severity, Severity] = {
    Severity.CRITICAL: Severity.HIGH,
    Severity.HIGH:     Severity.MEDIUM,
    Severity.MEDIUM:   Severity.LOW,
    Severity.LOW:      Severity.LOW,
}

# Generic second-order effects library keyed by intervention type
_SECOND_ORDER_EFFECTS: dict[str, list[str]] = {
    InterventionType.POLICY: [
        "Policy implementation lag may expose a window of increased vulnerability.",
        "New policy may create compliance overhead that slows operational agility.",
        "Cross-jurisdictional policy conflicts may arise in multi-region operations.",
    ],
    InterventionType.SUPPLY: [
        "Dual-sourcing increases procurement complexity and coordination costs.",
        "Safety stock build-up ties up working capital, reducing financial flexibility.",
        "New suppliers require quality audits; onboarding delays may temporarily worsen supply.",
    ],
    InterventionType.FINANCIAL: [
        "Hedging instruments carry counterparty risk if the hedge provider defaults.",
        "Over-hedging can lock in losses if market conditions reverse.",
        "Liquidity reserves reduce capital available for growth investment.",
    ],
    InterventionType.OPERATIONAL: [
        "Process redesign may cause short-term productivity dip during transition.",
        "New operational controls may introduce bottlenecks in high-throughput workflows.",
        "Staff resistance to process change can delay realisation of benefits.",
    ],
    InterventionType.REGULATORY: [
        "Regulatory engagement may expose undisclosed compliance gaps to authorities.",
        "Prescriptive regulations may reduce operational flexibility.",
        "Competitor regulatory strategy may diverge, creating an uneven playing field.",
    ],
}

# Keyword → intervention type suggestion mapping (used for hypothesis generation)
_KEYWORD_INTERVENTION_MAP: list[tuple[list[str], InterventionType]] = [
    (["supply", "shortage", "inventory", "logistics", "port", "shipping"],  InterventionType.SUPPLY),
    (["price", "currency", "inflation", "interest", "market", "financial"], InterventionType.FINANCIAL),
    (["regulation", "FDA", "compliance", "approval", "law", "policy"],      InterventionType.REGULATORY),
    (["workforce", "staffing", "hiring", "talent", "labour"],               InterventionType.OPERATIONAL),
    (["policy", "government", "tariff", "sanction", "geopolitical"],        InterventionType.POLICY),
]


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVENTION HYPOTHESIS GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_intervention_types(order: AnalystWorkOrder) -> list[InterventionType]:
    """
    Infer the most relevant intervention types for a risk by scanning the
    causal chain, recommended action, and category for domain keywords.
    Always returns exactly `order.n_scenarios` types (with fallback cycling).
    """
    text = (
        f"{order.risk_title} {order.risk_category} "
        f"{order.causal_chain} {order.recommended_action}"
    ).lower()

    scores: dict[InterventionType, int] = {t: 0 for t in InterventionType}
    for keywords, intervention_type in _KEYWORD_INTERVENTION_MAP:
        for kw in keywords:
            if kw in text:
                scores[intervention_type] += 1

    # Sort by score descending; fall back to round-robin if all zero
    ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
    # Ensure diversity: don't repeat the same type if we need multiple scenarios
    selected: list[InterventionType] = []
    for t in ranked:
        if t not in selected:
            selected.append(t)
        if len(selected) == order.n_scenarios:
            break

    # Fill remaining slots by cycling through all types
    all_types = list(InterventionType)
    i = 0
    while len(selected) < order.n_scenarios:
        if all_types[i % len(all_types)] not in selected:
            selected.append(all_types[i % len(all_types)])
        i += 1

    return selected[:order.n_scenarios]


def _generate_intervention_text(
    order: AnalystWorkOrder,
    intervention_type: InterventionType,
    scenario_index: int,
) -> str:
    """
    Produce a concise, specific intervention description by combining
    the intervention type template with key terms from the risk item.
    """
    domain  = order.domain
    title   = order.risk_title
    action  = order.recommended_action[:80] if order.recommended_action else title

    templates: dict[InterventionType, str] = {
        InterventionType.POLICY: (
            f"Implement a formal {domain} risk-governance policy that addresses '{title}': "
            f"mandate cross-functional oversight, set exposure limits, and establish "
            f"escalation protocols aligned with '{action}'."
        ),
        InterventionType.SUPPLY: (
            f"Activate a dual/multi-sourcing programme for critical inputs linked to '{title}': "
            f"qualify at least two geographically diverse suppliers within 60 days "
            f"and pre-negotiate surge-capacity agreements."
        ),
        InterventionType.FINANCIAL: (
            f"Deploy targeted financial hedging instruments for '{title}'-driven exposure: "
            f"establish rolling forward contracts or reserves covering the next "
            f"three months of identified cash-flow risk."
        ),
        InterventionType.OPERATIONAL: (
            f"Redesign the operational workflow most exposed to '{title}': "
            f"introduce automated early-warning triggers, standardised response runbooks, "
            f"and regular scenario drills to operationalise '{action}'."
        ),
        InterventionType.REGULATORY: (
            f"Launch a proactive regulatory engagement programme for '{title}': "
            f"schedule quarterly pre-submission meetings with relevant authorities "
            f"and assign a dedicated compliance team to monitor guidance changes."
        ),
    }
    return templates[intervention_type]


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY & SEVERITY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_probability(
    baseline: float,
    intervention_type: InterventionType,
    feasibility: Feasibility,
    confidence: float,
) -> float:
    """
    Compute counterfactual probability using:
      reduction = base_reduction × feasibility_multiplier × sqrt(confidence)

    Feasibility multiplier:
      HIGH   → 1.00 (full reduction realised)
      MEDIUM → 0.65 (partial realisation)
      LOW    → 0.30 (marginal impact)
    """
    base_reduction = _INTERVENTION_REDUCTION[intervention_type]

    feasibility_multiplier = {
        Feasibility.HIGH:   1.00,
        Feasibility.MEDIUM: 0.65,
        Feasibility.LOW:    0.30,
    }[feasibility]

    # Scale by analyst confidence (sqrt dampens extreme values)
    effective_reduction = base_reduction * feasibility_multiplier * math.sqrt(confidence)

    new_p = baseline * (1.0 - effective_reduction)
    return round(max(0.01, min(0.99, new_p)), 4)


def _simulate_severity(baseline: Severity, feasibility: Feasibility) -> Severity:
    """Downgrade severity only if the intervention is at least MEDIUM feasibility."""
    if feasibility == Feasibility.LOW:
        return baseline   # low feasibility → severity unchanged
    return _SEVERITY_DOWNGRADE[baseline]


def _assess_feasibility(
    order: AnalystWorkOrder,
    intervention_type: InterventionType,
) -> tuple[Feasibility, str]:
    """
    Heuristically assess feasibility using:
      - Severity of the risk (higher severity → more organisational impetus)
      - Whether the recommended_action already implies readiness
      - Intervention type cost / lead-time profile
    """
    severity_feas = _SEVERITY_FEASIBILITY[order.severity]

    # Boost feasibility if the recommended_action already references the intervention type
    action_lower = order.recommended_action.lower()
    type_keywords = {
        InterventionType.POLICY:      ["policy", "governance", "mandate"],
        InterventionType.SUPPLY:      ["supplier", "sourcing", "inventory", "stock"],
        InterventionType.FINANCIAL:   ["hedge", "reserve", "financial", "capital"],
        InterventionType.OPERATIONAL: ["process", "workflow", "training", "runbook"],
        InterventionType.REGULATORY:  ["regulatory", "compliance", "regulator", "FDA"],
    }
    already_signalled = any(kw in action_lower for kw in type_keywords.get(intervention_type, []))
    if already_signalled and severity_feas != Feasibility.LOW:
        final_feas = Feasibility.HIGH
    else:
        final_feas = severity_feas

    time_min, time_max = _TIME_TO_IMPACT[intervention_type]
    rationale = (
        f"{intervention_type.value} interventions typically cost {_COST_RANGE[intervention_type]} "
        f"and take {time_min}–{time_max} days to show impact. "
        f"Given {order.severity.value} severity"
        + (" and existing alignment in the recommended action," if already_signalled else ",")
        + f" feasibility is assessed as {final_feas.value}."
    )
    return final_feas, rationale


def _time_to_impact(intervention_type: InterventionType, feasibility: Feasibility) -> int:
    """
    Return a representative time-to-impact in days.
    LOW feasibility → upper bound of range (slower realisation).
    HIGH feasibility → lower bound (faster).
    """
    t_min, t_max = _TIME_TO_IMPACT[intervention_type]
    if feasibility == Feasibility.HIGH:
        return t_min
    elif feasibility == Feasibility.MEDIUM:
        return (t_min + t_max) // 2
    else:
        return t_max


def _build_assumptions(order: AnalystWorkOrder, intervention_type: InterventionType) -> list[str]:
    return [
        f"The organisation has sufficient budget and executive sponsorship to execute a "
        f"{intervention_type.value.lower()} intervention within the assessed lead time.",
        f"External conditions driving '{order.risk_title}' do not materially worsen "
        f"during the intervention implementation window.",
        f"Key stakeholders in {', '.join(order.affected_entities[:2]) or order.domain} "
        f"are willing to co-operate with the proposed changes.",
    ]


def _get_second_order_effects(intervention_type: InterventionType) -> list[str]:
    return _SECOND_ORDER_EFFECTS.get(intervention_type, [
        "Implementation complexity may consume management bandwidth.",
        "Benefits may take longer to materialise than the 30-day horizon.",
    ])[:3]


def _make_scenario_id(risk_rank: int, index: int, intervention_type: InterventionType) -> str:
    return f"cf_{risk_rank}_{chr(96 + index)}_{intervention_type.value[:3].lower()}"


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(order: AnalystWorkOrder) -> list[dict[str, Any]]:
    """
    Entry point called exclusively by the Analyst Agent.

    Accepts a validated AnalystWorkOrder, runs all heuristic simulations,
    and returns a list of raw scenario dicts (not yet Pydantic-validated —
    that validation step is the Analyst Agent's responsibility).
    """
    intervention_types = _infer_intervention_types(order)
    raw_scenarios: list[dict[str, Any]] = []

    for idx, i_type in enumerate(intervention_types, start=1):
        feasibility, feas_rationale = _assess_feasibility(order, i_type)

        cf_prob = _simulate_probability(
            baseline         = order.probability_next_30d,
            intervention_type = i_type,
            feasibility      = feasibility,
            confidence       = order.confidence,
        )
        cf_severity = _simulate_severity(order.severity, feasibility)
        delta       = round(cf_prob - order.probability_next_30d, 4)
        tti         = _time_to_impact(i_type, feasibility)

        raw_scenarios.append({
            "scenario_id":                _make_scenario_id(order.risk_rank, idx, i_type),
            "risk_rank":                  order.risk_rank,
            "risk_title":                 order.risk_title,
            "intervention":               _generate_intervention_text(order, i_type, idx),
            "intervention_type":          i_type.value,
            "baseline_probability":       order.probability_next_30d,
            "counterfactual_probability": cf_prob,
            "delta_probability":          delta,
            "baseline_severity":          order.severity.value,
            "counterfactual_severity":    cf_severity.value,
            "feasibility":                feasibility.value,
            "feasibility_rationale":      feas_rationale,
            "assumptions":                _build_assumptions(order, i_type),
            "second_order_effects":       _get_second_order_effects(i_type),
            "estimated_cost_usd":         _COST_RANGE[i_type],
            "time_to_impact_days":        tti,
            "confidence":                 round(order.confidence * 0.9, 4),  # slight discount for simulation uncertainty
        })

    return raw_scenarios
