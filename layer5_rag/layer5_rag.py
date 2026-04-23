from __future__ import annotations
 
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
 
from confidence_scorer import build_mitigation_solutions
from schemas_layer5 import (
    MitigationSolution,
    RiskSolutionBundle,
    RiskSolutionMapping,
)
from vector_store import VectorStore
 
ROOT_DIR     = Path(__file__).resolve().parent
DATA_DIR     = ROOT_DIR / "data"
DEFAULT_IN   = DATA_DIR / "counterfactual_results.json"
DEFAULT_OUT  = DATA_DIR / "risk_solution_bundle.json"
 
SEVERITY_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
TOP_K_DEFAULT = 3
RETRIEVE_N    = 8    # fetch more than top_k to allow scoring to filter
 
 
# ─────────────────────────────────────────────────────────────────────────────
# QUERY BUILDER
# ─────────────────────────────────────────────────────────────────────────────
 
def build_retrieval_query(risk_data: Dict[str, Any]) -> str:
    """
    Construct a rich semantic query from the Layer 4 best_intervention record.
    Combines: risk title, category, causal variable, intervention variable,
    causal path, and severity to maximise retrieval precision.
    """
    best = risk_data.get("best_intervention", {})
    intervention = best.get("intervention", {})
 
    risk_title    = risk_data.get("risk_title", "")
    category      = risk_data.get("category", "")
    causal_var    = intervention.get("variable", "")
    causal_path   = best.get("explained_causal_path", "")
    severity      = _extract_severity(best)
    p_improve     = best.get("probability_of_improvement", 0.0)
 
    # Convert causal variable snake_case → human readable
    causal_readable = causal_var.replace("_", " ")
 
    query = (
        f"AI-driven job risk: {risk_title}. "
        f"Risk category: {category}. "
        f"Root causal variable: {causal_readable}. "
        f"Causal path: {causal_path}. "
        f"Severity: {severity}. "
        f"Probability of improvement with intervention: {p_improve:.0%}. "
        f"Mitigation strategy for {causal_readable} in {category.lower()} workforce transition risk."
    )
    return query
 
 
def _extract_severity(best_result: Dict[str, Any]) -> str:
    """Extract severity from the best counterfactual result record."""
    # Layer 4 stores it in the risk_title area or we default
    category = best_result.get("category", "").upper()
    if "CRITICAL" in category:
        return "CRITICAL"
    # Check observed_risk_severity as proxy
    obs_sev = float(best_result.get("observed_risk_severity", 0.5))
    if obs_sev >= 0.75:
        return "CRITICAL"
    if obs_sev >= 0.60:
        return "HIGH"
    if obs_sev >= 0.40:
        return "MEDIUM"
    return "LOW"
 
 
def _severity_from_score(score: float) -> str:
    if score >= 0.75: 
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.40:  
        return "MEDIUM"
    return "LOW"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PER-RISK PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
 
def process_risk(
    risk_data: Dict[str, Any],
    store:     VectorStore,
    top_k:     int = TOP_K_DEFAULT,
) -> RiskSolutionMapping:
    """
    Full RAG pipeline for one risk:
      1. Build query
      2. Retrieve from vector store
      3. Score and rank mitigations
      4. Return RiskSolutionMapping
    """
    risk_id    = risk_data.get("risk_id", "")
    risk_title = risk_data.get("risk_title", "")
    category   = risk_data.get("category", "")
    best       = risk_data.get("best_intervention", {})
    intervention = best.get("intervention", {})
 
    causal_var    = intervention.get("variable", "transition_friction")
    ite_mean      = float(best.get("ite_mean", 0.0))
    p_improve     = float(best.get("probability_of_improvement", 0.5))
    obs_severity  = float(best.get("observed_risk_severity", 0.5))
    severity_label = _severity_from_score(obs_severity)
 
    # ── Step 1: Build query ───────────────────────────────────────────────────
    query = build_retrieval_query(risk_data)
 
    # ── Step 2: Retrieve ──────────────────────────────────────────────────────
    retrieved = store.query(query, n_results=RETRIEVE_N)
 
    if not retrieved:
        # Broaden query if no results
        retrieved = store.query(f"{category} AI workforce risk mitigation", n_results=RETRIEVE_N)
 
    # ── Step 3: Score and rank ────────────────────────────────────────────────
    solutions = build_mitigation_solutions(
        retrieved_chunks          = retrieved,
        risk_id                   = risk_id,
        risk_title                = risk_title,
        risk_category             = category,
        severity_label            = severity_label,
        causal_variable           = causal_var,
        ite_mean                  = ite_mean,
        probability_of_improvement= p_improve,
        top_k                     = top_k,
    )
 
    return RiskSolutionMapping(
        risk_id                   = risk_id,
        risk_title                = risk_title,
        risk_category             = category,
        severity_label            = severity_label,
        causal_variable           = causal_var,
        ite_mean                  = ite_mean,
        probability_of_improvement= p_improve,
        top_mitigations           = solutions,
        retrieval_query           = query[:300],
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────
 
def _print_summary(bundle: RiskSolutionBundle) -> None:
    print("\n" + "=" * 65)
    print("  RISK-SOLUTION BUNDLE SUMMARY")
    print("=" * 65)
    print(f"  Mapped at        : {bundle.mapped_at}")
    print(f"  Domain           : {bundle.domain}")
    print(f"  Vector DB        : {bundle.vector_db_backend}")
    print(f"  Embedding model  : {bundle.embedding_model}")
    print(f"  Risks mapped     : {bundle.total_risks_mapped}")
    print(f"  Total solutions  : {bundle.total_solutions}")
 
    for mapping in bundle.mappings:
        icon = SEVERITY_ICON.get(mapping.severity_label, "⚪")
        print(f"\n  {icon} [{mapping.severity_label}]  {mapping.risk_title[:55]}")
        print(f"      Category      : {mapping.risk_category}")
        print(f"      Causal var    : {mapping.causal_variable}")
        print(f"      ITE mean      : {mapping.ite_mean:+.4f}  |  P(improve): {mapping.probability_of_improvement:.0%}")
        print(f"      Solutions     : {len(mapping.top_mitigations)}")
        for sol in mapping.top_mitigations:
            print(
                f"        #{sol.solution_rank}  [{sol.intervention_type:10s}]  "
                f"conf={sol.confidence_score:.3f}  cos={sol.cosine_similarity:.3f}  "
                f"{sol.title[:45]}"
            )
            if sol.action_steps:
                print(f"             → {sol.action_steps[0][:70]}")
 
    if bundle.ranked_mitigations:
        print("\n  ── PORTFOLIO TOP-5 MITIGATIONS (by confidence) ─────────────")
        for sol in bundle.ranked_mitigations[:5]:
            print(
                f"    [{sol.confidence_score:.3f}]  {sol.title[:50]}  "
                f"({sol.risk_category[:25]})"
            )
 
    print("\n" + "=" * 65)
    print("[→] Ready for Layer 6 (Human-readable cause-effect output)\n")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
 
def run_layer5(
    input_path:  Path = DEFAULT_IN,
    output_path: Path = DEFAULT_OUT,
    rebuild:     bool = False,
    top_k:       int  = TOP_K_DEFAULT,
) -> RiskSolutionBundle:
 
    print("\n" + "=" * 65)
    print("  LAYER 5 — RAG RISK-SOLUTION MAPPING")
    print("=" * 65)
 
    # ── 1. Load Layer 4 output ────────────────────────────────────────────────
    print(f"\n[1/4] Loading counterfactual results from '{input_path}' …")
    with open(input_path, "r", encoding="utf-8") as f:
        cf_results: List[Dict[str, Any]] = json.load(f)
 
    print(f"    Risks loaded    : {len(cf_results)}")
    if not cf_results:
        print("  [!] No results found. Run Layer 4 first.")
        cf_results = _demo_cf_results()
        print(f"    Using demo data : {len(cf_results)} risks")
 
    # ── 2. Build / load vector store ─────────────────────────────────────────
    print("\n[2/4] Initialising vector store …")
    if rebuild:
        print("    Rebuilding vector store from scratch …")
        store = VectorStore.build()
    else:
        store = VectorStore.load_or_build()
    print(f"    Backend         : {store.backend_name}")
 
    # ── 3. Process each risk ──────────────────────────────────────────────────
    print(f"\n[3/4] Retrieving and scoring mitigations for {len(cf_results)} risks …")
    mappings: List[RiskSolutionMapping] = []
 
    for risk_data in cf_results:
        rid   = risk_data.get("risk_id", "?")
        title = risk_data.get("risk_title", "?")[:55]
        print(f"\n    [{rid}] {title}")
 
        mapping = process_risk(risk_data, store, top_k=top_k)
        mappings.append(mapping)
 
        for sol in mapping.top_mitigations:
            print(
                f"      #{sol.solution_rank}  conf={sol.confidence_score:.3f}  "
                f"[{sol.intervention_type}]  {sol.title[:45]}"
            )
 
    # ── 4. Assemble bundle ────────────────────────────────────────────────────
    print(f"\n[4/4] Assembling RiskSolutionBundle → '{output_path}' …")
    all_solutions: List[MitigationSolution] = []
    for m in mappings:
        all_solutions.extend(m.top_mitigations)
 
    # Portfolio ranked list: sort all solutions by confidence descending
    ranked = sorted(all_solutions, key=lambda s: s.confidence_score, reverse=True)
 
    bundle = RiskSolutionBundle(
        domain                = "ai_job_risk",
        mapped_at             = datetime.utcnow().isoformat(),
        layer4_source         = str(input_path),
        total_risks_mapped    = len(mappings),
        total_solutions       = len(all_solutions),
        mappings              = mappings,
        ranked_mitigations    = ranked,
        vector_db_backend     = store.backend_name,
        embedding_model       = "all-MiniLM-L6-v2",
        chunk_size_tokens     = 512,
    )
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle.model_dump(), f, indent=2, default=str)
 
    print(f"    ✓ Saved to '{output_path}'")
    _print_summary(bundle)
    return bundle
 
  
# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA (when Layer 4 output is empty)
# ─────────────────────────────────────────────────────────────────────────────
 
def _demo_cf_results() -> List[Dict[str, Any]]:
    return [
        {
            "risk_id": "risk_1", "risk_title": "Major port backlog at Singapore and Rotterdam",
            "category": "Port Congestion",
            "best_intervention": {
                "risk_id": "risk_1", "risk_title": "Major port backlog at Singapore and Rotterdam",
                "category": "Port Congestion", "iteration": 2,
                "intervention": {"variable": "port_congestion", "intervened_value": 0.35,
                                  "rationale": "Reduce port congestion via rerouting", "iteration": 2},
                "observed_risk_severity": 0.72, "cf_risk_severity_mean": 0.55,
                "ite_mean": -0.17, "ite_ci95_low": -0.22, "ite_ci95_high": -0.12,
                "probability_of_improvement": 0.88, "threshold_cleared": True,
                "explained_causal_path": "port_congestion -> shipping_delay -> inventory_shortage -> risk_severity",
            },
        },
        {
            "risk_id": "risk_2", "risk_title": "Oil price spike disrupting freight costs",
            "category": "Commodity Price Shock",
            "best_intervention": {
                "risk_id": "risk_2", "risk_title": "Oil price spike disrupting freight costs",
                "category": "Commodity Price Shock", "iteration": 3,
                "intervention": {"variable": "demand_shock", "intervened_value": 0.28,
                                  "rationale": "Hedge commodity exposure", "iteration": 3},
                "observed_risk_severity": 0.65, "cf_risk_severity_mean": 0.48,
                "ite_mean": -0.17, "ite_ci95_low": -0.21, "ite_ci95_high": -0.13,
                "probability_of_improvement": 0.82, "threshold_cleared": True,
                "explained_causal_path": "demand_shock -> inventory_shortage -> risk_severity",
            },
        },
        {
            "risk_id": "risk_3", "risk_title": "Typhoon season threatening East Asian shipping lanes",
            "category": "Weather / Natural Disaster",
            "best_intervention": {
                "risk_id": "risk_3", "risk_title": "Typhoon season threatening East Asian shipping lanes",
                "category": "Weather / Natural Disaster", "iteration": 2,
                "intervention": {"variable": "weather_severity", "intervened_value": 0.22,
                                  "rationale": "Pre-position stock before typhoon season", "iteration": 2},
                "observed_risk_severity": 0.55, "cf_risk_severity_mean": 0.41,
                "ite_mean": -0.14, "ite_ci95_low": -0.19, "ite_ci95_high": -0.09,
                "probability_of_improvement": 0.76, "threshold_cleared": True,
                "explained_causal_path": "weather_severity -> shipping_delay -> risk_severity",
            },
        },
        {
            "risk_id": "risk_4", "risk_title": "Currency fluctuations increasing import costs",
            "category": "Financial Market Volatility",
            "best_intervention": {
                "risk_id": "risk_4", "risk_title": "Currency fluctuations increasing import costs",
                "category": "Financial Market Volatility", "iteration": 2,
                "intervention": {"variable": "demand_shock", "intervened_value": 0.30,
                                  "rationale": "FX hedging programme", "iteration": 2},
                "observed_risk_severity": 0.58, "cf_risk_severity_mean": 0.44,
                "ite_mean": -0.14, "ite_ci95_low": -0.18, "ite_ci95_high": -0.10,
                "probability_of_improvement": 0.74, "threshold_cleared": True,
                "explained_causal_path": "demand_shock -> risk_severity",
            },
        },
        {
            "risk_id": "risk_5", "risk_title": "Trade policy uncertainty affecting cross-border flows",
            "category": "Geopolitical Disruption",
            "best_intervention": {
                "risk_id": "risk_5", "risk_title": "Trade policy uncertainty affecting cross-border flows",
                "category": "Geopolitical Disruption", "iteration": 3,
                "intervention": {"variable": "geopolitical_tension", "intervened_value": 0.20,
                                  "rationale": "Diversify supplier base", "iteration": 3},
                "observed_risk_severity": 0.52, "cf_risk_severity_mean": 0.38,
                "ite_mean": -0.14, "ite_ci95_low": -0.18, "ite_ci95_high": -0.10,
                "probability_of_improvement": 0.71, "threshold_cleared": True,
                "explained_causal_path": "geopolitical_tension -> shipping_delay -> risk_severity",
            },
        },
    ]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 5 — RAG Risk-Solution Mapping")
    parser.add_argument("--input",   type=Path, default=DEFAULT_IN,
                        help="Layer 4 counterfactual_results.json")
    parser.add_argument("--output",  type=Path, default=DEFAULT_OUT,
                        help="Output risk_solution_bundle.json")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild vector store from scratch")
    parser.add_argument("--top-k",  type=int, default=TOP_K_DEFAULT,
                        help=f"Solutions per risk (default: {TOP_K_DEFAULT})")
    args = parser.parse_args()
 
    run_layer5(
        input_path  = args.input,
        output_path = args.output,
        rebuild     = args.rebuild,
        top_k       = args.top_k,
    )