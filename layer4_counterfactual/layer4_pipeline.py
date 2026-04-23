from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import networkx as nx

from agent_loop import run_agentic_counterfactual_loop
from causal_graph import build_job_risk_dag
from schemas_layer4 import ObservedRiskState
from scm_fitter import load_or_fit_scm

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "risk_report.json"
DEFAULT_RESULTS_PATH = ROOT_DIR / "data" / "counterfactual_results.json"


SEVERITY_MAP = {
    "LOW": 0.25,
    "MEDIUM": 0.50,
    "HIGH": 0.75,
    "CRITICAL": 0.95,
}


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _base_state_from_category(category: str) -> Dict[str, float]:
    c = (category or "").lower()
    base = {
        "ai_adoption_rate": 0.60,
        "automation_exposure": 0.55,
        "reskilling_capacity": 0.45,
        "labor_market_demand": 0.50,
        "policy_support": 0.40,
        "wage_pressure": 0.52,
        "transition_friction": 0.50,
    }

    if "displacement" in c or "automation" in c:
        base["automation_exposure"] += 0.25
        base["transition_friction"] += 0.15
    if "skills" in c or "reskilling" in c:
        base["transition_friction"] += 0.22
        base["reskilling_capacity"] -= 0.15
    if "wage" in c or "pay" in c:
        base["wage_pressure"] += 0.25
    if "hiring" in c or "labor demand" in c:
        base["labor_market_demand"] -= 0.20
        base["transition_friction"] += 0.12
    if "policy" in c or "regulation" in c or "public" in c:
        base["policy_support"] -= 0.18
        base["reskilling_capacity"] -= 0.10

    for key in list(base.keys()):
        base[key] = _clip01(base[key])

    return base


def risk_item_to_observed_state(item: Dict[str, object]) -> ObservedRiskState:
    category = str(item.get("category", "unknown"))
    confidence = float(item.get("confidence", 0.5))
    p_next = float(item.get("probability_next_30d", 0.5))
    sev_label = str(item.get("severity", "MEDIUM")).upper()
    sev_score = SEVERITY_MAP.get(sev_label, 0.5)

    base = _base_state_from_category(category)

    # Calibrate observed risk severity to Layer 3 score + confidence/probability.
    risk_severity = _clip01(0.55 * sev_score + 0.25 * p_next + 0.20 * confidence)

    # Keep downstream causal variables coherent with severity signal.
    base["transition_friction"] = _clip01(0.55 * base["transition_friction"] + 0.45 * risk_severity)
    base["wage_pressure"] = _clip01(0.60 * base["wage_pressure"] + 0.40 * risk_severity)

    return ObservedRiskState(
        risk_id=f"risk_{int(item.get('rank', 0))}",
        rank=int(item.get("rank", 0)),
        category=category,
        title=str(item.get("title", "Untitled Risk")),
        severity_label=sev_label,
        confidence=confidence,
        probability_next_30d=p_next,
        evidence=[str(x) for x in item.get("evidence", [])],
        affected_entities=[str(x) for x in item.get("affected_entities", [])],
        affected_geo=[str(x) for x in item.get("affected_geo", [])],
        causal_chain=str(item.get("causal_chain", "")),
        recommended_action=str(item.get("recommended_action", "")),
        ai_adoption_rate=base["ai_adoption_rate"],
        automation_exposure=base["automation_exposure"],
        reskilling_capacity=base["reskilling_capacity"],
        labor_market_demand=base["labor_market_demand"],
        policy_support=base["policy_support"],
        wage_pressure=base["wage_pressure"],
        transition_friction=base["transition_friction"],
        risk_severity=risk_severity,
    )


def run_layer4_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_RESULTS_PATH,
) -> List[Dict[str, object]]:
    with open(input_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    top_risks = report.get("top_risks", [])[:5]

    dag: nx.DiGraph = build_job_risk_dag()
    fitted_scm = load_or_fit_scm()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

    per_risk_results: List[Dict[str, object]] = []
    for item in top_risks:
        observed_state = risk_item_to_observed_state(item)
        memory, best = run_agentic_counterfactual_loop(observed_state, dag, fitted_scm)

        iterations = [r.model_dump(mode="json") for r in memory.all_results]
        per_risk_results.append(
            {
                "risk_id": observed_state.risk_id,
                "risk_title": observed_state.title,
                "category": observed_state.category,
                "all_iterations": iterations,
                "best_intervention": best.model_dump(mode="json"),
                "total_iterations": len(iterations),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(per_risk_results, f, indent=2)

    return per_risk_results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Layer 4 Counterfactual-Driven Agentic AI")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to risk_report.json")
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to output counterfactual_results.json",
    )
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    results = run_layer4_pipeline(input_path=args.input, output_path=args.output)
    print(f"Layer 4 complete. Saved {len(results)} risk intervention traces to {args.output}")
