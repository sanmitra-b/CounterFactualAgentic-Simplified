from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx


CAUSAL_NODES = [
    "ai_adoption_rate",
    "automation_exposure",
    "reskilling_capacity",
    "labor_market_demand",
    "policy_support",
    "wage_pressure",
    "transition_friction",
    "risk_severity",
]

CAUSAL_EDGES = [
    ("ai_adoption_rate", "automation_exposure"),
    ("ai_adoption_rate", "transition_friction"),
    ("policy_support", "reskilling_capacity"),
    ("reskilling_capacity", "transition_friction"),
    ("labor_market_demand", "transition_friction"),
    ("automation_exposure", "wage_pressure"),
    ("automation_exposure", "transition_friction"),
    ("transition_friction", "risk_severity"),
    ("wage_pressure", "risk_severity"),
    ("automation_exposure", "risk_severity"),
]


def build_job_risk_dag() -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(CAUSAL_NODES)
    dag.add_edges_from(CAUSAL_EDGES)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Configured causal graph is not a DAG.")
    return dag


def build_supply_chain_dag() -> nx.DiGraph:
    """Backward-compatible wrapper used by existing imports."""
    return build_job_risk_dag()


def _path_weight(path: List[str]) -> float:
    # Shorter path => stronger causal leverage.
    return 1.0 / max(1, len(path) - 1)


def get_paths_to_risk_severity(dag: nx.DiGraph) -> List[Dict[str, object]]:
    """Return all simple paths to risk_severity with path-based weights."""
    all_paths: List[Dict[str, object]] = []
    target = "risk_severity"
    for node in dag.nodes:
        if node == target:
            continue
        for path in nx.all_simple_paths(dag, source=node, target=target):
            all_paths.append(
                {
                    "source": node,
                    "target": target,
                    "path": path,
                    "path_length": len(path) - 1,
                    "weight": _path_weight(path),
                }
            )
    all_paths.sort(key=lambda x: x["weight"], reverse=True)
    return all_paths


def get_top_variables_by_causal_weight(dag: nx.DiGraph, top_n: int = 3) -> List[Tuple[str, float]]:
    """Aggregate path-weights per source variable and return top N."""
    paths = get_paths_to_risk_severity(dag)
    scores: Dict[str, float] = {}
    for item in paths:
        source = str(item["source"])
        weight = float(item["weight"])
        scores[source] = scores.get(source, 0.0) + weight

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
