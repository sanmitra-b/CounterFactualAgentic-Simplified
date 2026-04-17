from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx


CAUSAL_NODES = [
    "port_congestion",
    "weather_severity",
    "geopolitical_tension",
    "shipping_delay",
    "supplier_reliability",
    "inventory_shortage",
    "demand_shock",
    "risk_severity",
]

CAUSAL_EDGES = [
    ("port_congestion", "shipping_delay"),
    ("weather_severity", "shipping_delay"),
    ("geopolitical_tension", "shipping_delay"),
    ("shipping_delay", "inventory_shortage"),
    ("supplier_reliability", "inventory_shortage"),
    ("demand_shock", "inventory_shortage"),
    ("inventory_shortage", "risk_severity"),
    ("shipping_delay", "risk_severity"),
    ("demand_shock", "risk_severity"),
    ("geopolitical_tension", "risk_severity"),
]


def build_supply_chain_dag() -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(CAUSAL_NODES)
    dag.add_edges_from(CAUSAL_EDGES)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Configured causal graph is not a DAG.")
    return dag


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
