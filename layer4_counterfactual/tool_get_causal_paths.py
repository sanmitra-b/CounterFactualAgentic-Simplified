from __future__ import annotations

from typing import Dict, List

import networkx as nx

from schemas_layer4 import ObservedRiskState


CATEGORY_TO_ROOT = {
    "commodity price volatility": "demand_shock",
    "trade disruptions": "geopolitical_tension",
    "economic instability": "demand_shock",
    "commodity market volatility": "demand_shock",
    "weather disruptions": "weather_severity",
}


def map_category_to_root_node(category: str) -> str:
    c = (category or "").strip().lower()

    if c in CATEGORY_TO_ROOT:
        return CATEGORY_TO_ROOT[c]

    # Fallback fuzzy mapping for near-matching labels.
    if "weather" in c:
        return "weather_severity"
    if "trade" in c or "geopolit" in c or "tariff" in c or "war" in c:
        return "geopolitical_tension"
    if "commodity" in c or "economic" in c or "demand" in c or "market" in c:
        return "demand_shock"
    if "port" in c or "shipping" in c:
        return "port_congestion"

    return "demand_shock"


def get_causal_paths_tool(observed_state: ObservedRiskState, dag: nx.DiGraph) -> Dict[str, object]:
    root = map_category_to_root_node(observed_state.category)
    target = "risk_severity"

    paths: List[Dict[str, object]] = []
    for p in nx.all_simple_paths(dag, source=root, target=target):
        weight = 1.0 / max(1, len(p) - 1)
        paths.append({"path": p, "path_length": len(p) - 1, "weight": weight})

    paths.sort(key=lambda x: float(x["weight"]), reverse=True)

    # Recommend variables from highest-weight paths that originate at the mapped root.
    recommended: List[str] = []
    for item in paths:
        path_nodes = [node for node in item["path"] if node != "risk_severity"]
        for node in path_nodes:
            if node not in recommended:
                recommended.append(node)
            if len(recommended) >= 3:
                break
        if len(recommended) >= 3:
            break

    if not recommended:
        recommended = [root]

    return {
        "risk_id": observed_state.risk_id,
        "category": observed_state.category,
        "mapped_root_node": root,
        "causal_paths": paths,
        "recommended_variables": recommended,
    }
