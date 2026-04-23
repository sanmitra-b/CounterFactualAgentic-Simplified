from __future__ import annotations

from typing import Dict, List

import networkx as nx

from schemas_layer4 import ObservedRiskState


CATEGORY_TO_ROOT = {
    "job displacement risk": "automation_exposure",
    "skills obsolescence risk": "transition_friction",
    "wage suppression risk": "wage_pressure",
    "hiring slowdown risk": "labor_market_demand",
    "public sector workforce transition risk": "policy_support",
    "inequality amplification risk": "transition_friction",
}


def map_category_to_root_node(category: str) -> str:
    c = (category or "").strip().lower()

    if c in CATEGORY_TO_ROOT:
        return CATEGORY_TO_ROOT[c]

    # Fallback fuzzy mapping for near-matching labels.
    if "displacement" in c or "automation" in c:
        return "automation_exposure"
    if "skill" in c or "reskilling" in c or "transition" in c:
        return "transition_friction"
    if "wage" in c or "pay" in c:
        return "wage_pressure"
    if "hiring" in c or "labor demand" in c or "job opening" in c:
        return "labor_market_demand"
    if "policy" in c or "regulation" in c or "public" in c:
        return "policy_support"

    return "automation_exposure"


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
