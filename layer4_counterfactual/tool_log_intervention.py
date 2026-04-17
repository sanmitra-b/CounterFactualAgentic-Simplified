from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from schemas_layer4 import AgentMemory, CounterfactualResult

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = ROOT_DIR / "data" / "counterfactual_results.json"


def _read_existing(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def _append_result(path: Path, result: CounterfactualResult) -> None:
    existing = _read_existing(path)
    existing.append(result.model_dump(mode="json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def log_intervention_tool(
    result: CounterfactualResult,
    memory: AgentMemory,
    results_path: Path = DEFAULT_RESULTS_PATH,
) -> AgentMemory:
    already_present = any(
        r.iteration == result.iteration
        and r.intervention.variable == result.intervention.variable
        and abs(r.intervention.intervened_value - result.intervention.intervened_value) < 1e-12
        for r in memory.all_results
    )

    if not already_present:
        memory.all_results.append(result)
        memory.tried_interventions.append(result.intervention)

    if memory.best_result is None or result.ite_mean < memory.best_result.ite_mean:
        memory.best_result = result

    _append_result(results_path, result)
    return memory
