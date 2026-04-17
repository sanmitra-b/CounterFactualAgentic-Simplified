from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from dowhy import gcm

from causal_graph import build_supply_chain_dag
from schemas_layer4 import CounterfactualResult, InterventionParams, ObservedRiskState


STATE_COLUMNS = [
    "port_congestion",
    "weather_severity",
    "geopolitical_tension",
    "shipping_delay",
    "supplier_reliability",
    "inventory_shortage",
    "demand_shock",
    "risk_severity",
]


def _build_observed_world_row(observed_state: ObservedRiskState) -> pd.DataFrame:
    row = {k: float(getattr(observed_state, k)) for k in STATE_COLUMNS}
    return pd.DataFrame([row])


def _add_small_noise(df: pd.DataFrame, noise_std: float, rng: np.random.Generator) -> pd.DataFrame:
    noisy = df.copy()
    for col in STATE_COLUMNS:
        noisy[col] = np.clip(noisy[col].to_numpy(dtype=float) + rng.normal(0.0, noise_std), 0.0, 1.0)
    return noisy


def _create_intervention_callable(value: float):
    # gcm.counterfactual_samples expects callables mapping previous samples to new values.
    return lambda x: np.ones_like(x) * float(value)


def _best_explanatory_path(intervened_variable: str) -> str:
    dag = build_supply_chain_dag()
    try:
        paths = list(nx_path_to_target(dag, intervened_variable, "risk_severity"))
    except Exception:
        paths = []

    if not paths:
        return f"No direct path found from {intervened_variable} to risk_severity."

    best = min(paths, key=lambda p: len(p))
    return " -> ".join(best)


def nx_path_to_target(dag, source: str, target: str) -> List[List[str]]:
    import networkx as nx

    return list(nx.all_simple_paths(dag, source=source, target=target))


def run_counterfactual_tool(
    observed_state: ObservedRiskState,
    intervention: InterventionParams,
    fitted_scm,
    n_samples: int = 200,
    noise_std: float = 0.02,
) -> CounterfactualResult:
    """
    Execute Pearl-style counterfactual using dowhy.gcm.counterfactual_samples:
    Abduction + Action + Prediction anchored to this specific observed instance.
    """
    observed_df = _build_observed_world_row(observed_state)
    observed_value = float(observed_state.risk_severity)

    interventions: Dict[str, object] = {
        intervention.variable: _create_intervention_callable(intervention.intervened_value)
    }

    # Bootstrap-style repeated counterfactual calls to estimate an uncertainty distribution.
    rng = np.random.default_rng(
        abs(hash((observed_state.risk_id, intervention.variable, intervention.iteration))) % (2**32)
    )
    cf_outcomes: List[float] = []
    ite_values: List[float] = []

    for _ in range(n_samples):
        noisy_observed = _add_small_noise(observed_df, noise_std=noise_std, rng=rng)
        cf_samples = gcm.counterfactual_samples(
            causal_model=fitted_scm,
            interventions=interventions,
            observed_data=noisy_observed,
        )
        cf_value = float(cf_samples["risk_severity"].iloc[0])
        cf_outcomes.append(cf_value)
        ite_values.append(cf_value - observed_value)

    cf_values = np.array(cf_outcomes, dtype=float)
    ite_array = np.array(ite_values, dtype=float)

    ite_mean = float(np.mean(ite_array))
    ci_low, ci_high = np.percentile(ite_array, [2.5, 97.5])
    p_improve = float(np.mean(ite_array < 0.0))

    threshold_cleared = ite_mean < -0.10 and p_improve > 0.70

    return CounterfactualResult(
        risk_id=observed_state.risk_id,
        risk_title=observed_state.title,
        category=observed_state.category,
        iteration=intervention.iteration,
        intervention=intervention,
        observed_risk_severity=observed_value,
        cf_risk_severity_mean=float(np.mean(cf_values)),
        ite_mean=ite_mean,
        ite_ci95_low=float(ci_low),
        ite_ci95_high=float(ci_high),
        probability_of_improvement=p_improve,
        explained_causal_path=_best_explanatory_path(intervention.variable),
        threshold_cleared=threshold_cleared,
    )
