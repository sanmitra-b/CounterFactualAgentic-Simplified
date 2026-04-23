from __future__ import annotations

from pathlib import Path
from typing import Optional

import cloudpickle
import numpy as np
import pandas as pd
from dowhy import gcm

from causal_graph import build_job_risk_dag

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCM_PATH = ROOT_DIR / "data" / "fitted_scm.pkl"


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def generate_synthetic_data(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data consistent with the AI job-risk DAG."""
    rng = np.random.default_rng(seed)

    ai_adoption_rate = _clip01(rng.beta(2.6, 1.8, size=n_rows))
    policy_support = _clip01(rng.beta(2.0, 2.4, size=n_rows))
    labor_market_demand = _clip01(rng.beta(2.2, 2.0, size=n_rows))

    automation_exposure = _clip01(
        0.65 * ai_adoption_rate
        + 0.10 * (1.0 - policy_support)
        + rng.normal(0.0, 0.05, size=n_rows)
    )

    reskilling_capacity = _clip01(
        0.55 * policy_support
        + 0.30 * labor_market_demand
        + rng.normal(0.0, 0.06, size=n_rows)
    )

    transition_friction = _clip01(
        0.45 * automation_exposure
        + 0.30 * (1.0 - reskilling_capacity)
        + 0.25 * (1.0 - labor_market_demand)
        + 0.15 * ai_adoption_rate
        + rng.normal(0.0, 0.06, size=n_rows)
    )

    wage_pressure = _clip01(
        0.50 * automation_exposure
        + 0.30 * transition_friction
        + 0.15 * (1.0 - labor_market_demand)
        + rng.normal(0.0, 0.06, size=n_rows)
    )

    risk_severity = _clip01(
        0.45 * transition_friction
        + 0.30 * wage_pressure
        + 0.20 * automation_exposure
        + 0.05 * ai_adoption_rate
        + rng.normal(0.0, 0.05, size=n_rows)
    )

    return pd.DataFrame(
        {
            "ai_adoption_rate": ai_adoption_rate,
            "automation_exposure": automation_exposure,
            "reskilling_capacity": reskilling_capacity,
            "labor_market_demand": labor_market_demand,
            "policy_support": policy_support,
            "wage_pressure": wage_pressure,
            "transition_friction": transition_friction,
            "risk_severity": risk_severity,
        }
    )


def fit_and_save_scm(save_path: Path = DEFAULT_SCM_PATH, n_rows: int = 500):
    """Fit SCM once and persist for reuse by the agent loop."""
    dag = build_job_risk_dag()
    data = generate_synthetic_data(n_rows=n_rows)

    # Invertible SCM is required for abduction from observed_data in counterfactual calls.
    scm = gcm.InvertibleStructuralCausalModel(dag)
    gcm.auto.assign_causal_mechanisms(scm, data)
    gcm.fit(scm, data)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        cloudpickle.dump(scm, f)

    return scm


def load_fitted_scm(save_path: Path = DEFAULT_SCM_PATH):
    with open(save_path, "rb") as f:
        return cloudpickle.load(f)


def load_or_fit_scm(save_path: Optional[Path] = None):
    target = save_path or DEFAULT_SCM_PATH
    if target.exists():
        scm = load_fitted_scm(target)
        if isinstance(scm, gcm.InvertibleStructuralCausalModel):
            return scm
        return fit_and_save_scm(save_path=target)
    return fit_and_save_scm(save_path=target)
