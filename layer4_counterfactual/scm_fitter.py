from __future__ import annotations

from pathlib import Path
from typing import Optional

import cloudpickle
import numpy as np
import pandas as pd
from dowhy import gcm

from causal_graph import build_supply_chain_dag

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCM_PATH = ROOT_DIR / "data" / "fitted_scm.pkl"


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def generate_synthetic_data(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data that respects the hardcoded DAG structure."""
    rng = np.random.default_rng(seed)

    port_congestion = _clip01(rng.beta(2.2, 2.0, size=n_rows))
    weather_severity = _clip01(rng.beta(2.0, 2.8, size=n_rows))
    geopolitical_tension = _clip01(rng.beta(1.8, 3.0, size=n_rows))
    demand_shock = _clip01(rng.beta(2.1, 2.3, size=n_rows))

    shipping_delay = _clip01(
        0.45 * port_congestion
        + 0.25 * weather_severity
        + 0.30 * geopolitical_tension
        + rng.normal(0.0, 0.06, size=n_rows)
    )

    supplier_reliability = _clip01(
        1.0
        - 0.35 * geopolitical_tension
        - 0.15 * weather_severity
        + rng.normal(0.0, 0.08, size=n_rows)
    )

    inventory_shortage = _clip01(
        0.55 * shipping_delay
        + 0.30 * demand_shock
        + 0.25 * (1.0 - supplier_reliability)
        + rng.normal(0.0, 0.07, size=n_rows)
    )

    risk_severity = _clip01(
        0.45 * inventory_shortage
        + 0.30 * shipping_delay
        + 0.15 * demand_shock
        + 0.10 * geopolitical_tension
        + rng.normal(0.0, 0.05, size=n_rows)
    )

    return pd.DataFrame(
        {
            "port_congestion": port_congestion,
            "weather_severity": weather_severity,
            "geopolitical_tension": geopolitical_tension,
            "shipping_delay": shipping_delay,
            "supplier_reliability": supplier_reliability,
            "inventory_shortage": inventory_shortage,
            "demand_shock": demand_shock,
            "risk_severity": risk_severity,
        }
    )


def fit_and_save_scm(save_path: Path = DEFAULT_SCM_PATH, n_rows: int = 500):
    """Fit SCM once and persist for reuse by the agent loop."""
    dag = build_supply_chain_dag()
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
