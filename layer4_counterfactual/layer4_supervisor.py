from __future__ import annotations

import json
import logging
import sys
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Allows running as `python layer4_counterfactual/supervisor.py`
# or being imported by layer0's run_pipeline().
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# ──────────────────────────────────────────────────────────────────────────────

from contracts import (
    AnalystWorkOrder,
    AnalystWorkResult,
    CounterfactualBundle,
    CounterfactualScenario,
    Severity,
)
from analyst_agent import AnalystAgent

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH  = _ROOT / "data" / "risk_report.json"
DEFAULT_OUTPUT_PATH = _ROOT / "data" / "counterfactual_bundle.json"
SCENARIOS_PER_RISK  = 2


# ═══════════════════════════════════════════════════════════════════════════════
# WORK ORDER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def _build_work_order(risk: dict, domain: str, n_scenarios: int) -> AnalystWorkOrder:
    """
    Translate one raw risk dict (from RiskReport) into a typed AnalystWorkOrder.
    ValidationError here means the Layer 3 output is malformed — fail fast.
    """
    return AnalystWorkOrder(
        request_id          = f"l4_risk_{risk.get('rank', 0)}",
        domain              = domain,
        risk_rank           = int(risk["rank"]),
        risk_title          = risk["title"],
        risk_category       = risk.get("category", "Unknown"),
        severity            = Severity(risk.get("severity", "MEDIUM").upper()),
        confidence          = float(risk.get("confidence", 0.5)),
        probability_next_30d = float(risk.get("probability_next_30d", 0.5)),
        causal_chain        = risk.get("causal_chain", ""),
        recommended_action  = risk.get("recommended_action", ""),
        affected_entities   = risk.get("affected_entities", []),
        affected_geo        = risk.get("affected_geo", []),
        evidence            = risk.get("evidence", []),
        n_scenarios         = n_scenarios,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _calculate_mitigation_efficiency(scenarios: list[CounterfactualScenario]) -> float:
    """
    Calculate the Macro Counterfactual Mitigation Efficiency (η_mitigation).
    
    Formula:
        η_mitigation = (1/M) * Σ(|Δ_m| / R_base,m)
    
    Where:
        M = number of active scenarios
        Δ_m = Individual Treatment Effect = baseline_probability - counterfactual_probability
              (absolute value to capture risk reduction as positive)
        R_base,m = baseline probability for scenario m
    
    Interprets delta_probability (counterfactual - baseline, typically negative)
    as a risk reduction by taking its absolute value.
    
    Returns:
        Average percentage reduction in risk scores across all scenarios.
        Range: [0.0, 1.0] where 1.0 = 100% average mitigation
    """
    if not scenarios:
        return 0.0
    
    mitigation_ratios: list[float] = []
    for scenario in scenarios:
        # Avoid division by zero; skip scenarios with near-zero baseline
        if scenario.baseline_probability > 0.001:
            # Use absolute value of delta since it's typically negative (reduction)
            # This represents the magnitude of risk reduction
            ratio = abs(scenario.delta_probability) / scenario.baseline_probability
            # Clamp to [0, 1] to handle edge cases
            mitigation_ratios.append(max(0.0, min(1.0, ratio)))
    
    if not mitigation_ratios:
        return 0.0
    
    # Return the average mitigation ratio across all scenarios
    efficiency = sum(mitigation_ratios) / len(mitigation_ratios)
    return round(efficiency, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT VALIDATOR  (the Guardrail)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_result(result: AnalystWorkResult) -> list[CounterfactualScenario]:
    """
    The Supervisor's guardrail: re-validate every scenario in the result
    before allowing it into the bundle.

    Why re-validate here if the Agent already validated?
    ─ Defence in depth: a compromised or buggy agent could return objects
      that were valid when created but have since been mutated.
    ─ Ensures the Supervisor — not the worker — is the last line of defence.

    Returns the valid scenarios; logs and discards any that fail.
    """
    accepted: list[CounterfactualScenario] = []

    for sc in result.scenarios:
        try:
            # Re-instantiate from dict to trigger all validators fresh
            revalidated = CounterfactualScenario(**sc.model_dump())
            accepted.append(revalidated)
        except ValidationError as ve:
            logger.warning(
                "Supervisor REJECTED scenario '%s' from WorkResult '%s': %s",
                getattr(sc, "scenario_id", "?"),
                result.request_id,
                ve,
            )

    return accepted


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

SICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
FICON = {"HIGH": "✅", "MEDIUM": "⚠️ ", "LOW": "❌"}

def _print_summary(bundle: CounterfactualBundle) -> None:
    print("\n" + "=" * 65)
    print("  LAYER 4 — COUNTERFACTUAL SIMULATION COMPLETE")
    print("=" * 65)
    print(f"  Domain        : {bundle.domain}")
    print(f"  Simulated at  : {bundle.simulated_at}")
    print(f"  Risks covered : {bundle.total_risks}")
    print(f"  Scenarios     : {bundle.total_scenarios}")
    print(f"  Avg Δ P(30d)  : {bundle.avg_delta:+.2%}")
    print(f"  Feasibility   : {bundle.feasibility_dist}")
    print(f"  η_mitigation  : {bundle.mitigation_efficiency:.2%}  (Macro Counterfactual Mitigation Efficiency)")
    print()

    current_risk = None
    for sc in bundle.scenarios:
        if sc.risk_rank != current_risk:
            current_risk = sc.risk_rank
            sev_icon = SICON.get(sc.baseline_severity, "⚪")
            print(f"  {sev_icon} Risk #{sc.risk_rank}: {sc.risk_title[:55]}")

        f_icon = FICON.get(sc.feasibility, "?")
        cf_icon = SICON.get(sc.counterfactual_severity, "⚪")
        print(f"    [{sc.scenario_id}]")
        print(f"      Type       : {sc.intervention_type}")
        print(f"      Intervention: {sc.intervention[:70]}")
        print(f"      P(30d) Δ   : {sc.delta_probability:+.2%}  "
              f"({sc.baseline_probability:.0%} → {sc.counterfactual_probability:.0%})")
        print(f"      Severity   : {sc.baseline_severity} → {cf_icon} {sc.counterfactual_severity}")
        print(f"      Feasibility: {f_icon} {sc.feasibility}")
        print(f"      ETA        : {sc.time_to_impact_days}d  |  Cost ≈ {sc.estimated_cost_usd or 'N/A'}")
        if sc.second_order_effects:
            print(f"      Side-effect: {sc.second_order_effects[0][:80]}")
        print()

    print(f"  Note: {bundle.summary_note[:120]}")
    print("=" * 65)
    print("[→] Ready for Layer 5 (RAG solution mapping)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SUPERVISOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_layer4(
    input_path:        str = str(DEFAULT_INPUT_PATH),
    output_path:       str = str(DEFAULT_OUTPUT_PATH),
    scenarios_per_risk: int = SCENARIOS_PER_RISK,
) -> CounterfactualBundle:
    """
    Full Layer 4 pipeline — entry point for layer0.py's run_pipeline().

    1. Load Layer 3 RiskReport
    2. Build typed AnalystWorkOrders (one per top-risk)
    3. Dispatch each to the AnalystAgent
    4. Guardrail-validate every AnalystWorkResult
    5. Assemble CounterfactualBundle
    6. Write to disk
    """

    print("\n" + "=" * 65)
    print("  LAYER 4 — COUNTERFACTUAL SIMULATION  [Supervisor + Analyst]")
    print("=" * 65)

    # ── 1. Load input ─────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading RiskReport from '{input_path}' …")
    with open(input_path, "r") as fh:
        report = json.load(fh)

    domain      = report.get("domain", "custom_domain")
    analysed_at = report.get("analysed_at", "")
    top_risks   = report.get("top_risks", [])
    print(f"    Domain      : {domain}")
    print(f"    Top risks   : {len(top_risks)}")

    if not top_risks:
        raise ValueError("No top_risks found in risk_report.json — run Layer 3 first.")

    # ── 2. Instantiate the specialised worker ─────────────────────────────────
    analyst = AnalystAgent()
    print(f"\n[2/4] Analyst Agent initialised "
          f"(contract v{analyst.CONTRACT_VERSION})")

    # ── 3. Dispatch work orders ───────────────────────────────────────────────
    print(f"\n[3/4] Dispatching {len(top_risks)} work orders "
          f"({scenarios_per_risk} scenarios each) …")

    all_scenarios: list[CounterfactualScenario] = []
    failed_orders: list[str] = []

    for risk in top_risks:
        rank = risk.get("rank", "?")
        print(f"\n    → Dispatching WorkOrder for Risk #{rank}: "
              f"{risk.get('title', '')[:50]}")

        # Build the typed work order — fails fast on malformed Layer 3 output
        try:
            order = _build_work_order(risk, domain, scenarios_per_risk)
        except (ValidationError, KeyError) as exc:
            logger.error("Failed to build WorkOrder for Risk #%s: %s", rank, exc)
            failed_orders.append(f"Risk #{rank}: WorkOrder build failed — {exc}")
            continue

        # Dispatch to Analyst Agent
        result: AnalystWorkResult = analyst.execute(order)

        if result.error and not result.scenarios:
            logger.error(
                "Supervisor: WorkResult for '%s' has no valid scenarios: %s",
                result.request_id, result.error,
            )
            failed_orders.append(f"Risk #{rank}: {result.error}")
            continue

        # Guardrail: re-validate every scenario in the result
        valid_scenarios = _validate_result(result)

        if not valid_scenarios:
            msg = (f"Supervisor KILLED WorkResult '{result.request_id}' — "
                   f"all scenarios failed re-validation.")
            logger.error(msg)
            failed_orders.append(msg)
            continue

        print(f"      ✓ {len(valid_scenarios)} scenarios passed guardrail for Risk #{rank}")
        all_scenarios.extend(valid_scenarios)

    # ── 4. Assemble bundle ────────────────────────────────────────────────────
    print(f"\n[4/4] Assembling and persisting CounterfactualBundle …")

    feasibility_dist = dict(
        Counter(sc.feasibility for sc in all_scenarios)
    )
    avg_delta = (
        sum(sc.delta_probability for sc in all_scenarios) / len(all_scenarios)
        if all_scenarios else 0.0
    )
    
    # Calculate Macro Counterfactual Mitigation Efficiency
    mitigation_efficiency = _calculate_mitigation_efficiency(all_scenarios)

    note_parts = [
        f"Supervisor dispatched {len(top_risks)} WorkOrders; "
        f"{len(all_scenarios)} scenarios passed all guardrail checks.",
    ]
    if failed_orders:
        note_parts.append(f"Failed orders: {'; '.join(failed_orders)}.")
    note_parts.append(
        "All delta_probability values were auto-verified for arithmetic consistency. "
        "Scenarios with invalid schemas were dropped before bundle assembly."
    )

    bundle = CounterfactualBundle(
        simulated_at       = datetime.now(timezone.utc).isoformat(),
        domain             = domain,
        model_used         = "engine/heuristic-v1 via AnalystAgent",
        layer3_analysed_at = analysed_at,
        scenarios          = all_scenarios,
        summary_note       = " ".join(note_parts),
        total_risks        = len(top_risks),
        total_scenarios    = len(all_scenarios),
        feasibility_dist   = feasibility_dist,
        avg_delta          = round(avg_delta, 4),
        mitigation_efficiency = mitigation_efficiency,
    )

    # Persist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(bundle.model_dump(), fh, indent=2, default=str)
    print(f"    ✓ CounterfactualBundle written to '{output_path}'")

    _print_summary(bundle)
    return bundle


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Layer 4 Supervisor — Counterfactual Simulation (Supervisor + Analyst pattern)"
    )
    parser.add_argument("--input",  default=str(DEFAULT_INPUT_PATH),  help="Layer 3 risk_report.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output counterfactual_bundle.json")
    parser.add_argument(
        "--scenarios-per-risk", type=int, default=SCENARIOS_PER_RISK, dest="scenarios_per_risk",
        help="Number of counterfactual scenarios per risk (default: 2)",
    )
    args = parser.parse_args()
    run_layer4(
        input_path         = args.input,
        output_path        = args.output,
        scenarios_per_risk = args.scenarios_per_risk,
    )
