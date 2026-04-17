from __future__ import annotations

import os
from typing import List, Optional, Tuple

import networkx as nx
from groq import Groq

from schemas_layer4 import AgentMemory, CounterfactualResult, InterventionParams, ObservedRiskState
from tool_get_causal_paths import get_causal_paths_tool
from tool_log_intervention import log_intervention_tool
from tool_run_counterfactual import run_counterfactual_tool


DEFAULT_MODEL = "llama3-70b-8192"


def _is_reduction_variable(var: str) -> bool:
    return var in {
        "port_congestion",
        "weather_severity",
        "geopolitical_tension",
        "shipping_delay",
        "inventory_shortage",
        "demand_shock",
    }


def _propose_intervention(
    observed: ObservedRiskState,
    variable: str,
    iteration: int,
    previous_result: Optional[CounterfactualResult],
) -> InterventionParams:
    current_value = float(getattr(observed, variable))

    # Baseline attempt then magnitude tuning based on last ITE feedback.
    delta = 0.12 + 0.05 * max(0, iteration - 2)
    if previous_result is not None and previous_result.probability_of_improvement < 0.5:
        delta += 0.05

    if _is_reduction_variable(variable):
        new_value = max(0.0, current_value - delta)
        rationale = (
            f"Reduce {variable} from {current_value:.2f} to {new_value:.2f} based on causal leverage and prior ITE feedback."
        )
    else:
        new_value = min(1.0, current_value + delta)
        rationale = (
            f"Increase {variable} from {current_value:.2f} to {new_value:.2f} because higher reliability reduces downstream shortage."
        )

    return InterventionParams(
        variable=variable,
        intervened_value=float(new_value),
        rationale=rationale,
        iteration=iteration,
    )


def _maybe_create_groq_client() -> Optional[Groq]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _reflect_with_groq(
    client: Groq,
    observed_state: ObservedRiskState,
    result: CounterfactualResult,
    causal_info: dict,
) -> Optional[str]:
    """Optional concise reflection; does not alter the deterministic intervention policy."""
    try:
        prompt = (
            "You are assisting a causal intervention agent. "
            "Given current observed risk state and latest ITE result, "
            "provide one concise reasoning sentence on whether to increase intervention magnitude "
            "or switch variable next. Do not output JSON.\n\n"
            f"Risk category: {observed_state.category}\n"
            f"Observed severity: {observed_state.risk_severity:.4f}\n"
            f"Latest variable: {result.intervention.variable}\n"
            f"Latest ITE mean: {result.ite_mean:.4f}\n"
            f"Latest P(improve): {result.probability_of_improvement:.4f}\n"
            f"Recommended variables: {causal_info.get('recommended_variables', [])}\n"
        )
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "Respond in exactly one short sentence."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # Reflection is optional and should never break the main loop.
        return None


def _pick_variable(
    recommended_variables: List[str],
    tried: List[str],
    previous_result: Optional[CounterfactualResult],
) -> str:
    # Iteration 2 starts with highest-weight variable, then escalate or switch.
    if previous_result is None:
        return recommended_variables[0]

    if previous_result.ite_mean > -0.05:
        # Weak effect: try next high-weight variable not yet attempted.
        for var in recommended_variables:
            if var not in tried:
                return var

    # Otherwise keep the same variable and tune magnitude.
    return previous_result.intervention.variable


def run_agentic_counterfactual_loop(
    observed_state: ObservedRiskState,
    dag: nx.DiGraph,
    fitted_scm,
    max_iterations: int = 5,
) -> Tuple[AgentMemory, CounterfactualResult]:
    """ReAct-style agent loop with causal tools and ITE-driven feedback."""
    memory = AgentMemory(risk_id=observed_state.risk_id, risk_title=observed_state.title)
    groq_client = _maybe_create_groq_client()

    # Iteration 1: always inspect causal structure first.
    causal_info = get_causal_paths_tool(observed_state, dag)
    recommended = list(causal_info["recommended_variables"])
    if not recommended:
        recommended = ["inventory_shortage", "shipping_delay", "demand_shock"]

    memory.metadata["causal_info"] = causal_info
    memory.notes.append("Iteration 1: causal paths computed.")

    previous_result: Optional[CounterfactualResult] = None
    tried_variables: List[str] = []

    for iteration in range(2, max_iterations + 1):
        variable = _pick_variable(recommended, tried_variables, previous_result)
        intervention = _propose_intervention(observed_state, variable, iteration, previous_result)

        if variable not in tried_variables:
            tried_variables.append(variable)

        # Core feedback signal: counterfactual ITE on this observed state.
        result = run_counterfactual_tool(observed_state, intervention, fitted_scm)
        memory.tried_interventions.append(intervention)
        memory.all_results.append(result)

        reasoning_note = (
            f"Iter {iteration}: var={variable}, ite_mean={result.ite_mean:.4f}, "
            f"p_improve={result.probability_of_improvement:.4f}"
        )
        memory.notes.append(reasoning_note)

        # Optional single-sentence reflection, if GROQ key is available.
        if groq_client is not None:
            reflection = _reflect_with_groq(groq_client, observed_state, result, causal_info)
            if reflection:
                memory.notes.append(f"LLM_reflection: {reflection}")

        if result.threshold_cleared:
            # Keep the tool invocation semantics while retaining full in-memory history.
            log_intervention_tool(result, memory)
            return memory, result

        # Keep best candidate in memory even before final logging.
        if memory.best_result is None or result.ite_mean < memory.best_result.ite_mean:
            memory.best_result = result

        previous_result = result

    # After max iterations: log best found regardless.
    if memory.best_result is None:
        fallback_intervention = InterventionParams(
            variable=recommended[0],
            intervened_value=max(0.0, float(getattr(observed_state, recommended[0])) - 0.1),
            rationale="Fallback intervention after no successful iteration.",
            iteration=max_iterations,
        )
        fallback_result = run_counterfactual_tool(observed_state, fallback_intervention, fitted_scm)
        memory.tried_interventions.append(fallback_intervention)
        memory.all_results.append(fallback_result)
        memory.best_result = fallback_result

    log_intervention_tool(memory.best_result, memory)
    return memory, memory.best_result
