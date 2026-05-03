from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

# Engine is the only external dependency this agent is allowed to import
from engine import simulate
from contracts import (
    AnalystWorkOrder,
    AnalystWorkResult,
    CounterfactualScenario,
)

logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Specialised worker agent for counterfactual simulation.

    Instantiate once; call `.execute(order)` for each work order.
    The agent is stateless between calls — no memory, no side effects.
    """

    # Name shown in logs; helps distinguish this worker from others
    AGENT_NAME = "AnalystAgent[Layer4]"
    # Version lets the Supervisor reject stale agents if the contract changes
    CONTRACT_VERSION = "1.0"

    def execute(self, order: AnalystWorkOrder) -> AnalystWorkResult:
        """
        Main entry point called by the Layer 4 Supervisor.

        Steps:
          1. Log receipt of work order
          2. Delegate to engine.simulate()
          3. Validate each raw scenario → CounterfactualScenario
          4. Build and return AnalystWorkResult

        Never raises — always returns a result; errors go into result.error.
        """
        logger.info(
            "%s | WorkOrder %s | Risk #%d: '%s' | Requesting %d scenarios",
            self.AGENT_NAME,
            order.request_id,
            order.risk_rank,
            order.risk_title[:50],
            order.n_scenarios,
        )

        error_messages: list[str] = []
        validated_scenarios: list[CounterfactualScenario] = []

        # ── Step 1: Call the engine (no LLM, no network) ──────────────────────
        try:
            raw_scenarios: list[dict[str, Any]] = simulate(order)
            logger.debug(
                "%s | Engine returned %d raw scenarios for WorkOrder %s",
                self.AGENT_NAME, len(raw_scenarios), order.request_id,
            )
        except Exception as engine_exc:
            error_msg = f"Engine simulation failed: {engine_exc}"
            logger.error("%s | %s | WorkOrder %s", self.AGENT_NAME, error_msg, order.request_id)
            return AnalystWorkResult(
                request_id = order.request_id,
                risk_rank  = order.risk_rank,
                scenarios  = [],
                model_used = "engine/heuristic-v1",
                error      = error_msg,
            )

        # ── Step 2: Validate each raw scenario against the contract ───────────
        for idx, raw in enumerate(raw_scenarios):
            try:
                scenario = CounterfactualScenario(**raw)
                validated_scenarios.append(scenario)
                logger.debug(
                    "%s | ✓ Scenario %s validated (Δp=%+.2f%%)",
                    self.AGENT_NAME,
                    scenario.scenario_id,
                    scenario.delta_probability * 100,
                )
            except ValidationError as val_err:
                # Log and DROP the invalid scenario — it must not flow upstream
                msg = (
                    f"Scenario #{idx + 1} from Risk #{order.risk_rank} "
                    f"failed contract validation and was dropped: {val_err}"
                )
                logger.warning("%s | %s", self.AGENT_NAME, msg)
                error_messages.append(msg)

        # ── Step 3: Guard — at least one valid scenario must survive ──────────
        if not validated_scenarios:
            error_summary = "; ".join(error_messages) or "All scenarios failed schema validation."
            logger.error(
                "%s | WorkOrder %s produced ZERO valid scenarios.",
                self.AGENT_NAME, order.request_id,
            )
            return AnalystWorkResult(
                request_id = order.request_id,
                risk_rank  = order.risk_rank,
                scenarios  = [],
                model_used = "engine/heuristic-v1",
                error      = error_summary,
            )

        logger.info(
            "%s | WorkOrder %s complete — %d/%d scenarios validated ✓",
            self.AGENT_NAME,
            order.request_id,
            len(validated_scenarios),
            len(raw_scenarios),
        )

        return AnalystWorkResult(
            request_id = order.request_id,
            risk_rank  = order.risk_rank,
            scenarios  = validated_scenarios,
            model_used = "engine/heuristic-v1",
            error      = "; ".join(error_messages) if error_messages else None,
        )
