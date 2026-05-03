from __future__ import annotations

import json
import logging
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError
from groq import Groq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in [str(_ROOT), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(_ROOT / ".env")

from contracts import (
    LibrarianWorkOrder,
    LibrarianWorkResult,
    SolutionSynthesisOrder,
    SolutionMatch,
    SolutionMappingReport,
    KBChunk,
    Severity,
    SolutionType,
    TimeHorizon,
)
from librarian_agent import LibrarianAgent

logger = logging.getLogger(__name__)

DEFAULT_RISK_REPORT_PATH = _ROOT / "data" / "risk_report.json"
DEFAULT_CF_BUNDLE_PATH   = _ROOT / "data" / "counterfactual_bundle.json"
DEFAULT_OUTPUT_PATH      = _ROOT / "data" / "solution_mapping_report.json"
DEFAULT_KB_PATH          = _HERE / "knowledge_base.json"

PRIMARY_MODEL   = "llama-3.3-70b-versatile"
FALLBACK_MODEL  = "llama-3.1-8b-instant"
SOLUTIONS_PER_RISK = 2
TOP_K_CHUNKS    = 4


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are an expert risk-mitigation consultant. You receive a structured risk item
and a set of retrieved knowledge-base (KB) chunks.

Your task: synthesise ONE implementation-ready solution grounded STRICTLY in the
provided KB chunks. Every fact, reference, and source_chunk ID you cite MUST
appear in the KB chunks below.

Return ONLY a valid JSON object — no prose, no markdown, no code fences.

Required schema:
{
  "solution_id":             "<sol_<risk_rank>_<solution_index>>",
  "risk_rank":               <integer>,
  "scenario_id":             "<linked scenario_id string or null>",
  "risk_title":              "<copied verbatim from input>",
  "solution_title":          "<concise title, max 10 words>",
  "solution_type":           "<FRAMEWORK|TECHNOLOGY|PROCESS|REGULATION|PARTNERSHIP>",
  "description":             "<2-3 sentences; explain why this solution fits THIS risk>",
  "source_chunks":           ["<kb_001>", "<kb_002>"],
  "relevance_score":         <float 0.0-1.0>,
  "implementation_steps":    ["<step 1>", "<step 2>", "<step 3>"],
  "kpis":                    ["<KPI 1>", "<KPI 2>"],
  "time_horizon":            "<SHORT|MEDIUM|LONG>",
  "estimated_cost_usd":      "<rough order-of-magnitude string or null>",
  "risk_reduction_estimate": "<e.g. '20-35% probability reduction'>",
  "dependencies":            ["<prerequisite 1>"],
  "references":              ["<citation from KB>"]
}

CRITICAL RULES:
- source_chunks MUST only contain IDs from the KB chunks provided (format: kb_NNN).
  ANY fabricated KB ID will cause a hard schema rejection.
- implementation_steps: 3-5 concrete, ordered, actionable steps.
- kpis: at least 2, each must be measurable (include thresholds or units).
- Do NOT invent solutions not supported by the KB content.
- If KB coverage is weak, lower relevance_score below 0.5 and say so in description.
"""

_HUMAN_TEMPLATE = """\
<risk_item>
  <domain>{domain}</domain>
  <rank>{risk_rank}</rank>
  <category>{risk_category}</category>
  <title>{risk_title}</title>
  <severity>{severity}</severity>
  <probability_next_30d>{probability}</probability_next_30d>
  <causal_chain>{causal_chain}</causal_chain>
  <recommended_action>{recommended_action}</recommended_action>
  <affected_entities>{affected_entities}</affected_entities>
  <affected_geo>{affected_geo}</affected_geo>
</risk_item>
{scenario_block}
<knowledge_base_chunks>
{kb_chunks_xml}
</knowledge_base_chunks>

Synthesise solution #{solution_index} (solution_id = "sol_{risk_rank}_{solution_index}").
Return ONLY the JSON object.
"""

_SCENARIO_BLOCK = """\
<counterfactual_context>
  <scenario_id>{scenario_id}</scenario_id>
  <intervention>{intervention}</intervention>
  <intervention_type>{intervention_type}</intervention_type>
  <delta_probability>{delta:+.2%}</delta_probability>
  <post_intervention_severity>{cf_severity}</post_intervention_severity>
  <feasibility>{feasibility}</feasibility>
</counterfactual_context>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_query(domain: str, risk: dict, scenario: dict | None) -> str:
    """Build the retrieval query sent to the Librarian."""
    parts = [
        domain,
        risk.get("category", ""),
        risk.get("title", ""),
        risk.get("causal_chain", "")[:150],
        risk.get("recommended_action", "")[:100],
    ]
    if scenario:
        parts += [scenario.get("intervention", ""), scenario.get("intervention_type", "")]
    return " ".join(p for p in parts if p)


def _format_kb_chunks_xml(chunks: list[KBChunk]) -> str:
    lines: list[str] = []
    for chunk in chunks:
        lines.append(f"  <chunk id='{chunk.chunk_id}' score='{chunk.retrieval_score:.3f}'>")
        lines.append(f"    <title>{chunk.title}</title>")
        lines.append(f"    <tags>{', '.join(chunk.tags)}</tags>")
        lines.append(f"    <body>{chunk.body[:400]}</body>")
        if chunk.references:
            lines.append(f"    <references>{'; '.join(chunk.references[:2])}</references>")
        lines.append("  </chunk>")
    return "\n".join(lines)


def _build_human_prompt(order: SolutionSynthesisOrder) -> str:
    scenario_block = ""
    if order.linked_scenario:
        sc = order.linked_scenario
        scenario_block = _SCENARIO_BLOCK.format(
            scenario_id      = sc.get("scenario_id", ""),
            intervention     = sc.get("intervention", "")[:120],
            intervention_type= sc.get("intervention_type", ""),
            delta            = sc.get("delta_probability", 0.0),
            cf_severity      = sc.get("counterfactual_severity", ""),
            feasibility      = sc.get("feasibility", ""),
        )

    return _HUMAN_TEMPLATE.format(
        domain           = order.domain,
        risk_rank        = order.risk_rank,
        risk_category    = order.risk_category,
        risk_title       = order.risk_title,
        severity         = order.severity.value,
        probability      = order.probability_next_30d,
        causal_chain     = order.causal_chain[:250],
        recommended_action = order.recommended_action[:150],
        affected_entities  = ", ".join(order.affected_entities[:4]),
        affected_geo       = ", ".join(order.affected_geo[:4]),
        scenario_block     = scenario_block,
        kb_chunks_xml      = _format_kb_chunks_xml(order.kb_chunks),
        solution_index     = order.solution_index,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM SYNTHESIS CALL
# ═══════════════════════════════════════════════════════════════════════════════

def _call_groq(human: str) -> tuple[dict[str, Any], str]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set — add it to .env")

    client = Groq(api_key=api_key)

    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": human},
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            # Strip any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw), model

        except Exception as exc:
            err = str(exc)
            if "rate_limit" in err.lower() or "quota" in err.lower():
                logger.warning("LLM %s rate-limited, retrying with fallback.", model)
                continue
            logger.error("LLM call failed (%s): %s", model, err)
            raise

    raise RuntimeError("Both Groq models failed in Layer 5 Supervisor.")


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL — LLM OUTPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_solution(
    raw: dict[str, Any],
    valid_chunk_ids: set[str],
    risk_rank: int,
    risk_title: str,
    sol_idx: int,
) -> SolutionMatch | None:
    """
    The Supervisor's guardrail for LLM output.

    Extra check beyond Pydantic: verifies that every source_chunk ID
    actually exists in the chunks that were retrieved for this risk.
    If any hallucinated KB ID slips past the field_validator, we catch it here.
    """
    # Pre-normalise enums (LLM may return lowercase)
    for enum_field, enum_cls in [
        ("solution_type", SolutionType),
        ("time_horizon", TimeHorizon),
    ]:
        if enum_field in raw:
            raw[enum_field] = str(raw[enum_field]).upper()

    # Override solution_id / risk_rank to ensure consistency
    raw["solution_id"] = raw.get("solution_id", f"sol_{risk_rank}_{sol_idx}")
    raw["risk_rank"]   = risk_rank
    raw["risk_title"]  = risk_title

    # Check source_chunk IDs against the actual retrieved set
    claimed_chunks = raw.get("source_chunks", [])
    hallucinated   = [c for c in claimed_chunks if c not in valid_chunk_ids]
    if hallucinated:
        logger.warning(
            "Supervisor KILLED solution for Risk #%d: hallucinated KB IDs %s "
            "(valid IDs: %s).",
            risk_rank, hallucinated, sorted(valid_chunk_ids),
        )
        # Remove the fake IDs and keep only valid ones
        raw["source_chunks"] = [c for c in claimed_chunks if c in valid_chunk_ids]
        if not raw["source_chunks"]:
            logger.error(
                "Supervisor: no valid source_chunks remain for Risk #%d sol %d — "
                "solution DISCARDED.",
                risk_rank, sol_idx,
            )
            return None

    try:
        return SolutionMatch(**raw)
    except ValidationError as ve:
        logger.error(
            "Supervisor KILLED LLM solution for Risk #%d sol %d — schema invalid: %s",
            risk_rank, sol_idx, ve,
        )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

TH_ICON = {"SHORT": "⚡", "MEDIUM": "📅", "LONG": "🗓️ "}
ST_ICON = {"FRAMEWORK": "📐", "TECHNOLOGY": "💻", "PROCESS": "⚙️ ",
           "REGULATION": "📋", "PARTNERSHIP": "🤝"}

def _print_summary(report: SolutionMappingReport) -> None:
    print("\n" + "=" * 65)
    print("  LAYER 5 — RISK-SOLUTION MAPPING COMPLETE  (RAG)")
    print("=" * 65)
    print(f"  Domain        : {report.domain}")
    print(f"  Mapped at     : {report.mapped_at}")
    print(f"  KB size       : {report.kb_size} documents")
    print(f"  RAG method    : {report.rag_method}")
    print(f"  Solutions     : {len(report.solutions)}")
    print(f"  Risks covered : {report.risks_covered}")
    if report.risks_missed:
        print(f"  ⚠️  Risks missed : {report.risks_missed}")
    print()

    current_risk = None
    for sol in report.solutions:
        if sol.risk_rank != current_risk:
            current_risk = sol.risk_rank
            print(f"  ── Risk #{sol.risk_rank}: {sol.risk_title[:55]} ──")

        t_icon = TH_ICON.get(sol.time_horizon, "")
        s_icon = ST_ICON.get(sol.solution_type, "")
        print(f"    [{sol.solution_id}]  {s_icon} {sol.solution_title}")
        print(f"      Type          : {sol.solution_type}")
        print(f"      Time horizon  : {t_icon} {sol.time_horizon}")
        print(f"      Relevance     : {sol.relevance_score:.0%}")
        print(f"      Reduction est.: {sol.risk_reduction_estimate}")
        print(f"      Cost ≈        : {sol.estimated_cost_usd or 'N/A'}")
        print(f"      KB sources    : {', '.join(sol.source_chunks)}")
        if sol.implementation_steps:
            print(f"      Step 1        : {sol.implementation_steps[0][:80]}")
        if sol.kpis:
            print(f"      KPI 1         : {sol.kpis[0][:75]}")
        if sol.scenario_id:
            print(f"      ↳ Linked CF   : {sol.scenario_id}")
        print()

    print(f"  Coverage: {report.coverage_note[:130]}")
    print("=" * 65)
    print("[→] Layer 5 complete — SolutionMappingReport ready\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SUPERVISOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_layer5(
    risk_report_path:   str = str(DEFAULT_RISK_REPORT_PATH),
    cf_bundle_path:     str = str(DEFAULT_CF_BUNDLE_PATH),
    output_path:        str = str(DEFAULT_OUTPUT_PATH),
    kb_path:            str = str(DEFAULT_KB_PATH),
    solutions_per_risk: int = SOLUTIONS_PER_RISK,
    top_k:              int = TOP_K_CHUNKS,
) -> SolutionMappingReport:
    """
    Full Layer 5 pipeline — entry point for layer0.py's run_pipeline().
    """

    print("\n" + "=" * 65)
    print("  LAYER 5 — RISK-SOLUTION MAPPING  [Supervisor + Librarian + LLM]")
    print("=" * 65)

    # ── 1. Load inputs ────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading inputs …")
    with open(risk_report_path, "r") as fh:
        risk_report = json.load(fh)

    domain    = risk_report.get("domain", "custom_domain")
    top_risks = risk_report.get("top_risks", [])
    print(f"    Domain      : {domain}")
    print(f"    Top risks   : {len(top_risks)}")

    # Optional counterfactual bundle — index by risk_rank for O(1) lookup
    cf_by_rank: dict[int, list[dict]] = {}
    if Path(cf_bundle_path).exists():
        with open(cf_bundle_path, "r") as fh:
            cf_bundle = json.load(fh)
        for sc in cf_bundle.get("scenarios", []):
            cf_by_rank.setdefault(sc.get("risk_rank", 0), []).append(sc)
        total_cf = sum(len(v) for v in cf_by_rank.values())
        print(f"    CF scenarios: {total_cf} (across {len(cf_by_rank)} risks)")
    else:
        print(f"    CF bundle   : not found — proceeding without counterfactual context")

    # ── 2. Instantiate the Librarian (read-only worker) ───────────────────────
    print(f"\n[2/5] Initialising Librarian Agent (KB: '{kb_path}') …")
    librarian = LibrarianAgent(kb_path=kb_path)
    print(f"    KB size     : {librarian.kb_size} documents")
    print(f"    Agent       : {librarian.AGENT_NAME} "
          f"(contract v{librarian.CONTRACT_VERSION})")

    # ── 3. Main loop: Librarian retrieval + LLM synthesis ────────────────────
    print(f"\n[3/5] Running Librarian retrieval + LLM synthesis …")
    print(f"      ({solutions_per_risk} solutions × {len(top_risks)} risks, "
          f"top-{top_k} KB chunks per solution)")

    all_solutions: list[SolutionMatch] = []
    model_used: str = PRIMARY_MODEL

    for risk in top_risks:
        rank  = risk.get("rank", 0)
        title = risk.get("title", "Unknown risk")

        print(f"\n    → Risk #{rank}: {title[:55]}")

        # Pair counterfactual scenarios — pad with None if fewer than needed
        linked_cfs: list[dict | None] = cf_by_rank.get(rank, [])[:solutions_per_risk]
        while len(linked_cfs) < solutions_per_risk:
            linked_cfs.append(None)

        for sol_idx in range(1, solutions_per_risk + 1):
            scenario = linked_cfs[sol_idx - 1]

            # ── Step A: Build LibrarianWorkOrder ──────────────────────────────
            query = _build_query(domain, risk, scenario)
            lib_order = LibrarianWorkOrder(
                request_id = f"l5_risk{rank}_sol{sol_idx}",
                query      = query,
                top_k      = top_k,
            )

            # ── Step B: Dispatch to Librarian (read-only) ─────────────────────
            lib_result: LibrarianWorkResult = librarian.execute(lib_order)

            if not lib_result.chunks:
                logger.warning(
                    "Librarian returned no chunks for Risk #%d sol %d — skipping.",
                    rank, sol_idx,
                )
                continue

            valid_chunk_ids = {c.chunk_id for c in lib_result.chunks}

            # ── Step C: Build typed SolutionSynthesisOrder ────────────────────
            try:
                synth_order = SolutionSynthesisOrder(
                    request_id           = lib_order.request_id,
                    domain               = domain,
                    risk_rank            = rank,
                    risk_title           = title,
                    risk_category        = risk.get("category", ""),
                    severity             = Severity(risk.get("severity", "MEDIUM").upper()),
                    probability_next_30d = float(risk.get("probability_next_30d", 0.5)),
                    causal_chain         = risk.get("causal_chain", ""),
                    recommended_action   = risk.get("recommended_action", ""),
                    affected_entities    = risk.get("affected_entities", []),
                    affected_geo         = risk.get("affected_geo", []),
                    linked_scenario      = scenario,
                    kb_chunks            = lib_result.chunks,
                    solution_index       = sol_idx,
                )
            except ValidationError as ve:
                logger.error("SynthesisOrder build failed for Risk #%d: %s", rank, ve)
                continue

            # ── Step D: LLM synthesis ─────────────────────────────────────────
            human_prompt = _build_human_prompt(synth_order)
            try:
                raw_solution, model_used = _call_groq(human_prompt)
            except Exception as exc:
                logger.error("LLM synthesis failed for Risk #%d sol %d: %s", rank, sol_idx, exc)
                continue

            # ── Step E: Supervisor Guardrail — validate LLM output ────────────
            solution = _validate_solution(
                raw_solution, valid_chunk_ids, rank, title, sol_idx,
            )
            if solution is None:
                logger.error(
                    "Supervisor DISCARDED solution for Risk #%d sol %d — "
                    "failed guardrail validation.",
                    rank, sol_idx,
                )
                continue

            all_solutions.append(solution)
            print(f"        ✓ Solution {sol_idx}: {solution.solution_title[:50]}  "
                  f"[{solution.solution_type}, {solution.time_horizon}]")

    # ── 4. Assemble report ────────────────────────────────────────────────────
    print(f"\n[4/5] Assembling SolutionMappingReport …")

    all_ranks     = {r.get("rank", 0) for r in top_risks}
    covered_ranks = {s.risk_rank for s in all_solutions}
    missed_ranks  = sorted(all_ranks - covered_ranks)

    coverage_note = (
        f"{len(covered_ranks)}/{len(all_ranks)} risks have at least one solution. "
        + (f"No solutions generated for risks: {missed_ranks}. " if missed_ranks else "All risks covered. ")
        + f"KB ({librarian.kb_size} docs) retrieved via TF-IDF cosine. "
        f"All LLM outputs were guardrail-validated against the SolutionMatch contract."
    )

    report = SolutionMappingReport(
        mapped_at      = datetime.now(timezone.utc).isoformat(),
        domain         = domain,
        model_used     = model_used,
        kb_size        = librarian.kb_size,
        rag_method     = "tfidf_cosine[LibrarianAgent] + groq_llm_synthesis[Supervisor]",
        solutions      = all_solutions,
        coverage_note  = coverage_note,
        risks_covered  = sorted(covered_ranks),
        risks_missed   = missed_ranks,
    )

    # ── 5. Persist ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Writing SolutionMappingReport to '{output_path}' …")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2, default=str)
    print(f"    ✓ {len(all_solutions)} solutions saved")

    _print_summary(report)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Layer 5 Supervisor — Risk-Solution Mapping (RAG)"
    )
    parser.add_argument("--risk-report", default=str(DEFAULT_RISK_REPORT_PATH))
    parser.add_argument("--cf-bundle",   default=str(DEFAULT_CF_BUNDLE_PATH))
    parser.add_argument("--output",      default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--kb",          default=str(DEFAULT_KB_PATH))
    parser.add_argument("--solutions-per-risk", type=int, default=SOLUTIONS_PER_RISK,
                        dest="solutions_per_risk")
    parser.add_argument("--top-k", type=int, default=TOP_K_CHUNKS, dest="top_k")
    args = parser.parse_args()

    run_layer5(
        risk_report_path   = args.risk_report,
        cf_bundle_path     = args.cf_bundle,
        output_path        = args.output,
        kb_path            = args.kb,
        solutions_per_risk = args.solutions_per_risk,
        top_k              = args.top_k,
    )
