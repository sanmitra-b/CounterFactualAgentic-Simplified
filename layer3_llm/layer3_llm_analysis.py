"""
layer3_llm_analysis.py — Layer 3: LLM Risk Analysis (Groq · LLaMA-3 70B)
=========================================================================
Architecture (from design):
  ┌──────────────────────────────────────────────────────────────┐
  │                  Top 5 Risk Report (JSON)                    │
  │   rank · category · severity · confidence · evidence · entities│
  └──────────────────────────────────────────────────────────────┘
        ▲                       ▲
  Prompt Orchestrator      Groq Inference
  LangChain PromptTemplate  llama3-70b-8192
  Token budget / ctx window  Structured JSON output
  Fallback: Together.AI       Top 5 + soft risks
 
Input  : enriched_risk_bundle.json  (Layer 2 output)
Output : risk_report.json           (Layer 4 input)
 
Pipeline steps:
  1. Load EnrichedRiskInputBundle from JSON
  2. Prompt Orchestrator — build context window within token budget
       · Rank signals by reliability × |sentiment_score|
       · Serialise top-N signals per category into XML sections
       · Inject aggregate NLP metadata + geo_tags
  3. Groq Inference — structured JSON output (Top 5 risks + soft risks)
  4. Parse + validate via Pydantic (RiskReport)
  5. Persist risk_report.json → ready for Layer 4
 
Usage:
    python layer3_llm/layer3_llm_analysis.py
    python layer3_llm/layer3_llm_analysis.py --input data/enriched_risk_bundle.json
    python layer3_llm/layer3_llm_analysis.py --output data/risk_report.json
"""
 
from __future__ import annotations
 
import os
import json
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
 
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "enriched_risk_bundle.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "risk_report.json"

load_dotenv(ROOT_DIR / ".env")
 
warnings.filterwarnings("ignore")
logging.getLogger("groq").setLevel(logging.ERROR)
 
from groq import Groq
from schemas_layer3 import (
    RiskItem,
    SoftRisk,
    RiskReport,
)
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
 
PRIMARY_MODEL  = "llama-3.3-70b-versatile"   
FALLBACK_MODEL = "llama-3.1-8b-instant"      
MAX_CONTEXT_CHARS = 12_000               # safe token budget (~3 000 tokens)
TOP_N_PER_CATEGORY = 3                   # max signals per category in prompt
 
# Risk categories aligned with supply chain domain
RISK_CATEGORIES = [
    "Port Congestion",
    "Shipping Delay",
    "Commodity Price Shock",
    "Geopolitical Disruption",
    "Weather / Natural Disaster",
    "Financial Market Volatility",
    "Supplier Insolvency",
    "Demand Shock",
    "Regulatory / Trade Policy",
    "Cyber / Infrastructure Risk",
]
 
# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
 
SYSTEM_PROMPT = """\
You are a world-class supply chain risk analyst with expertise in global logistics,
commodity markets, geopolitics, and financial risk modelling.
 
Your task is to analyse a structured intelligence bundle collected from news feeds,
social media, financial markets, port data, weather stations, and commodity prices.
Each signal has already been enriched with NLP sentiment scores and named entity tags.
 
You MUST return ONLY a valid JSON object — no prose, no markdown, no code fences.
The JSON schema is:
 
{
  "analysed_at": "<ISO timestamp>",
  "domain": "<domain string>",
  "top_risks": [
    {
      "rank": <1-5>,
      "category": "<risk category>",
      "title": "<concise risk title, max 12 words>",
      "severity": "<CRITICAL|HIGH|MEDIUM|LOW>",
      "confidence": <0.0-1.0>,
      "probability_next_30d": <0.0-1.0>,
      "evidence": ["<evidence point 1>", "<evidence point 2>", ...],
      "affected_entities": ["<org/location 1>", ...],
      "affected_geo": ["<country or city>", ...],
      "causal_chain": "<brief cause → effect chain, 1-2 sentences>",
      "recommended_action": "<immediate mitigation recommendation>"
    }
  ],
  "soft_risks": [
    {
      "category": "<risk category>",
      "title": "<concise title>",
      "note": "<1 sentence observation>"
    }
  ],
  "data_quality_note": "<comment on completeness/reliability of input data>"
}
 
Rules:
- Produce exactly 5 items in top_risks (ranked 1=highest severity).
- Produce 3-5 items in soft_risks (emerging or low-confidence signals).
- severity must be one of: CRITICAL, HIGH, MEDIUM, LOW.
- confidence and probability_next_30d must be floats between 0.0 and 1.0.
- Base ALL claims strictly on the provided intelligence bundle.
- If data is sparse, reflect lower confidence scores and note it in data_quality_note.
- Do NOT hallucinate events not present in the bundle.
"""
 
HUMAN_PROMPT_TEMPLATE = """\
<intelligence_bundle>
  <metadata>
    <domain>{domain}</domain>
    <fetched_at>{fetched_at}</fetched_at>
    <enriched_at>{enriched_at}</enriched_at>
    <layer1_completeness>{completeness}</layer1_completeness>
    <aggregate_sentiment>{agg_sentiment}</aggregate_sentiment>
    <sentiment_breakdown>{sentiment_breakdown}</sentiment_breakdown>
    <avg_reliability>{avg_reliability}</avg_reliability>
    <top_geo_tags>{geo_tags}</top_geo_tags>
    <total_signals>{total_signals}</total_signals>
  </metadata>
 
{news_section}
 
{social_section}
 
{stock_section}
 
{port_section}
 
{weather_section}
 
{commodity_section}
 
</intelligence_bundle>
 
Based on this intelligence bundle, identify the TOP 5 supply chain risks and 3-5 soft/emerging risks.
Return ONLY the JSON object as specified.
"""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PROMPT ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
 
def _sentiment_weight(sig: dict) -> float:
    """Compute importance weight = reliability × |negative sentiment score|."""
    reliability = sig.get("reliability", 0.65)
    sent = sig.get("sentiment") or {}
    neg  = sent.get("negative", 0.0)
    pos  = sent.get("positive", 0.0)
    # Negative sentiment drives risk priority; positive still matters
    sentiment_signal = max(neg, pos * 0.5)
    return reliability * (0.3 + sentiment_signal)
 
 
def _top_n(signals: list[dict], n: int = TOP_N_PER_CATEGORY) -> list[dict]:
    """Return top-N signals sorted by importance weight."""
    return sorted(signals, key=_sentiment_weight, reverse=True)[:n]
 
 
def _format_news_section(news: list[dict]) -> str:
    if not news:
        return "  <news>NO_DATA</news>"
    lines = ["  <news>"]
    for i, item in enumerate(_top_n(news), 1):
        sent  = (item.get("sentiment") or {}).get("label", "neutral")
        score = (item.get("sentiment") or {}).get("score", 0.0)
        geo   = ", ".join(item.get("geo_tags", [])[:3]) or "—"
        lines.append(f"    <item id='{i}'>")
        lines.append(f"      <title>{item.get('title', '')[:120]}</title>")
        lines.append(f"      <body>{item.get('body', '')[:200]}</body>")
        lines.append(f"      <source>{item.get('source', '')}</source>")
        lines.append(f"      <sentiment>{sent} ({score:.2f})</sentiment>")
        lines.append(f"      <geo>{geo}</geo>")
        lines.append(f"      <reliability>{item.get('reliability', 0.65):.2f}</reliability>")
        lines.append(f"    </item>")
    lines.append("  </news>")
    return "\n".join(lines)
 
 
def _format_social_section(social: list[dict]) -> str:
    if not social:
        return "  <social>NO_DATA</social>"
    lines = ["  <social>"]
    for i, item in enumerate(_top_n(social), 1):
        sent  = (item.get("sentiment") or {}).get("label", "neutral")
        score = (item.get("sentiment") or {}).get("score", 0.0)
        lines.append(f"    <post id='{i}'>")
        lines.append(f"      <text>{item.get('text', '')[:200]}</text>")
        lines.append(f"      <source>{item.get('source', '')} / r/{item.get('subreddit', '')}</source>")
        lines.append(f"      <sentiment>{sent} ({score:.2f})</sentiment>")
        lines.append(f"    </post>")
    lines.append("  </social>")
    return "\n".join(lines)
 
 
def _format_stock_section(stocks: list[dict]) -> str:
    if not stocks:
        return "  <stocks>NO_DATA</stocks>"
    lines = ["  <stocks>"]
    for item in _top_n(stocks, n=6):
        sent  = (item.get("sentiment") or {}).get("label", "neutral")
        chg   = item.get("change_pct", 0.0)
        vol   = item.get("volatility_30d")
        vol_s = f"{vol:.4f}" if vol else "N/A"
        lines.append(
            f"    <stock ticker='{item.get('ticker')}' "
            f"price='{item.get('price', 0):.2f}' "
            f"change_pct='{chg:+.2f}' "
            f"volatility_30d='{vol_s}' "
            f"sentiment='{sent}' "
            f"reliability='{item.get('reliability', 0.9):.2f}' />"
        )
    lines.append("  </stocks>")
    return "\n".join(lines)
 
 
def _format_port_section(ports: list[dict]) -> str:
    if not ports:
        return "  <ports>NO_DATA</ports>"
    lines = ["  <ports>"]
    for item in _top_n(ports):
        sent  = (item.get("sentiment") or {}).get("label", "neutral")
        geo   = ", ".join(item.get("geo_tags", [])[:3]) or "—"
        lines.append(
            f"    <port name='{item.get('port_name')}' "
            f"country='{item.get('country')}' "
            f"congestion='{item.get('congestion_flag')}' "
            f"commodity='{item.get('commodity', 'N/A')}' "
            f"trade_value_usd='{item.get('trade_value_usd', 'N/A')}' "
            f"sentiment='{sent}' "
            f"geo='{geo}' />"
        )
    lines.append("  </ports>")
    return "\n".join(lines)
 
 
def _format_weather_section(weather: list[dict]) -> str:
    if not weather:
        return "  <weather>NO_DATA</weather>"
    lines = ["  <weather>"]
    for item in _top_n(weather, n=5):
        sent  = (item.get("sentiment") or {}).get("label", "neutral")
        lines.append(
            f"    <reading city='{item.get('city')}' "
            f"temp_c='{item.get('temp_celsius', 0):.1f}' "
            f"desc='{item.get('description', '')}' "
            f"wind_mps='{item.get('wind_speed', 0):.1f}' "
            f"disruption='{item.get('disruption_flag')}' "
            f"sentiment='{sent}' />"
        )
    lines.append("  </weather>")
    return "\n".join(lines)
 
 
def _format_commodity_section(commodities: list[dict]) -> str:
    if not commodities:
        return "  <commodities>NO_DATA</commodities>"
    lines = ["  <commodities>"]
    for item in _top_n(commodities, n=6):
        sent  = (item.get("sentiment") or {}).get("label", "neutral")
        lines.append(
            f"    <commodity name='{item.get('commodity')}' "
            f"price='{item.get('price', 0):.2f}' "
            f"currency='{item.get('currency', 'USD')}' "
            f"sentiment='{sent}' "
            f"reliability='{item.get('reliability', 0.85):.2f}' />"
        )
    lines.append("  </commodities>")
    return "\n".join(lines)
 
 
def build_prompt(bundle: dict) -> str:
    """
    Assemble the human-turn prompt from the enriched bundle.
    Stays within MAX_CONTEXT_CHARS token budget.
    """
    meta = bundle
 
    news_section      = _format_news_section(bundle.get("news", []))
    social_section    = _format_social_section(bundle.get("social", []))
    stock_section     = _format_stock_section(bundle.get("stocks", []))
    port_section      = _format_port_section(bundle.get("ports", []))
    weather_section   = _format_weather_section(bundle.get("weather", []))
    commodity_section = _format_commodity_section(bundle.get("commodities", []))
 
    prompt = HUMAN_PROMPT_TEMPLATE.format(
        domain             = bundle.get("domain", "supply_chain"),
        fetched_at         = bundle.get("fetched_at", ""),
        enriched_at        = bundle.get("enriched_at", ""),
        completeness       = f"{bundle.get('layer1_completeness', 0.0) * 100:.0f}%",
        agg_sentiment      = bundle.get("aggregate_sentiment", "neutral"),
        sentiment_breakdown= json.dumps(bundle.get("sentiment_breakdown", {})),
        avg_reliability    = f"{bundle.get('avg_reliability', 0.0):.2f}",
        geo_tags           = ", ".join(bundle.get("top_geo_tags", [])[:10]) or "N/A",
        total_signals      = bundle.get("total_items", 0),
        news_section       = news_section,
        social_section     = social_section,
        stock_section      = stock_section,
        port_section       = port_section,
        weather_section    = weather_section,
        commodity_section  = commodity_section,
    )
 
    # Hard-trim to stay within token budget
    if len(prompt) > MAX_CONTEXT_CHARS:
        prompt = prompt[:MAX_CONTEXT_CHARS] + "\n  ...[truncated for token budget]\n</intelligence_bundle>"
 
    return prompt
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GROQ INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
 
def call_groq(prompt: str, domain: str) -> dict:
    """
    Call Groq API with primary model; fall back to smaller model on quota error.
    Returns parsed JSON dict from LLaMA response.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            print(f"    Calling Groq [{model}] …")
            response = client.chat.completions.create(
                model    = model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature      = 0.2,
                max_tokens       = 2048,
                response_format  = {"type": "json_object"},
            )
            raw_text = response.choices[0].message.content.strip()
            print(f"    ✓ Response received ({len(raw_text)} chars)")
 
            # Strip any accidental markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
 
            parsed = json.loads(raw_text)
            parsed["_model_used"] = model
            return parsed
 
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "quota" in err.lower():
                print(f"    [!] {model} rate limited, trying fallback …")
                continue
            print(f"    [✗] Groq call failed: {err}")
            raise
 
    raise RuntimeError("Both Groq models failed. Check API key and quota.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────
 
def parse_risk_report(raw: dict, bundle_meta: dict) -> RiskReport:
    """Validate and coerce the LLM JSON output into a RiskReport Pydantic model."""
 
    top_risks: list[RiskItem] = []
    for item in raw.get("top_risks", []):
        try:
            top_risks.append(RiskItem(
                rank                 = int(item.get("rank", 0)),
                category             = item.get("category", "Unknown"),
                title                = item.get("title", ""),
                severity             = item.get("severity", "MEDIUM").upper(),
                confidence           = float(item.get("confidence", 0.5)),
                probability_next_30d = float(item.get("probability_next_30d", 0.5)),
                evidence             = item.get("evidence", []),
                affected_entities    = item.get("affected_entities", []),
                affected_geo         = item.get("affected_geo", []),
                causal_chain         = item.get("causal_chain", ""),
                recommended_action   = item.get("recommended_action", ""),
            ))
        except Exception as e:
            print(f"    [!] Skipping malformed risk item: {e}")
 
    soft_risks: list[SoftRisk] = []
    for item in raw.get("soft_risks", []):
        try:
            soft_risks.append(SoftRisk(
                category = item.get("category", "Unknown"),
                title    = item.get("title", ""),
                note     = item.get("note", ""),
            ))
        except Exception as e:
            print(f"    [!] Skipping malformed soft risk: {e}")
 
    return RiskReport(
        analysed_at          = raw.get("analysed_at", datetime.utcnow().isoformat()),
        domain               = raw.get("domain", bundle_meta.get("domain", "supply_chain")),
        top_risks            = top_risks,
        soft_risks           = soft_risks,
        data_quality_note    = raw.get("data_quality_note", ""),
        model_used           = raw.get("_model_used", PRIMARY_MODEL),
        layer1_completeness  = bundle_meta.get("layer1_completeness", 0.0),
        layer2_enriched_at   = bundle_meta.get("enriched_at", ""),
        aggregate_sentiment  = bundle_meta.get("aggregate_sentiment", "neutral"),
        avg_reliability      = bundle_meta.get("avg_reliability", 0.0),
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────
 
SEVERITY_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
 
def _print_summary(report: RiskReport) -> None:
    print("\n" + "=" * 65)
    print("  RISK REPORT SUMMARY")
    print("=" * 65)
    print(f"  Analysed at      : {report.analysed_at}")
    print(f"  Domain           : {report.domain}")
    print(f"  Model used       : {report.model_used}")
    print(f"  Data quality     : {report.data_quality_note[:80]}")
    print(f"  Layer 1 complete : {report.layer1_completeness * 100:.0f}%")
    print(f"  Avg reliability  : {report.avg_reliability:.2f}")
    print()
 
    print("  ── TOP 5 RISKS ──────────────────────────────────────────")
    for risk in report.top_risks:
        icon = SEVERITY_ICON.get(risk.severity, "⚪")
        print(f"\n  #{risk.rank}  {icon} [{risk.severity}]  {risk.title}")
        print(f"      Category   : {risk.category}")
        print(f"      Confidence : {risk.confidence:.0%}  |  P(30d): {risk.probability_next_30d:.0%}")
        print(f"      Cause→Effect: {risk.causal_chain[:100]}")
        print(f"      Action     : {risk.recommended_action[:100]}")
        if risk.affected_geo:
            print(f"      Geo        : {', '.join(risk.affected_geo[:5])}")
        if risk.evidence:
            print(f"      Evidence   : {risk.evidence[0][:100]}")
 
    if report.soft_risks:
        print("\n  ── SOFT / EMERGING RISKS ────────────────────────────────")
        for sr in report.soft_risks:
            print(f"    · [{sr.category}]  {sr.title}")
            print(f"        {sr.note[:100]}")
 
    print("\n" + "=" * 65)
    print("[→] Ready for Layer 4 (Counterfactual Simulation)\n")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
 
def run_layer3(
    input_path:  str = str(DEFAULT_INPUT_PATH),
    output_path: str = str(DEFAULT_OUTPUT_PATH),
) -> RiskReport:
    """
    Full Layer 3 pipeline.
    Reads EnrichedRiskInputBundle → Groq LLaMA inference → writes RiskReport.
    """
 
    print("\n" + "=" * 65)
    print("  LAYER 3 — LLM RISK ANALYSIS  (Groq · LLaMA-3 70B)")
    print("=" * 65)
 
    # ── 1. Load Layer 2 output ────────────────────────────────────────────────
    print(f"\n[1/4] Loading EnrichedRiskInputBundle from '{input_path}' …")
    with open(input_path, "r") as f:
        bundle = json.load(f)
 
    total = bundle.get("total_items", 0)
    print(f"    Domain          : {bundle.get('domain', '?')}")
    print(f"    Total signals   : {total}")
    print(f"    Agg sentiment   : {bundle.get('aggregate_sentiment', '?')}")
    print(f"    Avg reliability : {bundle.get('avg_reliability', 0.0):.2f}")
    print(f"    Geo tags        : {', '.join(bundle.get('top_geo_tags', [])[:5]) or 'none'}")
 
    # ── 2. Build prompt ───────────────────────────────────────────────────────
    print("\n[2/4] Building prompt (Prompt Orchestrator) …")
    prompt = build_prompt(bundle)
    print(f"    Prompt length   : {len(prompt):,} chars")
 
    # ── 3. Groq inference ─────────────────────────────────────────────────────
    print("\n[3/4] Running Groq inference …")
    raw_output = call_groq(prompt, bundle.get("domain", "supply_chain"))
 
    # ── 4. Parse + persist ────────────────────────────────────────────────────
    print("\n[4/4] Parsing risk report and writing to JSON …")
    report = parse_risk_report(raw_output, bundle)
 
    with open(output_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
 
    print(f"    ✓ Risk report saved to '{output_path}'")
    print(f"    ✓ Top risks identified: {len(report.top_risks)}")
    print(f"    ✓ Soft risks identified: {len(report.soft_risks)}")
 
    _print_summary(report)
    return report
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 3 — LLM Risk Analysis")
    parser.add_argument("--input",  default=str(DEFAULT_INPUT_PATH),  help="Layer 2 output JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Layer 3 output JSON")
    args = parser.parse_args()
 
    run_layer3(input_path=args.input, output_path=args.output)
 