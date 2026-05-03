from __future__ import annotations
 
import os
import json
import argparse
import warnings
import logging
import re
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

GENERIC_DOMAIN_TERMS = {
    "risk", "risks", "chain", "supply", "logistics", "transport", "global", "market",
    "public", "health", "industry", "cost", "costs", "operations", "operation",
    "management", "distribution", "system", "systems", "trend", "trends", "alerts",
}
 
RISK_CATEGORIES = []


def _normalize_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", (text or "").lower()) if term}


def _extract_domain_context(bundle: dict) -> dict[str, Any]:
    profile = bundle.get("domain_profile") or {}
    if not isinstance(profile, dict):
        profile = {}

    terms: set[str] = set()
    phrases: list[str] = []

    domain = str(profile.get("name") or bundle.get("domain") or "").strip()
    if domain:
        phrases.append(domain.lower())
        terms.update(_normalize_terms(domain))

    description = str(profile.get("description") or "").strip()
    if description:
        phrases.extend([segment.strip().lower() for segment in re.split(r"[.;]", description) if segment.strip()])
        terms.update(_normalize_terms(description))

    for keyword in profile.get("keywords", []) or []:
        keyword_text = str(keyword).strip().lower()
        if keyword_text:
            phrases.append(keyword_text)
            terms.update(_normalize_terms(keyword_text))

    source_usage = profile.get("source_usage", {})
    if isinstance(source_usage, dict):
        for source in source_usage.values():
            if not isinstance(source, dict):
                continue
            for keyword in source.get("keywords", []) or []:
                keyword_text = str(keyword).strip().lower()
                if keyword_text:
                    phrases.append(keyword_text)
                    terms.update(_normalize_terms(keyword_text))

    return {"name": domain, "terms": terms, "phrases": list(dict.fromkeys(phrases))}


def _signal_text(item: dict) -> str:
    parts = [
        item.get("title", ""),
        item.get("body", ""),
        item.get("text", ""),
        item.get("description", ""),
        item.get("source", ""),
        item.get("company", ""),
        item.get("ticker", ""),
        item.get("port_name", ""),
        item.get("country", ""),
        item.get("city", ""),
        item.get("commodity", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def _domain_relevance_score(item: dict, domain_terms: set[str], domain_phrases: list[str]) -> float:
    if not domain_terms and not domain_phrases:
        return 1.0

    text = _signal_text(item).lower()
    tokens = _normalize_terms(text)

    specific_terms = {term for term in domain_terms if term not in GENERIC_DOMAIN_TERMS}
    specific_hits = len(tokens.intersection(specific_terms))
    generic_hits = len(tokens.intersection(domain_terms)) - specific_hits
    phrase_hits = sum(1 for phrase in domain_phrases if len(phrase) >= 4 and phrase in text)

    score = (specific_hits * 0.45) + (generic_hits * 0.08) + (phrase_hits * 0.6)

    # If only generic terms matched, keep score below inclusion threshold.
    if specific_hits == 0 and phrase_hits == 0 and generic_hits > 0:
        score = min(score, 0.12)

    if score == 0.0:
        generic_terms = {
            "world", "global", "international", "geopolitics", "war", "conflict", "economy",
            "politics", "public", "health", "market", "technology", "news",
        }
        if tokens.intersection(generic_terms):
            score = 0.05

    return min(score, 1.0)


def _domain_adjusted_weight(sig: dict, domain_terms: set[str], domain_phrases: list[str]) -> float:
    relevance = _domain_relevance_score(sig, domain_terms, domain_phrases)
    if relevance <= 0.0:
        return 0.0
    return _sentiment_weight(sig) * (0.35 + relevance)
 
# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
 
SYSTEM_PROMPT = """\
You are a world-class domain risk analyst with expertise in turning structured intelligence
bundles into concise, evidence-backed risk assessments.
 
Your task is to analyse a structured intelligence bundle collected from news feeds,
social media, financial markets, and USAJOBS labor-demand signals.
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
- Prefer risks that match the active domain profile keywords and descriptions.
- Exclude generic global stories unless there is explicit causal evidence connecting them to the active domain.
- If data is sparse, reflect lower confidence scores and note it in data_quality_note.
- Do NOT hallucinate events not present in the bundle.
"""
 
HUMAN_PROMPT_TEMPLATE = """\
<intelligence_bundle>
  <metadata>
    <domain>{domain}</domain>
        <domain_profile>{domain_profile}</domain_profile>
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
 
Based on this intelligence bundle, identify the TOP 5 risks for the active domain and 3-5 soft/emerging risks.
Use concise, domain-appropriate category labels that reflect the observed evidence.
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
 
 
def _top_n(signals: list[dict], n: int = TOP_N_PER_CATEGORY, domain_terms: set[str] | None = None, domain_phrases: list[str] | None = None) -> list[dict]:
    """Return top-N signals sorted by importance weight."""
    domain_terms = domain_terms or set()
    domain_phrases = domain_phrases or []

    relevant = [item for item in signals if _domain_relevance_score(item, domain_terms, domain_phrases) >= 0.15]
    # If we have domain context, do not fall back to unrelated signals.
    if (domain_terms or domain_phrases) and not relevant:
        return []
    candidates = relevant if relevant else signals
    return sorted(candidates, key=lambda item: _domain_adjusted_weight(item, domain_terms, domain_phrases), reverse=True)[:n]
 
 
def _format_news_section(news: list[dict], domain_terms: set[str], domain_phrases: list[str]) -> str:
    if not news:
        return "  <news>NO_DATA</news>"
    lines = ["  <news>"]
    for i, item in enumerate(_top_n(news, domain_terms=domain_terms, domain_phrases=domain_phrases), 1):
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
 
 
def _format_social_section(social: list[dict], domain_terms: set[str], domain_phrases: list[str]) -> str:
    if not social:
        return "  <social>NO_DATA</social>"
    lines = ["  <social>"]
    for i, item in enumerate(_top_n(social, domain_terms=domain_terms, domain_phrases=domain_phrases), 1):
        sent  = (item.get("sentiment") or {}).get("label", "neutral")
        score = (item.get("sentiment") or {}).get("score", 0.0)
        lines.append(f"    <post id='{i}'>")
        lines.append(f"      <text>{item.get('text', '')[:200]}</text>")
        lines.append(f"      <source>{item.get('source', '')} / r/{item.get('subreddit', '')}</source>")
        lines.append(f"      <sentiment>{sent} ({score:.2f})</sentiment>")
        lines.append(f"    </post>")
    lines.append("  </social>")
    return "\n".join(lines)
 
 
def _format_stock_section(stocks: list[dict], domain_terms: set[str], domain_phrases: list[str]) -> str:
    if not stocks:
        return "  <stocks>NO_DATA</stocks>"
    lines = ["  <stocks>"]
    for item in _top_n(stocks, n=6, domain_terms=domain_terms, domain_phrases=domain_phrases):
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
 
 
def _format_port_section(ports: list[dict], domain_terms: set[str], domain_phrases: list[str]) -> str:
    if not ports:
        return "  <ports>NO_DATA</ports>"
    lines = ["  <ports>"]
    for item in _top_n(ports, domain_terms=domain_terms, domain_phrases=domain_phrases):
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
 
 
def _format_weather_section(weather: list[dict], domain_terms: set[str], domain_phrases: list[str]) -> str:
    if not weather:
        return "  <weather>NO_DATA</weather>"
    lines = ["  <weather>"]
    for item in _top_n(weather, n=5, domain_terms=domain_terms, domain_phrases=domain_phrases):
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
 
 
def _format_commodity_section(commodities: list[dict], domain_terms: set[str], domain_phrases: list[str]) -> str:
    if not commodities:
        return "  <commodities>NO_DATA</commodities>"
    lines = ["  <commodities>"]
    for item in _top_n(commodities, n=6, domain_terms=domain_terms, domain_phrases=domain_phrases):
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

    domain_context = _extract_domain_context(bundle)
    domain_terms = domain_context["terms"]
    domain_phrases = domain_context["phrases"]

    news_section      = _format_news_section(bundle.get("news", []), domain_terms, domain_phrases)
    social_section    = _format_social_section(bundle.get("social", []), domain_terms, domain_phrases)
    stock_section     = _format_stock_section(bundle.get("stocks", []), domain_terms, domain_phrases)
    port_section      = _format_port_section(bundle.get("ports", []), domain_terms, domain_phrases)
    weather_section   = _format_weather_section(bundle.get("weather", []), domain_terms, domain_phrases)
    commodity_section = _format_commodity_section(bundle.get("commodities", []), domain_terms, domain_phrases)
 
    prompt = HUMAN_PROMPT_TEMPLATE.format(
        domain             = bundle.get("domain", "ai_job_risk"),
        domain_profile     = json.dumps(bundle.get("domain_profile", {}), ensure_ascii=False),
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
        risk_categories    = ", ".join(RISK_CATEGORIES) if RISK_CATEGORIES else "domain-appropriate categories",
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
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set — add it to .env")
    client = Groq(api_key=api_key)
 
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
        domain               = raw.get("domain", bundle_meta.get("domain", "custom_domain")),
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
    print("[→] Ready for the next layer\n")
 
 
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
    if bundle.get("domain_profile"):
        print(f"    Active profile  : {bundle.get('domain_profile', {}).get('name', '?')}")
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
    raw_output = call_groq(prompt, bundle.get("domain", "custom_domain"))
 
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
 
