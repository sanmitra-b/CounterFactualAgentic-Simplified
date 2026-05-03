from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class DomainProfile:
    domain: str
    description: str
    keywords: list[str]
    config_path: Path
    config: Dict[str, Any]

    @property
    def slug(self) -> str:
        return self.domain.strip().lower().replace(" ", "_")

# Try to import the Layer 1 validator (now located under layer0_domain_selector)
try:
    from .config_validator import (
        validate_config as _validate_config,
        ConfigValidationError as _ConfigValidationError,
    )
except Exception:
    # Fallback to importing from the sibling package if running from project root
    try:
        from layer0_domain_selector.config_validator import (
            validate_config as _validate_config,
            ConfigValidationError as _ConfigValidationError,
        )
    except Exception:
        _validate_config = None
        _ConfigValidationError = RuntimeError


ROOT_DIR = Path(__file__).resolve().parent.parent
LAYER1_DIR = ROOT_DIR / "layer1_data_collection"
BASE_CONFIG_PATH = LAYER1_DIR / "config.json"
ENV_PATH = ROOT_DIR / ".env"
GEMINI_MODEL = "gemini-2.5-flash"

# Default FRED series IDs used when extracted indicators are not valid series codes.
DEFAULT_FRED_SERIES_IDS = ["CPIAUCSL", "UNRATE", "PPIACO", "INDPRO"]


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "custom_domain"


def _load_base_config() -> Dict[str, Any]:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_fred_indicators(indicators: Any) -> list[str]:
    """Return valid-looking FRED series IDs, or domain-agnostic defaults.

    FRED `series_id` values are compact identifiers (e.g., CPIAUCSL), not
    natural-language phrases like "supply chain costs".
    """
    if not isinstance(indicators, list):
        return DEFAULT_FRED_SERIES_IDS.copy()

    cleaned: list[str] = []
    for item in indicators:
        code = str(item).strip().upper()
        if re.fullmatch(r"[A-Z0-9._-]{2,20}", code):
            cleaned.append(code)

    # Keep order, remove duplicates
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned if cleaned else DEFAULT_FRED_SERIES_IDS.copy()


def _looks_relevant(source_name: str, text: str) -> bool:
    text = text.lower()
    if source_name == "news":
        return True
    if source_name == "financial":
        return any(keyword in text for keyword in ["market", "price", "cost", "economic", "inflation", "demand", "supply", "business", "finance", "revenue"])
    if source_name == "weather":
        return any(keyword in text for keyword in ["weather", "temperature", "cold chain", "cold", "heat", "climate", "logistics", "transport", "shipment", "storage", "refrigeration"])
    if source_name == "social":
        return any(keyword in text for keyword in ["social", "sentiment", "public", "discussion", "opinion", "community", "trend", "news coverage"])
    if source_name == "jobs":
        return any(keyword in text for keyword in ["job", "employment", "hiring", "labor", "workforce", "vacancy", "career"])
    return False


def _prune_irrelevant_sources(profile: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    source_usage = profile.get("source_usage", {})
    if not isinstance(source_usage, dict):
        return profile

    for source_name in ["jobs", "social", "financial", "weather"]:
        source = source_usage.get(source_name)
        if not isinstance(source, dict):
            continue
        if source.get("enabled") and not _looks_relevant(source_name, user_input + " " + str(profile.get("domain", "")) + " " + " ".join(map(str, profile.get("keywords", [])))):
            source["enabled"] = False
            methods = source.get("methods", {})
            if isinstance(methods, dict):
                for method_name in list(methods.keys()):
                    methods[method_name] = False
            source["methods"] = methods

    return profile


def _parse_gemini_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _refine_profile_with_gemini(profile: Dict[str, Any], error_message: str) -> Dict[str, Any]:
    """Ask Gemini to adjust the extracted profile to satisfy the validator error.

    Returns a revised profile dict or raises RuntimeError on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to refine the domain profile")

    import requests

    instruction = (
        "The following domain profile JSON failed validation for Layer 1.\n"
        "Validation error: "
        + error_message
        + "\n\nPlease return a corrected domain profile JSON only, with keys: domain, description, keywords, source_usage. "
        "Preserve as much of the original intent as possible, but adjust structure/types so the Layer 1 validator accepts it. "
        "For source_usage.financial.indicators, return valid FRED series IDs only (e.g., CPIAUCSL, UNRATE, PPIACO, INDPRO), "
        "not natural-language phrases. "
        "Do not include any additional text or commentary.\n\nOriginal profile:\n"
        + json.dumps(profile, ensure_ascii=False)
    )

    prompt = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": instruction}],
            }
        ],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    try:
        resp = requests.post(url, json=prompt, timeout=30)
        if resp.status_code == 429:
            raise RuntimeError(f"Gemini rate-limited (429): {resp.text[:300]}")
        resp.raise_for_status()
        payload = resp.json()
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        new_profile = _parse_gemini_json(text)
        if not isinstance(new_profile, dict):
            raise RuntimeError("Gemini returned invalid profile during refinement")
        return new_profile
    except Exception as exc:
        raise RuntimeError(f"Failed to refine profile via Gemini: {exc}")


def _ensure_minimum_sources_enabled(profile: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """Ensure at least 4-5 sources are enabled in the profile.
    
    If fewer than 4 are enabled, enable additional sources based on heuristics.
    """
    source_usage = profile.get("source_usage", {})
    if not isinstance(source_usage, dict):
        source_usage = {}
    
    # Count currently enabled sources (exclude 'jobs' from auto-enablement)
    enabled_sources = []
    for source_name in ["news", "financial", "social", "weather"]:
        source = source_usage.get(source_name, {})
        if isinstance(source, dict):
            if source.get("enabled"):
                enabled_sources.append(source_name)
            else:
                # Check if any method is enabled
                methods = source.get("methods", {})
                if any(bool(m) for m in methods.values() if isinstance(m, bool)):
                    enabled_sources.append(source_name)
    
    # If fewer than 4 sources are enabled, enable more based on heuristics.
    if len(enabled_sources) < 4:
        keywords = profile.get("keywords", [])
        keywords_lower = " ".join(str(k).lower() for k in keywords)
        
        # Define source relevance heuristics
        relevance_scores = {
            "news": 2.0,  # News is almost always relevant
            "weather": 1.8,  # Weather often relevant for logistics/environment
            "social": 1.5,  # Social sentiment valuable for risk context
            "financial": 1.3,  # Financial data for market/impact analysis
        }
        
        # Boost scores based on keywords
        if any(kw in keywords_lower for kw in ["supply", "logistic", "transport", "weather", "temp", "environ"]):
            relevance_scores["weather"] = 2.2
            relevance_scores["news"] = 2.1
        if any(kw in keywords_lower for kw in ["market", "price", "stock", "financial", "economic"]):
            relevance_scores["financial"] = 2.0
            relevance_scores["news"] = 2.1
        # Don't auto-enable 'jobs' based on keywords; keep other boosts
        if any(kw in keywords_lower for kw in ["job", "employment", "hiring", "labor", "workforce"]):
            relevance_scores["social"] = 1.8
        if any(kw in keywords_lower for kw in ["sentiment", "opinion", "public", "trend", "social", "community"]):
            relevance_scores["social"] = 2.2
        
        # Sort by relevance and enable until we have the minimum (4)
        available_sources = [s for s in ["news", "financial", "social", "weather"] if s not in enabled_sources]
        available_sources.sort(key=lambda s: relevance_scores.get(s, 1.0), reverse=True)
        
        for source_name in available_sources:
            if len(enabled_sources) >= 4:
                break
            
            # Ensure source exists in source_usage
            if source_name not in source_usage:
                source_usage[source_name] = {"enabled": False, "methods": {}}
            
            source = source_usage[source_name]
            if not isinstance(source, dict):
                source = {"enabled": False, "methods": {}}
                source_usage[source_name] = source
            
            # Enable the source and at least one method
            source["enabled"] = True
            methods = source.get("methods", {})
            if not isinstance(methods, dict):
                methods = {}
            
            if source_name == "news":
                methods = {"newsapi": True, "gnews": False, "rss": True}
                if "keywords" not in source or not source.get("keywords"):
                    source["keywords"] = profile.get("keywords", [])
            elif source_name == "financial":
                methods = {"alpha_vantage": True, "fred": True}
                if "tickers" not in source:
                    source["tickers"] = []
                if "indicators" not in source:
                    source["indicators"] = []
            elif source_name == "social":
                methods = {"pushshift": False, "youtube": True, "mastodon": True, "hackernews": True}  # pushshift disabled permanently
                if "keywords" not in source or not source.get("keywords"):
                    source["keywords"] = profile.get("keywords", [])
                if "hashtags" not in source:
                    source["hashtags"] = []
            elif source_name == "weather":
                methods = {"openweather": True}
                if "keywords" not in source:
                    source["keywords"] = profile.get("keywords", [])
                if "cities" not in source or not source.get("cities"):
                    source["cities"] = ["New York", "London", "Tokyo", "Singapore"]
            
            source["methods"] = methods
            enabled_sources.append(source_name)
        
        profile["source_usage"] = source_usage
    
    return profile


def _auto_extract_domain_profile(user_input: str) -> Dict[str, Any]:
    """Extract a canonical domain profile from a natural-language user prompt using Gemini."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to extract the domain profile")

    import requests

    prompt = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Extract a canonical risk-analysis domain profile from the user's text. "
                            "The user text may be a question, sentence, topic, or domain label. "
                            "Return JSON only with these keys: domain, description, keywords, source_usage. "
                            "domain must be a short canonical label, not the raw question. "
                            "description must be 1-2 concise sentences. "
                            "keywords must be 5-8 concise search terms. "
                            "source_usage must contain these top-level keys: news, financial, jobs, social, weather. "
                            "Each source object must contain enabled as a boolean. "
                            "For news, financial, jobs, and social, also include a methods object with booleans for: "
                            "news => newsapi, gnews, rss; financial => alpha_vantage, fred; jobs => adzuna, usajobs; "
                            "social => pushshift, youtube, mastodon, hackernews; weather => openweather. "
                            "For financial, also include tickers and indicators arrays when relevant. "
                            "For news, jobs, social, and weather, also include keywords arrays when relevant. "
                            "For weather, also include cities or locations arrays when relevant. "
                            "For social, also include hashtags when relevant. "
                            "IMPORTANT for financial.indicators: return valid FRED series IDs only (e.g., CPIAUCSL, UNRATE, PPIACO, INDPRO). "
                            "Do NOT return natural-language indicator names like 'supply chain costs'. "
                            "IMPORTANT: Enable only sources that are clearly relevant to the user text. "
                            "Prefer a small, focused set of 2-3 sources over broad coverage. "
                            "Disable jobs unless labor, hiring, employment, workforce, or vacancy risk is explicit. "
                            "Disable social unless public sentiment, discussion, or social chatter is explicit. "
                            "Disable financial unless market, price, cost, inflation, business, or economic impact is explicit. "
                            "Disable weather unless temperature, climate, storage, shipment, logistics, or environmental risk is explicit. "
                            "Do not enable sources just to increase the count. Quality and relevance are more important than breadth. "
                            "User text: " + user_input
                        )
                    }
                ],
            }
        ],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    try:
        response = requests.post(url, json=prompt, timeout=30)
        if response.status_code == 429:
            raise RuntimeError(f"Gemini rate-limited (429): {response.text[:300]}")
        response.raise_for_status()
        payload = response.json()
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        profile = _parse_gemini_json(text)
        if not isinstance(profile, dict):
            raise RuntimeError("Gemini returned an invalid domain profile")

        # Preserve Gemini's original source_usage for debugging
        try:
            original_usage = deepcopy(profile.get("source_usage", {}))
        except Exception:
            original_usage = profile.get("source_usage", {})
        profile["__gemini_source_usage"] = original_usage

        # Prune clearly irrelevant sources before any auto-enablement kicks in.
        profile = _prune_irrelevant_sources(profile, user_input)

        # Enforce minimum sources (relaxed rule)
        profile = _ensure_minimum_sources_enabled(profile, user_input)
        return profile
    except Exception as exc:
        raise RuntimeError(f"Failed to extract a domain profile from: {user_input} — {exc}")


def build_domain_config(
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    config = deepcopy(_load_base_config())
    domain = str(profile.get("domain", "custom_domain")).strip()
    description = str(profile.get("description", "")).strip()
    keywords = profile.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    slug = _slugify(domain)
    source_usage = profile.get("source_usage", {})
    if not isinstance(source_usage, dict):
        source_usage = {}

    def source_cfg(name: str) -> Dict[str, Any]:
        value = source_usage.get(name, {})
        return value if isinstance(value, dict) else {}

    def enabled(source_name: str) -> bool:
        source = source_cfg(source_name)
        if source.get("enabled"):
            return True
        methods = source.get("methods", {})
        return any(bool(flag) for flag in methods.values())

    def method_enabled(source_name: str, method_name: str) -> bool:
        source = source_cfg(source_name)
        if not enabled(source_name):
            return False
        methods = source.get("methods", {})
        return bool(methods.get(method_name, False))

    news_keywords = source_cfg("news").get("keywords") or keywords
    job_keywords = source_cfg("jobs").get("keywords") or keywords
    social_keywords = source_cfg("social").get("keywords") or keywords
    social_hashtags = source_cfg("social").get("hashtags") or []
    financial_tickers = source_cfg("financial").get("tickers") or []
    fred_indicators = _sanitize_fred_indicators(source_cfg("financial").get("indicators") or [])

    config["output"]["domain"] = slug
    config["logging"]["log_file"] = f"{slug}_data_collection.log"

    config["sources"]["news"]["enabled"] = enabled("news")
    config["sources"]["news"]["keywords"] = news_keywords if enabled("news") else []
    config["sources"]["news"]["methods"]["newsapi"]["enabled"] = method_enabled("news", "newsapi")
    config["sources"]["news"]["methods"]["gnews"]["enabled"] = method_enabled("news", "gnews")
    config["sources"]["news"]["methods"]["rss"]["enabled"] = method_enabled("news", "rss")
    config["sources"]["news"]["methods"]["newsapi"]["note"] = f"NewsAPI for {domain} domain" if enabled("news") else "Disabled for this domain"
    config["sources"]["news"]["methods"]["gnews"]["note"] = f"GNews for {domain} domain" if enabled("news") else "Disabled for this domain"
    config["sources"]["news"]["methods"]["rss"]["note"] = f"RSS feeds for {domain} domain" if enabled("news") else "Disabled for this domain"

    config["sources"]["financial"]["enabled"] = enabled("financial")
    config["sources"]["financial"]["alpha_vantage"]["enabled"] = method_enabled("financial", "alpha_vantage")
    config["sources"]["financial"]["fred"]["enabled"] = method_enabled("financial", "fred")
    config["sources"]["financial"]["alpha_vantage"]["tickers"] = financial_tickers if enabled("financial") else []
    config["sources"]["financial"]["fred"]["indicators"] = fred_indicators if enabled("financial") else []
    config["sources"]["financial"]["alpha_vantage"]["note"] = f"Alpha Vantage for {domain} domain" if enabled("financial") else "Disabled for this domain"
    config["sources"]["financial"]["fred"]["note"] = f"FRED indicators for {domain} domain" if enabled("financial") else "Disabled for this domain"

    config["sources"]["jobs"]["enabled"] = enabled("jobs")
    config["sources"]["jobs"]["adzuna"]["enabled"] = method_enabled("jobs", "adzuna")
    config["sources"]["jobs"]["usajobs"]["enabled"] = method_enabled("jobs", "usajobs")
    config["sources"]["jobs"]["keywords"] = job_keywords if enabled("jobs") else []
    config["sources"]["jobs"]["adzuna"]["keywords"] = job_keywords if enabled("jobs") else []
    config["sources"]["jobs"]["usajobs"]["keywords"] = job_keywords if enabled("jobs") else []
    config["sources"]["jobs"]["adzuna"]["note"] = f"Adzuna signals for {domain} domain" if enabled("jobs") else "Disabled for this domain"
    config["sources"]["jobs"]["usajobs"]["note"] = f"USAJOBS signals for {domain} domain" if enabled("jobs") else "Disabled for this domain"

    config["sources"]["weather"]["enabled"] = enabled("weather")
    config["sources"]["weather"]["openweather"]["enabled"] = method_enabled("weather", "openweather")
    config["sources"]["weather"]["keywords"] = source_cfg("weather").get("keywords", []) if enabled("weather") else []
    config["sources"]["weather"]["cities"] = source_cfg("weather").get("cities", []) if enabled("weather") else []
    config["sources"]["weather"]["openweather"]["note"] = f"OpenWeather signals for {domain} domain" if enabled("weather") else "Disabled for this domain"

    config["sources"]["social"]["enabled"] = enabled("social")
    config["sources"]["social"]["pushshift"]["enabled"] = False  # Disabled permanently
    config["sources"]["social"]["youtube"]["enabled"] = method_enabled("social", "youtube")
    config["sources"]["social"]["mastodon"]["enabled"] = method_enabled("social", "mastodon")
    config["sources"]["social"]["hackernews"]["enabled"] = method_enabled("social", "hackernews")
    config["sources"]["social"]["pushshift"]["keywords"] = social_keywords if enabled("social") else []
    config["sources"]["social"]["youtube"]["keywords"] = social_keywords if enabled("social") else []
    config["sources"]["social"]["mastodon"]["keywords"] = social_keywords if enabled("social") else []
    config["sources"]["social"]["mastodon"]["hashtags"] = social_hashtags if enabled("social") else []
    config["sources"]["social"]["hackernews"]["keywords"] = social_keywords if enabled("social") else []

    config["domain_profile"] = {
        "name": domain,
        "slug": slug,
        "description": description,
        "keywords": keywords,
        "source_usage": source_usage,
    }
    return config


def write_config(config: Dict[str, Any]) -> None:
    with open(BASE_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def run_pipeline() -> None:
    # Layer 1
    subprocess.run([sys.executable, str(LAYER1_DIR / "collect_data.py")], cwd=str(LAYER1_DIR), check=True)
    # Layer 2
    subprocess.run([sys.executable, str(ROOT_DIR / "layer2_nlp" / "layer2_nlp.py")], cwd=str(ROOT_DIR / "layer2_nlp"), check=True)
    # Layer 3
    subprocess.run([sys.executable, str(ROOT_DIR / "layer3_llm" / "layer3_llm_analysis.py")], cwd=str(ROOT_DIR / "layer3_llm"), check=True)
    # Layer 4 (counterfactual)
    subprocess.run([sys.executable, str(ROOT_DIR / "layer4_counterfactual" / "layer4_supervisor.py")], cwd=str(ROOT_DIR / "layer4_counterfactual"), check=True)
    # Layer 5 (RAG)
    subprocess.run([sys.executable, str(ROOT_DIR / "layer5_rag" / "layer5_supervisor.py")], cwd=str(ROOT_DIR / "layer5_rag"), check=True)

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer 0 - Dynamic domain selector")
    parser.add_argument("--query", help="Natural-language risk question, topic, or domain to analyze")
    parser.add_argument("--domain", help="Alias for --query")
    parser.add_argument("--description", help="Override the extracted description")
    parser.add_argument("--keywords", help="Override the extracted keywords as a comma-separated list")
    return parser.parse_args()


def main() -> None:
    load_dotenv(ENV_PATH)
    args = _parse_cli_args()

    print("Layer 0 - Domain setup (CLI-only, natural-language extraction mode)")
    print("-" * 50)
    
    try:
        user_query = args.query or args.domain
        if not user_query:
            print("Error: --query or --domain is required.")
            print("Example: python layer0_domain_selector/layer0.py --query \"What is the risk in cold chain?\"")
            sys.exit(1)
        print(f"\n✓ Input: {user_query}")

        print("\nExtracting domain profile...")
        profile = _auto_extract_domain_profile(user_query)

        if args.description:
            profile["description"] = args.description
        if args.keywords:
            profile["keywords"] = [item.strip() for item in args.keywords.split(",") if item.strip()]

        domain = str(profile.get("domain", user_query)).strip()
        description = str(profile.get("description", "")).strip()
        keywords = profile.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        print(f"✓ Domain: {domain}")
        print(f"✓ Description: {description}")
        print(f"✓ Keywords: {', '.join(keywords)}")
        # Show what Gemini originally reported for source usage (before auto-enabling)
        gemini_usage = profile.get("__gemini_source_usage")
        if isinstance(gemini_usage, dict):
            print("\nSources reported by Gemini:")
            for src in ["news", "financial", "jobs", "social", "weather"]:
                val = gemini_usage.get(src)
                if val is None:
                    print(f"- {src}: (not present)")
                else:
                    enabled = val.get("enabled") if isinstance(val, dict) else bool(val)
                    methods = val.get("methods") if isinstance(val, dict) else None
                    methods_str = ", ".join([k for k, v in (methods or {}).items() if v]) if methods else ""
                    note = f"enabled={enabled}"
                    if methods_str:
                        note += f" methods=[{methods_str}]"
                    print(f"- {src}: {note}")
        print("\nApplying relevant API flags to Layer 1 config...")
        # Build the Layer 1 config from the profile
        config = build_domain_config(profile)

        # If a Layer 1 validator is available, validate before writing.
        if _validate_config:
            try:
                _validate_config(config)
            except _ConfigValidationError as verr:
                print(f"Validator error: {verr}")
                # Attempt one automated refinement via Gemini if possible
                try:
                    print("Attempting automated profile refinement via Gemini...")
                    refined = _refine_profile_with_gemini(profile, str(verr))
                    print("Retrying with refined profile...")
                    profile = refined
                    config = build_domain_config(profile)
                    _validate_config(config)
                    print("Refined profile validated successfully.")
                except Exception as exc:
                    print(f"Automatic refinement failed: {exc}")
                    print("Please fix the profile or provide overrides and try again.")
                    raise RuntimeError(f"Config validation failed: {verr}")

        write_config(config)

        profile = DomainProfile(
            domain=domain,
            description=description,
            keywords=keywords,
            config_path=BASE_CONFIG_PATH,
            config=config,
        )

        print(f"\n✓ Updated Layer 1 config for domain: {profile.slug}")
        print(f"✓ Config written to: {profile.config_path}")
        print("\nStarting Layer 1 -> Layer 3 pipeline...")
        run_pipeline()
    except RuntimeError as error:
        print(f"\nError: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
