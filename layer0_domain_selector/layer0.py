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

try:
    from domain_profile import DomainProfile
except ImportError:
    from .domain_profile import DomainProfile


ROOT_DIR = Path(__file__).resolve().parent.parent
LAYER1_DIR = ROOT_DIR / "layer1_data_collection"
BASE_CONFIG_PATH = LAYER1_DIR / "config.json"
ENV_PATH = ROOT_DIR / ".env"
GEMINI_MODEL = "gemini-2.5-flash-lite"


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "custom_domain"


def _prompt(message: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or default


def _prompt_list(message: str, default: List[str]) -> List[str]:
    joined = ", ".join(default)
    value = input(f"{message} (comma-separated) [{joined}]: ").strip()
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_base_config() -> Dict[str, Any]:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _fallback_keywords(domain: str, description: str) -> list[str]:
    text = f"{domain} {description}".lower()
    if any(token in text for token in ["supply", "logistics", "shipping", "procurement", "inventory"]):
        return ["supply chain disruption", "supplier risk", "logistics delay", "inventory shortage", "port congestion"]
    if any(token in text for token in ["health", "hospital", "clinical", "care", "patient"]):
        return ["healthcare risk", "patient safety", "clinical staffing", "hospital supply", "regulatory compliance"]
    if any(token in text for token in ["job", "employment", "workforce", "ai", "automation"]):
        return ["ai job displacement", "automation layoffs", "workforce transition", "reskilling", "hiring freeze"]
    return [f"{domain} risk", f"{domain} disruption", f"{domain} trend", f"{domain} outlook", f"{domain} impact"]


def _optional_gemini_suggestions(domain: str, description: str) -> dict[str, list[str]]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {}

    import requests

    prompt = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Suggest concise configuration lists for a risk-analysis pipeline. "
                            "Return JSON with keys: news_keywords, job_keywords, social_keywords, social_hashtags, "
                            "financial_tickers, fred_indicators. Each value must be an array of strings. "
                            f"Domain: {domain}. Description: {description}."
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
    response = requests.post(url, json=prompt, timeout=30)
    response.raise_for_status()
    payload = response.json()
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def build_domain_config(
    domain: str,
    description: str,
    keywords: list[str],
    use_gemini: bool = False,
) -> Dict[str, Any]:
    config = deepcopy(_load_base_config())
    slug = _slugify(domain)

    suggestions: Dict[str, Any] = {}
    if use_gemini:
        try:
            suggestions = _optional_gemini_suggestions(domain, description)
        except Exception:
            suggestions = {}

    news_keywords = suggestions.get("news_keywords") or keywords or _fallback_keywords(domain, description)
    job_keywords = suggestions.get("job_keywords") or keywords or _fallback_keywords(domain, description)
    social_keywords = suggestions.get("social_keywords") or news_keywords
    social_hashtags = suggestions.get("social_hashtags") or [re.sub(r"[^a-z0-9]+", "", word.lower()) for word in keywords[:4]]
    financial_tickers = suggestions.get("financial_tickers") or config.get("sources", {}).get("financial", {}).get("alpha_vantage", {}).get("tickers", [])
    fred_indicators = suggestions.get("fred_indicators") or config.get("sources", {}).get("financial", {}).get("fred", {}).get("indicators", [])

    config["output"]["domain"] = slug
    config["logging"]["log_file"] = f"{slug}_data_collection.log"

    config["sources"]["news"]["keywords"] = news_keywords
    config["sources"]["news"]["methods"]["gnews"]["note"] = f"GNews for {domain} domain"
    config["sources"]["news"]["methods"]["rss"]["note"] = f"RSS feeds for {domain} domain"

    config["sources"]["jobs"]["keywords"] = job_keywords
    config["sources"]["jobs"]["usajobs"]["keywords"] = job_keywords
    config["sources"]["jobs"]["usajobs"]["note"] = f"USAJOBS signals for {domain} domain"

    config["sources"]["social"]["pushshift"]["keywords"] = social_keywords
    config["sources"]["social"]["youtube"]["keywords"] = social_keywords
    config["sources"]["social"]["mastodon"]["keywords"] = social_keywords
    config["sources"]["social"]["mastodon"]["hashtags"] = social_hashtags
    config["sources"]["social"]["hackernews"]["keywords"] = social_keywords

    if financial_tickers:
        config["sources"]["financial"]["alpha_vantage"]["tickers"] = financial_tickers
    if fred_indicators:
        config["sources"]["financial"]["fred"]["indicators"] = fred_indicators

    config["domain_profile"] = {
        "name": domain,
        "slug": slug,
        "description": description,
        "keywords": keywords,
        "news_keywords": news_keywords,
        "job_keywords": job_keywords,
        "social_keywords": social_keywords,
    }
    return config


def write_config(config: Dict[str, Any]) -> None:
    with open(BASE_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def run_pipeline() -> None:
    subprocess.run([sys.executable, str(LAYER1_DIR / "collect_data.py")], cwd=str(LAYER1_DIR), check=True)
    subprocess.run([sys.executable, str(ROOT_DIR / "layer2_nlp" / "layer2_nlp.py")], cwd=str(ROOT_DIR), check=True)
    subprocess.run([sys.executable, str(ROOT_DIR / "layer3_llm" / "layer3_llm_analysis.py")], cwd=str(ROOT_DIR), check=True)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer 0 - Dynamic domain selector")
    parser.add_argument("--domain", help="Risk domain name to analyze")
    parser.add_argument("--description", help="Short description of the domain")
    parser.add_argument("--keywords", help="Comma-separated core keywords for the domain")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini to suggest config values")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini suggestions even if available")
    return parser.parse_args()


def main() -> None:
    load_dotenv(ENV_PATH)
    args = _parse_cli_args()

    print("Layer 0 - Domain setup")
    domain = args.domain or _prompt("Enter domain name", "Supply Chain")
    description = args.description or _prompt("Enter domain description", "Risk signals for the selected domain")
    default_keywords = _fallback_keywords(domain, description)
    if args.keywords:
        keywords = [item.strip() for item in args.keywords.split(",") if item.strip()]
        if not keywords:
            keywords = default_keywords
    else:
        keywords = _prompt_list("Enter core keywords", default_keywords)

    gemini_available = bool(os.getenv("GEMINI_API_KEY", "").strip())
    use_gemini = False
    if args.no_gemini:
        use_gemini = False
    elif args.use_gemini:
        use_gemini = gemini_available
    elif gemini_available:
        use_gemini = _prompt("Use Gemini to suggest additional config values?", "y").lower().startswith("y")

    config = build_domain_config(domain, description, keywords, use_gemini=use_gemini)
    write_config(config)

    profile = DomainProfile(
        domain=domain,
        description=description,
        keywords=keywords,
        config_path=BASE_CONFIG_PATH,
        config=config,
    )

    print(f"Updated Layer 1 config for domain: {profile.slug}")
    print(f"Config written to: {profile.config_path}")
    if use_gemini:
        print(f"Gemini suggestions were applied to the config using {GEMINI_MODEL}.")
    print("Starting Layer 1 -> Layer 3 pipeline...")
    run_pipeline()


if __name__ == "__main__":
    main()
