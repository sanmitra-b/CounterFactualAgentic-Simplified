"""
Standalone USAJOBS collector for AI-driven job risk monitoring.

Usage:
  python layer1_data_collection/collectors/collect_usajobs_data.py
"""

import json
from datetime import datetime
from pathlib import Path

from weather_collector import WeatherCollector


def main() -> None:
    collectors_dir = Path(__file__).resolve().parent
    layer1_dir = collectors_dir.parent
    config_path = layer1_dir / "config.json"
    output_path = layer1_dir.parent / "data" / "usajobs_job_signals.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    weather_cfg = config.get("sources", {}).get("weather", {})
    collector = WeatherCollector(weather_cfg)
    records = collector.collect_from_usajobs()

    payload = {
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "domain": config.get("output", {}).get("domain", "ai_job_risk"),
        "total_records": len(records),
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"USAJOBS collection complete: {len(records)} records")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()