"""
Main Data Collection Orchestrator
Coordinates collection from all sources, normalization, and storage.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add collectors to path
from collectors.news_collector import collect_news
from collectors.financial_collector import collect_financial
from collectors.weather_collector import collect_weather
from collectors.social_collector import collect_social
from normalizer import DataNormalizer
from storage import DataStorage


# Setup logging
def setup_logging(log_file: str = "data_collection.log"):
    """
    Configure logging to file and console.
    
    Args:
        log_file: Name of log file.
    """
    log_format = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def load_config(config_path: str = "config.json") -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json file.
    
    Returns:
        Configuration dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in config file: {config_path}")
        sys.exit(1)


def collect_data(config: dict) -> dict:
    """
    Orchestrate data collection from all sources.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        Dictionary with collected data by source.
    """
    logging.info("=" * 60)
    logging.info("STARTING DATA COLLECTION")
    logging.info("=" * 60)
    
    collected_data = {
        "news": [],
        "financial": [],
        "weather": [],
        "social": []
    }
    
    # Collect from News
    try:
        if config.get("sources", {}).get("news", {}).get("enabled"):
            logging.info("\n>>> Collecting NEWS data...")
            collected_data["news"] = collect_news(config["sources"]["news"])
        else:
            logging.info("News collection disabled in config")
    except Exception as e:
        logging.error(f"News collection failed: {e}")
    
    # Collect from Financial sources
    try:
        if config.get("sources", {}).get("financial", {}).get("enabled"):
            logging.info("\n>>> Collecting FINANCIAL data...")
            collected_data["financial"] = collect_financial(config["sources"]["financial"])
        else:
            logging.info("Financial collection disabled in config")
    except Exception as e:
        logging.error(f"Financial collection failed: {e}")
    
    # Collect from Weather
    try:
        if config.get("sources", {}).get("weather", {}).get("enabled"):
            logging.info("\n>>> Collecting WEATHER data...")
            collected_data["weather"] = collect_weather(config["sources"]["weather"])
        else:
            logging.info("Weather collection disabled in config")
    except Exception as e:
        logging.error(f"Weather collection failed: {e}")
    
    # Collect from Social
    try:
        if config.get("sources", {}).get("social", {}).get("enabled"):
            logging.info("\n>>> Collecting SOCIAL data...")
            collected_data["social"] = collect_social(config["sources"]["social"])
        else:
            logging.info("Social collection disabled in config")
    except Exception as e:
        logging.error(f"Social collection failed: {e}")
    
    return collected_data


def main():
    """
    Main orchestration function.
    """
    # Setup logging
    setup_logging()
    
    logging.info(f"Data Collection Started at {datetime.now().isoformat()}")
    
    # Load configuration
    config = load_config(str(Path(__file__).resolve().parent / "config.json"))
    
    # Collect data from all sources
    collected_data = collect_data(config)
    
    # Normalize data
    logging.info("\n" + "=" * 60)
    logging.info("NORMALIZING DATA")
    logging.info("=" * 60)
    
    normalized_data = DataNormalizer.normalize_all(
        collected_data["news"],
        collected_data["financial"],
        collected_data["weather"],
        collected_data["social"]
    )
    
    # Store data
    logging.info("\n" + "=" * 60)
    logging.info("STORING DATA")
    logging.info("=" * 60)
    
    configured_output_dir = config.get("output", {}).get("output_dir", "../data/")
    output_dir_path = Path(configured_output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = (Path(__file__).resolve().parent / output_dir_path).resolve()

    storage = DataStorage(output_dir=str(output_dir_path))
    
    # Verify data quality before saving
    stats = DataStorage.verify_data(normalized_data)
    
    # Save in all configured formats
    formats = config.get("output", {}).get("formats", ["json", "csv"])
    saved_files = storage.save_all(normalized_data, formats=formats)
    
    # Print summary
    logging.info("\n" + "=" * 60)
    logging.info("DATA COLLECTION COMPLETE")
    logging.info("=" * 60)
    
    logging.info(f"\nSummary:")
    logging.info(f"  Total Records Collected: {stats['total_records']}")
    logging.info(f"  Records by Source:")
    for source, count in stats["sources"].items():
        logging.info(f"    - {source}: {count}")
    logging.info(f"  Date Range: {stats['date_range']['min']} to {stats['date_range']['max']}")
    logging.info(f"  Records with Empty Content: {stats['empty_content']}")
    
    logging.info(f"\nFiles Saved:")
    for format_type, file_path in saved_files.items():
        logging.info(f"  - {format_type}: {file_path}")
    
    logging.info(f"\nCollection finished at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
