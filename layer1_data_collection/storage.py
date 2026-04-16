"""
Storage Module
Saves normalized data to JSON and CSV formats.
"""

import json
import logging
import hashlib
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DataStorage:
    """Stores normalized data in multiple formats."""
    
    def __init__(self, output_dir: str = "../data/"):
        """
        Initialize storage with output directory.
        
        Args:
            output_dir: Directory where data will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage initialized. Output dir: {self.output_dir}")
    
    @staticmethod
    def _to_iso_datetime(date_str: str) -> str:
        """Convert YYYY-MM-DD (or empty) into ISO datetime expected by Layer 2."""
        if not date_str:
            return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # If already datetime-like, normalize Z suffix.
        if "T" in date_str:
            return date_str if date_str.endswith("Z") else f"{date_str}Z"

        return f"{date_str}T00:00:00Z"

    @staticmethod
    def _stable_post_id(subreddit: str, text: str, date_str: str) -> str:
        """Build deterministic post id when API id is unavailable."""
        base = f"{subreddit}|{date_str}|{text[:80]}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()[:16]

    def _to_layer2_bundle(self, data: List[Dict]) -> Dict:
        """
        Convert normalized Layer 1 records into Layer 2 RiskInputBundle schema.

        Required by layer2_nlp.py:
          domain, fetched_at, completeness_score,
          news, social, stocks, ports, weather, commodities, errors
        """
        news_items = []
        social_items = []
        stock_items = []
        port_items = []
        weather_items = []
        commodity_items = []

        for record in data:
            source_type = record.get("source_type", "")
            source_name = record.get("source_name", "")
            metadata = record.get("metadata", {}) or {}
            date_val = record.get("date", "")

            if source_type == "news":
                news_items.append(
                    {
                        "title": record.get("title", ""),
                        "body": record.get("content", ""),
                        "source": source_name.lower() if source_name else "news",
                        "url": metadata.get("url", ""),
                        "published_at": self._to_iso_datetime(date_val),
                    }
                )

            elif source_type == "social":
                subreddit = metadata.get("subreddit", "unknown")
                text = record.get("content", "")
                social_items.append(
                    {
                        "text": text,
                        "source": "reddit",
                        "platform": "reddit",
                        "subreddit": subreddit,
                        "post_id": self._stable_post_id(subreddit, text, date_val),
                        "created_at": self._to_iso_datetime(date_val),
                        "score": int(metadata.get("score", 0) or 0),
                        "num_comments": int(metadata.get("num_comments", 0) or 0),
                    }
                )

            elif source_type == "financial":
                data_type = metadata.get("data_type", "")
                if data_type == "stock_price":
                    price = float(metadata.get("price", 0) or 0)
                    open_price = float(metadata.get("open", 0) or 0)
                    change_pct = ((price - open_price) / open_price * 100.0) if open_price else 0.0
                    ticker = metadata.get("ticker", "UNKNOWN")
                    stock_items.append(
                        {
                            "ticker": ticker,
                            "company": ticker,
                            "price": price,
                            "change_pct": round(change_pct, 4),
                            "volume": int(metadata.get("volume", 0) or 0),
                            "market_cap": 0,
                            "currency": "USD",
                            "source": source_name.lower() if source_name else "financial",
                            "fetched_at": self._to_iso_datetime(date_val),
                        }
                    )
                elif data_type == "economic_indicator":
                    indicator = metadata.get("indicator", "UNKNOWN")
                    commodity_items.append(
                        {
                            "commodity": indicator,
                            "price": float(metadata.get("value", 0) or 0),
                            "currency": "USD",
                            "unit": "index",
                            "change_pct": 0.0,
                            "source": source_name.lower() if source_name else "fred",
                            "fetched_at": self._to_iso_datetime(date_val),
                        }
                    )

            elif source_type == "weather":
                wind_speed_ms = float(metadata.get("wind_speed", 0) or 0)
                wind_kmh = round(wind_speed_ms * 3.6, 2)
                weather_main = (metadata.get("weather_main", "") or "").lower()
                disruption_flag = wind_kmh >= 40 or weather_main in {
                    "thunderstorm",
                    "tornado",
                    "squall",
                    "ash",
                    "sand",
                    "dust",
                }
                weather_items.append(
                    {
                        "city": metadata.get("city", ""),
                        "country": metadata.get("country", ""),
                        "description": metadata.get("weather_description", record.get("content", "")),
                        "temperature_c": float(metadata.get("temperature", 0) or 0),
                        "wind_kmh": wind_kmh,
                        "disruption_flag": disruption_flag,
                        "source": "openweather",
                        "fetched_at": self._to_iso_datetime(date_val),
                    }
                )

        active_groups = [
            news_items,
            social_items,
            stock_items,
            weather_items,
        ]
        non_empty = sum(1 for group in active_groups if len(group) > 0)
        completeness_score = round(non_empty / len(active_groups), 2) if active_groups else 0.0

        bundle = {
            "domain": "supply_chain",
            "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "completeness_score": completeness_score,
            "news": news_items,
            "social": social_items,
            "stocks": stock_items,
            "ports": port_items,
            "weather": weather_items,
            "commodities": commodity_items,
            "errors": [],
        }
        return bundle

    def save_json(self, data: List[Dict], filename: str = "risk_input_bundle.json") -> str:
        """
        Save data as Layer 2-compatible RiskInputBundle JSON object.
        
        Args:
            data: List of normalized records.
            filename: Output filename.
        
        Returns:
            Path to saved file.
        """
        file_path = self.output_dir / filename
        
        try:
            bundle = self._to_layer2_bundle(data)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(data)} records to JSON: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            raise
    
    def save_csv(self, data: List[Dict], filename: str = "risk_input_bundle.csv") -> str:
        """
        Save data as CSV for spreadsheet viewing.
        
        Args:
            data: List of normalized records.
            filename: Output filename.
        
        Returns:
            Path to saved file.
        """
        if not data:
            logger.warning("No data to save to CSV")
            return ""
        
        file_path = self.output_dir / filename
        
        try:
            # Flatten nested metadata into separate columns
            flattened = []
            for record in data:
                flat_record = {
                    "source_type": record.get("source_type"),
                    "source_name": record.get("source_name"),
                    "date": record.get("date"),
                    "title": record.get("title", ""),
                    "content": record.get("content", "")[:200]  # Truncate for CSV readability
                }
                
                # Add metadata fields as flat columns
                for key, value in record.get("metadata", {}).items():
                    flat_record[f"meta_{key}"] = str(value)[:100]  # Truncate long values
                
                flattened.append(flat_record)
            
            df = pd.DataFrame(flattened)
            df.to_csv(file_path, index=False, encoding="utf-8")
            
            logger.info(f"Saved {len(data)} records to CSV: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise
    
    def save_all(self, data: List[Dict], formats: List[str] = None) -> Dict[str, str]:
        """
        Save data in all specified formats.
        
        Args:
            data: List of normalized records.
            formats: List of formats to save ('json', 'csv'). 
                    If None, saves all formats.
        
        Returns:
            Dictionary mapping format names to file paths.
        """
        if formats is None:
            formats = ["json", "csv"]
        
        saved_paths = {}
        
        csv_filename = "risk_input_bundle.csv"

        if "json" in formats:
            saved_paths["json"] = self.save_json(data)
            json_path = Path(saved_paths["json"])
            csv_filename = f"{json_path.stem}.csv"
        
        if "csv" in formats:
            saved_paths["csv"] = self.save_csv(data, filename=csv_filename)
        
        logger.info(f"Data saved in {len(saved_paths)} formats")
        return saved_paths
    
    @staticmethod
    def verify_data(data: List[Dict]) -> Dict:
        """
        Verify data quality and provide summary statistics.
        
        Args:
            data: List of normalized records.
        
        Returns:
            Dictionary with verification statistics.
        """
        stats = {
            "total_records": len(data),
            "sources": {},
            "date_range": {"min": None, "max": None},
            "empty_content": 0
        }
        
        dates = []
        for record in data:
            # Count by source
            source = record.get("source_type", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            # Track dates
            date = record.get("date")
            if date:
                dates.append(date)
            
            # Count empty content
            if not record.get("content"):
                stats["empty_content"] += 1
        
        if dates:
            dates.sort()
            stats["date_range"]["min"] = dates[0]
            stats["date_range"]["max"] = dates[-1]
        
        logger.info(f"Data verification: {json.dumps(stats, indent=2)}")
        return stats
