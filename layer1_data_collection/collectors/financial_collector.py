"""
Financial Data Collector Module
Fetches stock prices from Alpha Vantage and economic indicators from FRED.
"""

import logging
from typing import List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FinancialCollector:
    """Collects financial and economic data from free sources."""
    
    def __init__(self, config: Dict):
        """
        Initialize the financial collector with configuration.
        
        Args:
            config: Dictionary containing 'financial' settings from config.json
        """
        self.config = config
        self.data = []
    
    def collect_from_alpha_vantage(self) -> List[Dict]:
        """
        Fetch stock prices from Alpha Vantage API.
        
        Returns:
            List of stock price data dictionaries.
        """
        if not self.config.get("alpha_vantage", {}).get("enabled"):
            logger.info("Alpha Vantage disabled in config")
            return []
        
        api_key = self.config["alpha_vantage"].get("api_key")
        if not api_key or api_key == "YOUR_KEY_HERE":
            logger.warning("Alpha Vantage API key not configured. Skipping Alpha Vantage.")
            return []
        
        try:
            import requests
            import time
        except ImportError:
            logger.error("requests library not installed")
            return []
        
        tickers = self.config["alpha_vantage"].get("tickers", [])
        outputsize = self.config["alpha_vantage"].get("outputsize", "compact")
        rate_limit = self.config["alpha_vantage"].get("rate_limit_seconds", 1.5)
        prices = []
        
        base_url = "https://www.alphavantage.co/query"
        
        for ticker in tickers:
            try:
                # Get daily time series data
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": ticker,
                    "apikey": api_key,
                    "outputsize": outputsize  # Use compact for free tier
                }
                
                response = requests.get(base_url, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for errors
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                    continue
                
                if "Note" in data:
                    logger.warning(f"Alpha Vantage API limit hit: {data['Note']}")
                    continue

                if "Information" in data:
                    logger.warning(f"Alpha Vantage info for {ticker}: {data['Information']}")
                    continue
                
                time_series = data.get("Time Series (Daily)", {})

                if not time_series:
                    logger.warning(f"Alpha Vantage returned no time series data for {ticker}")
                    continue
                
                for date_str, daily_data in time_series.items():
                    prices.append({
                        "ticker": ticker,
                        "date": date_str,
                        "price": float(daily_data.get("4. close", 0)),
                        "open": float(daily_data.get("1. open", 0)),
                        "high": float(daily_data.get("2. high", 0)),
                        "low": float(daily_data.get("3. low", 0)),
                        "volume": int(daily_data.get("5. volume", 0)),
                        "source": "Alpha Vantage"
                    })
                
                logger.info(f"Alpha Vantage: Downloaded {len(time_series)} days of data for {ticker}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Alpha Vantage request failed for {ticker}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process Alpha Vantage data for {ticker}: {e}")
                continue
            
            # Rate limiting for free tier
            time.sleep(rate_limit)
        
        return prices
    
    def collect_from_fred(self) -> List[Dict]:
        """
        Fetch economic indicators from FRED (Federal Reserve Economic Data).
        
        Returns:
            List of economic indicator data dictionaries.
        """
        if not self.config.get("fred", {}).get("enabled"):
            logger.info("FRED disabled in config")
            return []
        
        api_key = self.config["fred"].get("api_key")
        if not api_key or api_key == "YOUR_KEY_HERE":
            logger.warning("FRED API key not configured. Skipping FRED collection.")
            return []
        
        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []
        
        indicators = self.config["fred"].get("indicators", [])
        economic_data = []
        
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        for indicator in indicators:
            try:
                # Get data for past 10 years
                start_date = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
                
                params = {
                    "series_id": indicator,
                    "api_key": api_key,
                    "file_type": "json",
                    "observation_start": start_date
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                for obs in data.get("observations", []):
                    # Skip if no value available
                    if obs.get("value") == ".":
                        continue
                    
                    economic_data.append({
                        "indicator": indicator,
                        "date": obs.get("date"),
                        "value": float(obs.get("value")),
                        "source": "FRED"
                    })
                
                logger.info(f"FRED: Downloaded {len(data.get('observations', []))} observations for {indicator}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"FRED request failed for {indicator}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process FRED data for {indicator}: {e}")
                continue
        
        return economic_data
    
    def collect(self) -> List[Dict]:
        """
        Collect financial data from all enabled sources.
        
        Returns:
            List of all collected financial data.
        """
        logger.info("Starting financial data collection...")
        
        all_data = []
        
        # Collect from Alpha Vantage
        all_data.extend(self.collect_from_alpha_vantage())
        
        # Collect economic indicators
        all_data.extend(self.collect_from_fred())
        
        self.data = all_data
        logger.info(f"Financial collection complete. Total records: {len(self.data)}")
        
        return self.data


def collect_financial(config: Dict) -> List[Dict]:
    """
    Convenience function to collect financial data.
    
    Args:
        config: Dictionary with financial configuration.
    
    Returns:
        List of financial data dictionaries.
    """
    collector = FinancialCollector(config)
    return collector.collect()
