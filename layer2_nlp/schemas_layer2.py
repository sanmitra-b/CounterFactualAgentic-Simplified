from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class SentimentScore:
    label: str
    score: float
    positive: float
    negative: float
    neutral: float
    model: str
    error: Optional[str] = None


@dataclass
class NEREntity:
    text: str
    entity_type: str
    score: float
    model: str


@dataclass
class EnrichedNewsItem:
    title: str
    body: str
    source: str
    url: str
    published_at: str
    sentiment: SentimentScore
    entities: List[NEREntity] = field(default_factory=list)
    geo_tags: List[str] = field(default_factory=list)
    reliability: float = 0.0


@dataclass
class EnrichedSocialSignal:
    text: str
    source: str
    platform: str
    subreddit: str
    post_id: str
    created_at: str
    score: int
    num_comments: int
    sentiment: SentimentScore
    entities: List[NEREntity] = field(default_factory=list)
    geo_tags: List[str] = field(default_factory=list)
    reliability: float = 0.0


@dataclass
class EnrichedStockSignal:
    ticker: str
    company: str
    price: float
    change_pct: float
    volume: int
    market_cap: float
    currency: str
    source: str
    fetched_at: str
    sentiment: SentimentScore
    reliability: float = 0.0


@dataclass
class EnrichedPortSignal:
    port_name: str
    country: str
    commodity: str
    congestion_flag: bool
    avg_wait_days: float
    throughput_teu: int
    source: str
    fetched_at: str
    sentiment: SentimentScore
    entities: List[NEREntity] = field(default_factory=list)
    geo_tags: List[str] = field(default_factory=list)
    reliability: float = 0.0


@dataclass
class EnrichedWeatherSignal:
    city: str
    country: str
    description: str
    temperature_c: float
    wind_kmh: float
    disruption_flag: bool
    source: str
    fetched_at: str
    sentiment: SentimentScore
    reliability: float = 0.0


@dataclass
class EnrichedCommoditySignal:
    commodity: str
    price: float
    currency: str
    unit: str
    change_pct: float
    source: str
    fetched_at: str
    sentiment: SentimentScore
    reliability: float = 0.0


@dataclass
class EnrichedRiskInputBundle:
    domain: str
    fetched_at: str
    enriched_at: str
    news: List[EnrichedNewsItem] = field(default_factory=list)
    social: List[EnrichedSocialSignal] = field(default_factory=list)
    stocks: List[EnrichedStockSignal] = field(default_factory=list)
    ports: List[EnrichedPortSignal] = field(default_factory=list)
    weather: List[EnrichedWeatherSignal] = field(default_factory=list)
    commodities: List[EnrichedCommoditySignal] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    layer1_completeness: float = 0.0
    aggregate_sentiment: str = "neutral"
    avg_reliability: float = 0.0
    total_items: int = 0
    top_geo_tags: List[str] = field(default_factory=list)

    def compute_aggregate_sentiment(self) -> None:
        labels = []
        reliabilities = []
        geo_counts: Dict[str, int] = {}

        text_groups: List[List[Any]] = [self.news, self.social, self.ports]
        score_groups: List[List[Any]] = [self.news, self.social, self.stocks, self.ports, self.weather, self.commodities]

        for group in score_groups:
            for item in group:
                if getattr(item, "sentiment", None):
                    labels.append(item.sentiment.label.lower())
                reliabilities.append(float(getattr(item, "reliability", 0.0)))

        for group in text_groups:
            for item in group:
                for geo in getattr(item, "geo_tags", []):
                    geo_counts[geo] = geo_counts.get(geo, 0) + 1

        pos = labels.count("positive")
        neg = labels.count("negative")
        neu = labels.count("neutral")

        if pos > max(neg, neu):
            self.aggregate_sentiment = "positive"
        elif neg > max(pos, neu):
            self.aggregate_sentiment = "negative"
        else:
            self.aggregate_sentiment = "neutral"

        self.avg_reliability = round(sum(reliabilities) / len(reliabilities), 4) if reliabilities else 0.0
        self.total_items = (
            len(self.news)
            + len(self.social)
            + len(self.stocks)
            + len(self.ports)
            + len(self.weather)
            + len(self.commodities)
        )
        self.top_geo_tags = [k for k, _ in sorted(geo_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
