# Counterfactual driven Agentic AI 

This repository is organized as a 7-layer pipeline:

0. **Layer 0: Domain selector** — Accept risk domain dynamically, generate domain-specific config
1. **Layer 1: Data collection and normalization** — Gather multi-source risk signals
2. **Layer 2: NLP enrichment** — Sentiment, entities, geo tags, reliability scoring
3. **Layer 3: LLM risk analysis** — Generate risk report via Groq
4. **Layer 4: Counterfactual-driven agentic optimization** — Causal interventions and SCM fitting
5. **Layer 5: RAG-based mitigation mapping** — Retrieve and rank mitigation actions
6. **Layer 6: Streamlit dashboard** — End-to-end risk intelligence visualization


## Folder Structure

```text
CFASimplified/
├── data/
│   ├── risk_input_bundle.json
│   ├── risk_input_bundle.csv
│   ├── enriched_risk_bundle.json
│   ├── risk_report.json
│   ├── counterfactual_results.json
│   └── risk_solution_bundle.json
├── layer0_domain_selector/
│   ├── __init__.py
│   ├── config_validator.py         # Config schema validation
│   └── layer0.py                   # Entry point: CLI-only domain setup (validator + Gemini refinement)
├── layer1_data_collection/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── financial_collector.py
│   │   ├── job_collector.py          # Adzuna + USAJOBS collection
│   │   ├── news_collector.py
│   │   ├── social_collector.py       # Multi-platform: Reddit, YouTube, Mastodon, HackerNews
│   │   └── weather_collector.py      # OpenWeather API (domain-specific cities)
│   ├── collect_data.py
│   ├── config.json
│   ├── normalizer.py
│   └── storage.py
├── layer2_nlp/
│   ├── layer2_nlp.py
│   └── schemas_layer2.py
├── layer3_llm/
│   ├── layer3_llm_analysis.py
│   └── schemas_layer3.py
├── layer4_counterfactual/
│   ├── analyst_agent.py
│   ├── engine.py
│   ├── layer4_supervisor.py
│   └── __init__.py
├── layer5_rag/
│   ├── knowledge_base.json
│   ├── layer5_supervisor.py
│   ├── librarian_agent.py
│   └── __init__.py
├── layer6/
│   └── layer6_dashboard.py
└── requirements.txt
```

## Prerequisites

- Python 3.11+
- Conda or venv for environment management

## Step 1: Clone the Repository

```bash
git clone https://github.com/sanmitra-b/CounterFactualAgentic-Simplified.git
cd CounterFactualAgentic-Simplified
```

## Step 2: Create and Activate Conda Environment

```bash
# Create environment
conda create -n Counterfactual python=3.11 -y

# Activate environment
conda activate Counterfactual
```

## Step 3: Configure API Keys

Layer 0 and Layer 1 require API keys for multiple services. Create a `.env` file in the project root:

```bash
cp .env.example .env  # or create manually
```

Edit `.env` with your API credentials:

```env
# LLM
GROQ_API_KEY="your_groq_api_key"
GEMINI_API_KEY="your_gemini_api_key"  # For Layer 0 domain extraction

# News
NEWSAPI_KEY="your_newsapi_key"
GNEWS_API_KEY="your_gnews_api_key"  # optional

# Financial
ALPHA_VANTAGE_KEY="your_alpha_vantage_key"
FRED_API_KEY="your_fred_api_key"

# Jobs (REQUIRED for Layer 1)
ADZUNA_ID="your_adzuna_id"
ADZUNA_API_KEY="your_adzuna_api_key"
USAJOBS_API_KEY="your_usajobs_api_key"
USAJOBS_USER_AGENT="your_email@example.com"

# Weather
OPENWEATHER_API_KEY="your_openweather_api_key"

# Social (Optional)
YOUTUBE_API_KEY="your_youtube_api_key"  # optional
# Mastodon and HackerNews require no API keys
```

## Setup

Ensure you are in the project root folder with the Counterfactual conda environment activated.

```bash
pip install -r requirements.txt
```

## Run Pipeline

### Layer 0 (Entry Point)

Layer 0 accepts a risk domain and orchestrates the full pipeline (Layers 1-5).

Layer 0 runs in CLI-only mode. Provide a natural-language question or domain via `--query` or `--domain`.

**CLI mode:**
```bash
# Provide a natural-language question or domain; Layer 0 will auto-extract description/keywords
python layer0_domain_selector/layer0.py --query "What is the risk in cold chain in vaccine transport?"

# Optional overrides (only needed if you want to replace the auto-extracted values):
python layer0_domain_selector/layer0.py --domain "Supply Chain" --description "Risk signals for logistics and procurement" --keywords "supply chain disruption,supplier risk,logistics delay"
```

Layer 0 will:
1. Generate domain-specific `layer1_data_collection/config.json` (validated before write; will attempt automated Gemini refinement on errors)
2. Run Layer 1 (data collection) with domain-specific keywords
3. Run Layer 2 (NLP enrichment)
4. Run Layer 3 (LLM risk analysis)
5. Run Layer 4 (counterfactual analysis) and Layer 5 (RAG mitigation mapping)
6. Output final artifacts including `data/risk_report.json`, `data/counterfactual_results.json`, and `data/risk_solution_bundle.json`


---

### Layer 1 (Manual Execution)

If running Layer 1 independently (without Layer 0):

```bash
python layer1_data_collection/collect_data.py
```

Expected output:
- `data/risk_input_bundle.json`
- `data/risk_input_bundle.csv`

### Layer 2 (Manual Execution)

If running Layer 2 independently:

```bash
python layer2_nlp/layer2_nlp.py --input data/risk_input_bundle.json --output data/enriched_risk_bundle.json
```

### Layer 3 (Manual Execution)

If running Layer 3 independently:

```bash
python layer3_llm/layer3_llm_analysis.py --input data/enriched_risk_bundle.json --output data/risk_report.json
```

### Layer 4 (Manual Execution)

If running Layer 4 independently:

```bash
python layer4_counterfactual/layer4_supervisor.py --input data/risk_report.json --output data/counterfactual_results.json
```

Expected Layer 4 output:
- `data/counterfactual_results.json`

### Layer 5 (Manual Execution)

If running Layer 5 independently:

```bash
python layer5_rag/layer5_supervisor.py --input data/counterfactual_results.json --output data/risk_solution_bundle.json
```

Expected Layer 5 output:
- `data/risk_solution_bundle.json`

### Layer 6 (Dashboard)

Run the Streamlit dashboard after Layers 1-5 outputs are available.

```bash
streamlit run layer6/layer6_dashboard.py
```

Layer 6 reads only pipeline outputs from `data/` (no demo fallback data):
- `data/risk_input_bundle.json`
- `data/enriched_risk_bundle.json`
- `data/risk_report.json`
- `data/counterfactual_results.json`
- `data/risk_solution_bundle.json`

If a required file is missing, the relevant page shows an empty-state message.

## End-to-End Data Flow (Simple)

This section shows how one complete run flows from Layer 0 through Layer 5.

### Step 0: Layer 0 — Domain Setup

User runs Layer 0 with a natural-language query:

```bash
python layer0_domain_selector/layer0.py --query "What is the risk in cold chain in vaccine transport?"
```

Layer 0 auto-extracts domain, description, and keywords using Gemini, generates domain-specific config, then orchestrates Layers 1-5.

### Step A: Layer 1 output (`data/risk_input_bundle.json`)

Layer 1 creates a structured bundle from raw APIs.

**Layer 1 Configuration:**
Layer 0 generates `layer1_data_collection/config.json` with domain-specific keywords, enabled/disabled sources, and relevant cities for weather collection.

**Layer 1 Data Sources:**
- **News:** NewsAPI and RSS feeds
- **Financial:** Alpha Vantage, FRED, yfinance
- **Jobs:** Adzuna and USAJOBS APIs
- **Social:** Reddit, YouTube, Mastodon, HackerNews
- **Weather:** OpenWeather API (domain-specific cities)


Example:

```json
{
	"domain": "supply_chain",
	"completeness_score": 0.75,
	"news": [{"title": "Prices For Physical Oil And Fertilizer Go Absolutely Nuts..."}],
	"social": [{"subreddit": "stocks", "text": "..."}],
	"stocks": [...],
	"weather": [...],
	"commodities": [...]
}
```

What this means:
- `completeness_score: 0.75` means 3 out of 4 key groups were available in this run.
- Data is still broad at this stage (not yet ranked as risks).

### Step B: Layer 2 output (`data/enriched_risk_bundle.json`)

Layer 2 adds sentiment, NER entities, geo tags, and reliability.

Example:

```json
{
	"aggregate_sentiment": "neutral",
	"avg_reliability": 0.9275,
	"news": [
		{
			"title": "...",
			"sentiment": {"label": "neutral", "score": 0.92},
			"entities": [{"text": "Norse Atlantic ASA", "entity_type": "ORG"}],
			"geo_tags": ["US", "Iran"],
			"reliability": 0.65
		}
	]
}
```

What this means:
- Now each signal has machine-readable quality and context.
- This is the input for LLM risk synthesis.

### Step C: Layer 3 output (`data/risk_report.json`)

Layer 3 converts enriched signals into top risks.

Example from current output:

```json
{
	"top_risks": [
		{
			"rank": 1,
			"category": "Commodity Price Volatility",
			"title": "Oil Price Fluctuations",
			"severity": "HIGH",
			"confidence": 0.85,
			"probability_next_30d": 0.7
		}
	]
}
```

What this means:
- Layer 3 picks and ranks the top 5 actionable risks.
- Each risk gets severity, confidence, evidence, and suggested action.

### Step D: Layer 4 output (`data/counterfactual_results.json`)

Layer 4 tests interventions iteratively and keeps the best one.

Example from current output for `risk_1`:

```json
{
	"risk_id": "risk_1",
	"risk_title": "Oil Price Fluctuations",
	"all_iterations": [
		{"iteration": 2, "intervention": {"variable": "demand_shock"}, "ite_mean": -0.0316, "threshold_cleared": false},
		{"iteration": 3, "intervention": {"variable": "inventory_shortage"}, "ite_mean": -0.0856, "threshold_cleared": false},
		{"iteration": 4, "intervention": {"variable": "inventory_shortage"}, "ite_mean": -0.1075, "threshold_cleared": true}
	],
	"best_intervention": {
		"intervention": {"variable": "inventory_shortage", "intervened_value": 0.44875},
		"ite_mean": -0.1075,
		"probability_of_improvement": 1.0,
		"threshold_cleared": true
	}
}
```

What this means:
- Negative `ite_mean` is good (risk goes down).
- The loop changed variable once effect was weak, then tuned magnitude.
- Final accepted intervention crossed the threshold.

### Step E: Layer 5 output (`data/risk_solution_bundle.json`)

Layer 5 retrieves playbook actions and ranks mitigation solutions.

Example from current output:

```json
{
	"total_risks_mapped": 5,
	"total_solutions": 15,
	"mappings": [
		{
			"risk_id": "risk_1",
			"risk_title": "Oil Price Fluctuations",
			"causal_variable": "inventory_shortage",
			"top_mitigations": [
				{"solution_rank": 1, "title": "Commodity Price Shock — Hedging and Dual-Sourcing Strategy", "confidence_score": 0.5948},
				{"solution_rank": 2, "title": "Commodity Price Shock — Hedging and Dual-Sourcing Strategy", "confidence_score": 0.5822},
				{"solution_rank": 3, "title": "Shipping Delay — Buffer Stock and Modal Shift Strategy", "confidence_score": 0.4872}
			]
		}
	]
}
```

What this means:
- We now have ranked, practical actions linked to causal evidence.
- This is the final handoff for decision support.

### Step F: Layer 6 visualization (`layer6/layer6_dashboard.py`)

Layer 6 presents all outputs in a single UI:
- Overview: aggregate sentiment, completeness, total signals, geo heatmap
- Sources: filterable all-source table with Layer 2 sentiment labels
- Risk Report: severity cards, confidence bars, probability, cause-effect chains
- Counterfactuals: ITE per risk, P(improve), causal variable traces
- Solutions: top mitigations ranked by confidence and similarity
