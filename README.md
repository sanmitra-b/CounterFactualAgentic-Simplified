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
│   ├── fitted_scm.pkl
│   ├── counterfactual_results.json
│   └── risk_solution_bundle.json
├── layer0_domain_selector/
│   ├── __init__.py
│   ├── domain_profile.py
│   └── layer0.py                   # Entry point: interactive domain setup
├── layer1_data_collection/
│   ├── collectors/
│   │   ├── financial_collector.py
│   │   ├── job_collector.py          # Adzuna + USAJOBS collection
│   │   ├── news_collector.py
│   │   ├── social_collector.py       # Multi-platform: Reddit, YouTube, Mastodon, HackerNews
│   │   └── __init__.py
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
│   ├── layer4_pipeline.py
│   ├── agent_loop.py
│   ├── tool_get_causal_paths.py
│   ├── tool_run_counterfactual.py
│   ├── tool_log_intervention.py
│   ├── scm_fitter.py
│   ├── causal_graph.py
│   └── schemas_layer4.py
├── layer5_rag/
│   ├── layer5_rag.py
│   ├── confidence_scorer.py
│   ├── vector_store.py
│   ├── playbook_kb.py
│   └── schemas_layer5.py
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

Layer 1 data collection requires API keys for multiple services. Create a `.env` file in the project root:

```bash
cp .env.example .env  # or create manually
```

Edit `.env` with your API credentials:

```env
# LLM
GROQ_API_KEY="your_groq_api_key"
GEMINI_API_KEY="your_gemini_api_key"  # For Layer 0 domain config suggestions (optional)

# News
NEWSAPI_KEY="your_newsapi_key"
GNEWS_API_KEY="your_gnews_api_key"  # optional for higher limits on GNews

# Financial
ALPHA_VANTAGE_KEY="your_alpha_vantage_key"
FRED_API_KEY="your_fred_api_key"

# Jobs (REQUIRED for Layer 1)
ADZUNA_ID="your_adzuna_id"
ADZUNA_API_KEY="your_adzuna_api_key"
USAJOBS_API_KEY="your_usajobs_api_key"
USAJOBS_USER_AGENT="your_email@example.com"

# Social (Optional)
YOUTUBE_API_KEY="your_youtube_api_key"  # For YouTube Data API v3 (optional, novel data source)
# Mastodon and HackerNews require no API keys
```

## Setup

Ensure you are in the project root folder with the Counterfactual conda environment activated.

```bash
pip install -r requirements.txt
```

## Run Pipeline

### Layer 0 (Entry Point)

Layer 0 accepts a risk domain and orchestrates the full pipeline (Layers 1-3).

**Interactive mode:**
```bash
python layer0_domain_selector/layer0.py
```
Prompts you for:
- Domain name (e.g., "Supply Chain", "Healthcare", "Cybersecurity")
- Domain description
- Core keywords (comma-separated)
- Whether to use Gemini 2.5 Flash Lite for config suggestions

**CLI mode:**
```bash
python layer0_domain_selector/layer0.py --domain "Supply Chain" --description "Risk signals for logistics and procurement" --keywords "supply chain disruption,supplier risk,logistics delay" --use-gemini
```

Layer 0 will:
1. Generate domain-specific `layer1_data_collection/config.json`
2. Run Layer 1 (data collection) with domain-specific keywords
3. Run Layer 2 (NLP enrichment)
4. Run Layer 3 (LLM risk analysis)
5. Output final `data/risk_report.json`

**Expected outputs after Layer 0 → Layer 3:**
- `layer1_data_collection/config.json` (updated with domain)
- `data/risk_input_bundle.json`
- `data/enriched_risk_bundle.json`
- `data/risk_report.json`

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

### Layer 4

```bash
python layer4_counterfactual/layer4_pipeline.py --input data/risk_report.json --output data/counterfactual_results.json
```

Expected Layer 4 output:
- `data/fitted_scm.pkl`
- `data/counterfactual_results.json`

### Layer 5

```bash
python layer5_rag/layer5_rag.py --input data/counterfactual_results.json --output data/risk_solution_bundle.json
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

User runs Layer 0 and enters:
- Domain: "Supply Chain Resilience"
- Description: "Risk signals across logistics, inventory, and supplier continuity"
- Keywords: "supply chain disruption, supplier risk, logistics delay"

Layer 0 generates domain-specific config, then orchestrates Layers 1-3.

### Step A: Layer 1 output (`data/risk_input_bundle.json`)

Layer 1 creates a structured bundle from raw APIs.

**Layer 1 Configuration:**
- Edit `layer1_data_collection/config.json` to enable/disable sources

**Layer 1 Data Sources:**
- **News:** NewsAPI and RSS feeds (130+ articles)
- **Financial:** Alpha Vantage, FRED, yfinance (800+ records)

#### Layer 1 Job Collection Details

The `job_collector.py` module fetches job postings from two major APIs:

**Adzuna API (Free Tier):**
- Endpoint: `https://api.adzuna.com/v1/api/jobs/us/search/1`
- Search terms: Financial Analyst, Economist, Data Scientist, Risk Analyst, Actuary

**USAJOBS API (Federal Jobs):**
- Endpoint: `https://data.usajobs.gov/api/search`
- Agencies: Treasury (TR), SEC (SE), Commerce (CM)
- Same search terms as Adzuna


**Layer 1 Configuration:**
- Edit `layer1_data_collection/config.json` to enable/disable sources


#### Layer 1 Social Data Collection Details

The `social_collector.py` module collects from four independent platforms:

**Reddit via Pushshift/PullPush (Free, Archive Mirror):**
- Endpoint: `https://api.pullpush.io/reddit/search/submission/`
- Subreddits: r/jobs, r/careerguidance, r/cscareerquestions, r/datascience
- Keywords: AI layoffs, job displacement, automation, hiring freeze, reskilling

**YouTube Data API v3 (Requires Google API Key):**
- Keywords: AI replacing jobs, artificial intelligence jobs, automation workforce


**Mastodon (Decentralized, No Approval):**
- Instances: fosstodon.org, techhub.social, mstdn.social
- Keywords: AI jobs, automation, tech layoffs, artificial intelligence, hiring freeze
- Hashtags: #ai, #jobs, #automation, #techcommunity


**HackerNews via Algolia Search (Free, Citable):**
- Endpoint: `https://hn.algolia.com/api/v1/search`
- Keywords: AI jobs, automation, tech layoffs, displacement, hiring freeze
- Filters: Stories with minimum 5 comments (signal filtering)


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

## Notes

- Layer 4 computes interventions using causal counterfactual simulation; optional GROQ reflection does not change core intervention math.
- Layer 5 retrieves mitigation playbooks from a vector store (Chroma/FAISS/NumPy fallback) and ranks them with a confidence scorer.
