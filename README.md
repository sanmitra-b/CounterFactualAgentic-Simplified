# CFA Simplified Pipeline

This repository is organized as a 5-layer pipeline:

1. Layer 1: Data collection and normalization
2. Layer 2: NLP enrichment (sentiment, entities, geo tags)
3. Layer 3: LLM risk analysis report generation
4. Layer 4: Counterfactual-driven agentic optimization (Causal interventions and SCM)
5. Layer 5: RAG-based mitigation mapping (retrieve and rank actions)

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
├── layer1_data_collection/
│   ├── collectors/
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
└── requirements.txt
```

## Prerequisites

- Python 3.11+
- Conda or venv for environment management
- Git for version control

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

## Setup

Ensure you are in the project root folder.

```bash
pip install -r requirements.txt
```

## Run Pipeline

### Layer 1

```bash
python layer1_data_collection/collect_data.py
```

Expected output:
- `data/risk_input_bundle.json`
- `data/risk_input_bundle.csv`

### Layer 2

```bash
python layer2_nlp/layer2_nlp.py --input data/risk_input_bundle.json --output data/enriched_risk_bundle.json
```

### Layer 3

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

## End-to-End Data Flow (Simple)

This section shows how one run flows from Layer 1 to Layer 5.

### Step A: Layer 1 output (`data/risk_input_bundle.json`)

Layer 1 creates a structured bundle from raw APIs.

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
- You now have ranked, practical actions linked to causal evidence.
- This is the final handoff for decision support.

## Quick Mental Model

- Layer 1: Collect signals
- Layer 2: Add intelligence (sentiment/entities/reliability)
- Layer 3: Decide top risks
- Layer 4: Test what intervention works causally
- Layer 5: Retrieve and rank concrete mitigations

## API Keys

Keep API keys in `.env` at the repository root.

Typical keys:
- `NEWSAPI_KEY`
- `ALPHA_VANTAGE_KEY`
- `FRED_API_KEY`
- `OPENWEATHER_API_KEY`
- `GROQ_API_KEY`

## Notes

- Layer 2 and Layer 3 now default to using files in `data/`.
- Layer 4 computes interventions using causal counterfactual simulation; optional GROQ reflection does not change core intervention math.
- Layer 5 retrieves mitigation playbooks from a vector store (Chroma/FAISS/NumPy fallback) and ranks them with a confidence scorer.
