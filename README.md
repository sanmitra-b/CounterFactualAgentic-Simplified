# CFA Simplified Pipeline

This repository is organized as a 4-layer pipeline:

1. Layer 1: Data collection and normalization
2. Layer 2: NLP enrichment (sentiment, entities, geo tags)
3. Layer 3: LLM risk analysis report generation
4. Layer 4: Counterfactual-driven agentic optimization (Causal interventions and SCM)

## Folder Structure

```text
CFASimplified/
├── data/
│   ├── risk_input_bundle.json
│   ├── risk_input_bundle.csv
│   ├── enriched_risk_bundle.json
│   ├── risk_report.json
│   ├── fitted_scm.pkl
│   └── counterfactual_results.json
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
