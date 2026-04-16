# CFA Simplified Pipeline

This repository is organized as a 3-layer pipeline:

1. Layer 1: Data collection and normalization
2. Layer 2: NLP enrichment (sentiment, entities, geo tags)
3. Layer 3: LLM risk analysis report generation

## Folder Structure

```text
CFASimplified/
├── data/
│   ├── risk_input_bundle.json
│   ├── risk_input_bundle.csv
│   ├── enriched_risk_bundle.json
│   └── risk_report.json
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
└── requirements.txt
```

## Setup


Ensure you are in root folder (../CFASimplified/)

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
