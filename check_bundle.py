import json
with open('data/counterfactual_bundle.json') as f:
    data = json.load(f)
    print(f"mitigation_efficiency: {data.get('mitigation_efficiency')}")
    print(f"total_scenarios: {data.get('total_scenarios')}")
    print(f"avg_delta: {data.get('avg_delta')}")
