from __future__ import annotations
 
from typing import List
from schemas_layer5 import PlaybookChunk
 
# ─────────────────────────────────────────────────────────────────────────────
# PLAYBOOK ENTRIES
# Each entry: id, title, category, intervention_type, causal_variable, chunks[]
# ─────────────────────────────────────────────────────────────────────────────
 
RAW_PLAYBOOKS: List[dict] = [
 
    # ── 1. PORT CONGESTION ────────────────────────────────────────────────────
    {
        "id": "PB-001", "title": "Port Congestion — Emergency Rerouting Protocol",
        "category": "Port Congestion", "intervention_type": "reroute",
        "causal_variable": "port_congestion",
        "chunks": [
            {
                "chunk_id": "PB-001-01",
                "text": (
                    "Port Congestion Rerouting Protocol: When vessel wait times exceed 72 hours at primary ports "
                    "(Singapore, Rotterdam, Los Angeles), immediately activate emergency rerouting procedures. "
                    "Step 1: Identify alternative deep-water ports within 500 nautical miles. "
                    "Step 2: Assess berth availability and customs clearance capacity at alternative ports. "
                    "Step 3: Negotiate spot rates with carriers for alternative routing. "
                    "Step 4: Notify all downstream warehouses and distribution centres of revised ETA. "
                    "Step 5: Update inventory buffers to absorb the 3–7 day delay differential. "
                    "Key metrics: vessel dwell time, berth utilisation rate, port throughput index. "
                    "Cost implication: 8–15% freight premium for emergency rerouting."
                ),
                "metadata": {"action_steps": [
                    "Identify alternative ports within 500nm",
                    "Assess berth and customs capacity",
                    "Negotiate spot carrier rates",
                    "Notify downstream warehouses of ETA changes",
                    "Increase inventory buffers by 10–20%",
                ]}
            },
            {
                "chunk_id": "PB-001-02",
                "text": (
                    "Port Congestion — Pre-emptive Carrier Diversification: Maintain active contracts with "
                    "minimum 3 ocean carriers per trade lane to ensure rerouting flexibility. "
                    "Pre-qualify at least 2 alternative ports per origin-destination pair. "
                    "Use real-time AIS vessel tracking (MarineTraffic, VesselFinder) to monitor wait times. "
                    "Trigger threshold: port congestion index above 0.65 or vessel queue > 15 ships. "
                    "SLA requirement: alternative routing decision within 4 hours of congestion trigger. "
                    "Long-term: establish relationships with inland container depots (ICDs) to decongest primary ports."
                ),
                "metadata": {"action_steps": [
                    "Maintain contracts with 3+ carriers per trade lane",
                    "Pre-qualify 2 alternative ports per OD pair",
                    "Deploy AIS monitoring with 0.65 congestion index trigger",
                    "4-hour SLA for rerouting decision",
                    "Establish ICD relationships for decongestio",
                ]}
            },
        ]
    },
 
    # ── 2. SHIPPING DELAY ─────────────────────────────────────────────────────
    {
        "id": "PB-002", "title": "Shipping Delay — Buffer Stock and Modal Shift Strategy",
        "category": "Shipping Delay", "intervention_type": "stockpile",
        "causal_variable": "shipping_delay",
        "chunks": [
            {
                "chunk_id": "PB-002-01",
                "text": (
                    "Shipping Delay Mitigation — Safety Stock Recalibration: When shipping delay probability "
                    "exceeds 0.50, increase safety stock levels by 20–35% for all A-class SKUs. "
                    "Calculate reorder point using: ROP = (Average Daily Demand × Lead Time) + Safety Stock. "
                    "Increase safety stock multiplier from 1.5× to 2.0× standard deviation of demand. "
                    "Deploy vendor-managed inventory (VMI) agreements with top 10 suppliers. "
                    "Action: pre-position 30-day safety stock at 3 regional distribution hubs. "
                    "Cost: 12–18% increase in working capital; justified when delay probability > 0.60."
                ),
                "metadata": {"action_steps": [
                    "Increase safety stock 20–35% for A-class SKUs",
                    "Recalculate ROP with delay-adjusted lead time",
                    "Raise safety stock multiplier to 2.0× std dev",
                    "Deploy VMI with top-10 suppliers",
                    "Pre-position 30-day stock at 3 regional hubs",
                ]}
            },
            {
                "chunk_id": "PB-002-02",
                "text": (
                    "Shipping Delay — Air Freight Modal Shift Protocol: For high-value, low-volume critical "
                    "components, activate air freight modal shift when ocean ETA delay exceeds 14 days. "
                    "Criteria: item criticality score > 8/10, unit value > $500, or production halt risk. "
                    "Pre-negotiate air freight capacity with freight forwarders (quarterly capacity reservations). "
                    "Air freight cost premium: 6–10× ocean freight; ROI breakeven at >$2,000/kg item value. "
                    "Use consolidation services (LCL air) for mid-value items to reduce per-unit air cost by 40%. "
                    "SLA: modal shift decision within 24 hours of confirmed 14-day delay threshold breach."
                ),
                "metadata": {"action_steps": [
                    "Define air freight trigger: 14-day delay + criticality >8/10",
                    "Pre-negotiate quarterly air capacity with forwarders",
                    "Evaluate ROI at $2,000/kg breakeven",
                    "Use LCL air consolidation for mid-value items",
                    "24-hour SLA for modal shift decision",
                ]}
            },
        ]
    },
 
    # ── 3. COMMODITY PRICE SHOCK ──────────────────────────────────────────────
    {
        "id": "PB-003", "title": "Commodity Price Shock — Hedging and Dual-Sourcing Strategy",
        "category": "Commodity Price Shock", "intervention_type": "hedge",
        "causal_variable": "demand_shock",
        "chunks": [
            {
                "chunk_id": "PB-003-01",
                "text": (
                    "Commodity Price Shock — Financial Hedging Framework: Establish forward contracts and "
                    "options positions for top-5 commodity inputs (crude oil, steel, copper, semiconductors, resins). "
                    "Hedging horizon: 3–12 months forward, 60–80% of forecasted volume. "
                    "Instrument selection: futures (standardised, exchange-traded) for oil and metals; "
                    "OTC forwards for specialty chemicals. Options for asymmetric downside protection. "
                    "Trigger: commodity price increase >10% MoM or volatility index >25%. "
                    "Governance: monthly review by Treasury and Supply Chain Risk Committee. "
                    "Target: lock in 70% of commodity spend at budget price for next quarter."
                ),
                "metadata": {"action_steps": [
                    "Identify top-5 commodity exposures",
                    "Establish 3–12 month forward contracts at 60–80% volume",
                    "Use exchange futures for oil/metals, OTC for specialty",
                    "Trigger at >10% MoM price increase",
                    "Monthly Treasury + Supply Chain Risk Committee review",
                ]}
            },
            {
                "chunk_id": "PB-003-02",
                "text": (
                    "Commodity Price Shock — Dual/Multi-Sourcing Protocol: For commodities with price "
                    "volatility >20% annualised, mandate minimum 2 qualified suppliers in different geographies. "
                    "Supplier split: 60/40 primary/secondary under normal conditions; "
                    "switch to 40/60 when primary region faces price shock. "
                    "Annual supplier qualification audit: financial health, capacity, ESG score. "
                    "Contractual price adjustment clauses: index-linked pricing with commodity benchmarks "
                    "(LME for metals, Platts for energy). Cap annual price increase at CPI + 3%. "
                    "Inventory strategy: maintain 45-day buffer stock of price-volatile commodities."
                ),
                "metadata": {"action_steps": [
                    "Qualify 2+ suppliers in different geographies per commodity",
                    "Implement 60/40 primary/secondary split",
                    "Annual supplier financial and ESG audit",
                    "Index-linked pricing with LME/Platts benchmarks",
                    "Maintain 45-day buffer stock for volatile commodities",
                ]}
            },
        ]
    },
 
    # ── 4. GEOPOLITICAL DISRUPTION ────────────────────────────────────────────
    {
        "id": "PB-004", "title": "Geopolitical Disruption — Supply Chain Diversification and Nearshoring",
        "category": "Geopolitical Disruption", "intervention_type": "diversify",
        "causal_variable": "geopolitical_tension",
        "chunks": [
            {
                "chunk_id": "PB-004-01",
                "text": (
                    "Geopolitical Disruption — China+1 / Nearshoring Strategy: When geopolitical tension "
                    "index exceeds 0.60, accelerate supplier diversification to reduce single-country dependency. "
                    "Target: no single country > 40% of any critical component sourcing. "
                    "Priority diversification destinations: Vietnam, India, Mexico (nearshore for US), "
                    "Poland/Romania (nearshore for EU). "
                    "Timeline: 6–18 months for new supplier qualification and ramp-up. "
                    "Cost: 5–15% unit cost premium for nearshore; offset by 30–50% reduction in lead time variance. "
                    "Immediate action: dual-qualify backup suppliers in low-tension regions within 90 days."
                ),
                "metadata": {"action_steps": [
                    "Cap single-country sourcing at 40% of critical components",
                    "Prioritise Vietnam, India, Mexico, Poland for diversification",
                    "90-day timeline for backup supplier dual-qualification",
                    "Accept 5–15% cost premium for nearshoring",
                    "Track geopolitical tension index; trigger at 0.60",
                ]}
            },
            {
                "chunk_id": "PB-004-02",
                "text": (
                    "Geopolitical Disruption — Trade Policy Contingency Protocol: "
                    "Maintain a tariff scenario library with pre-modelled cost impacts for "
                    "top-20 trade lane/product combinations. "
                    "Actions upon new tariff announcement: (1) assess landed cost impact within 48h, "
                    "(2) evaluate tariff engineering options (HTS reclassification, first sale valuation), "
                    "(3) accelerate inventory build ahead of tariff effective date, "
                    "(4) renegotiate supplier contracts to share tariff burden. "
                    "Engage customs broker for free-trade zone (FTZ) and bonded warehouse options. "
                    "Monitor: USTR, EU Trade Directorate, WTO dispute panels weekly."
                ),
                "metadata": {"action_steps": [
                    "Maintain tariff scenario library for top-20 trade lanes",
                    "48h landed cost impact assessment on new announcements",
                    "Evaluate HTS reclassification and first-sale valuation",
                    "Pre-build inventory ahead of tariff effective dates",
                    "Engage customs broker for FTZ/bonded warehouse options",
                ]}
            },
        ]
    },
 
    # ── 5. WEATHER / NATURAL DISASTER ─────────────────────────────────────────
    {
        "id": "PB-005", "title": "Weather / Natural Disaster — Pre-positioning and Route Contingency",
        "category": "Weather / Natural Disaster", "intervention_type": "stockpile",
        "causal_variable": "weather_severity",
        "chunks": [
            {
                "chunk_id": "PB-005-01",
                "text": (
                    "Weather Disruption — Pre-positioning Protocol: Monitor typhoon season (May–November "
                    "Western Pacific), hurricane season (June–November Atlantic), and monsoon windows. "
                    "60 days before peak season: pre-position 45-day safety stock at inland warehouses "
                    "away from coastal exposure. "
                    "Trigger: NWS/JMA typhoon watch within 500km of key port → immediate cargo hold protocol. "
                    "Route contingency: activate northern Pacific routing (Tokyo → Seattle) as alternative "
                    "to Southeast Asia lanes. "
                    "Carrier communication: 24h advance notice to carriers for port avoidance instructions. "
                    "Insurance: ensure cargo policies include weather-induced delay and general average clauses."
                ),
                "metadata": {"action_steps": [
                    "Pre-position 45-day stock at inland warehouses pre-season",
                    "Activate cargo hold on typhoon watch within 500km",
                    "Switch to northern Pacific routing as contingency",
                    "24h carrier notice for port avoidance",
                    "Verify weather delay and general average insurance coverage",
                ]}
            },
            {
                "chunk_id": "PB-005-02",
                "text": (
                    "Weather Disruption — Climate Risk Scoring for Supplier Network: "
                    "Assign climate risk scores to all Tier-1 and Tier-2 suppliers using "
                    "TCFD-aligned physical risk metrics (flood, heat stress, water stress, cyclone exposure). "
                    "High-risk suppliers (score >7/10): require business continuity plans with "
                    "documented backup production sites. "
                    "Annual supplier site visits to validate BCP readiness. "
                    "Contractual requirement: suppliers in high-risk zones must maintain 30-day "
                    "finished goods buffer at a geographically distinct location. "
                    "Internal: maintain supply chain digital twin to simulate weather scenario impacts."
                ),
                "metadata": {"action_steps": [
                    "Score all Tier-1/2 suppliers on TCFD physical climate risk",
                    "Require BCP with backup sites for suppliers scoring >7/10",
                    "Annual BCP readiness validation visits",
                    "Mandate 30-day buffer stock at separate location for high-risk suppliers",
                    "Build supply chain digital twin for weather scenario simulation",
                ]}
            },
        ]
    },
 
    # ── 6. FINANCIAL MARKET VOLATILITY ────────────────────────────────────────
    {
        "id": "PB-006", "title": "Financial Market Volatility — FX Hedging and Working Capital Management",
        "category": "Financial Market Volatility", "intervention_type": "hedge",
        "causal_variable": "demand_shock",
        "chunks": [
            {
                "chunk_id": "PB-006-01",
                "text": (
                    "Financial Market Volatility — FX Hedging Programme: "
                    "Identify top-10 currency pairs by transaction volume (USD/CNY, USD/EUR, USD/JPY, USD/INR). "
                    "Hedging policy: 75% of forecasted 3-month FX exposure via forward contracts; "
                    "25% left unhedged for opportunistic upside. "
                    "Trigger rebalancing: spot rate moves >5% from budget rate. "
                    "Natural hedging: where possible, match revenue and cost currencies. "
                    "Escalation: CFO approval required for hedge positions >$10M notional. "
                    "Working capital: negotiate extended payment terms (Net 60–90) with suppliers "
                    "in high-volatility currency regions to reduce FX translation risk."
                ),
                "metadata": {"action_steps": [
                    "Map top-10 currency exposures",
                    "Hedge 75% of 3-month FX exposure via forwards",
                    "Rebalance on >5% spot rate move from budget",
                    "Implement natural hedging through currency matching",
                    "Negotiate Net 60–90 payment terms in volatile currency regions",
                ]}
            },
        ]
    },
 
    # ── 7. SUPPLIER INSOLVENCY ────────────────────────────────────────────────
    {
        "id": "PB-007", "title": "Supplier Insolvency — Early Warning and Dual-Qualification",
        "category": "Supplier Insolvency", "intervention_type": "diversify",
        "causal_variable": "supplier_reliability",
        "chunks": [
            {
                "chunk_id": "PB-007-01",
                "text": (
                    "Supplier Insolvency Early Warning System: Monitor quarterly financial health of "
                    "all Tier-1 suppliers using Dun & Bradstreet / Coface financial risk scores. "
                    "Red flags: (1) D&B score drops below 50, (2) payment terms breach > 30 days, "
                    "(3) news of credit rating downgrade, (4) public reports of workforce reduction >10%. "
                    "Trigger: any 2 red flags → initiate dual-qualification of replacement supplier. "
                    "Emergency stock build: upon insolvency signal, immediately build 60-day buffer "
                    "of all components sole-sourced from at-risk supplier. "
                    "Legal: ensure all tooling and IP is contractually owned by buyer, not supplier."
                ),
                "metadata": {"action_steps": [
                    "Quarterly D&B/Coface financial health monitoring",
                    "Define 4 red flag triggers for insolvency risk",
                    "2-red-flag rule triggers dual-qualification process",
                    "Build 60-day emergency buffer on insolvency signal",
                    "Verify buyer ownership of tooling and IP contractually",
                ]}
            },
        ]
    },
 
    # ── 8. DEMAND SHOCK ───────────────────────────────────────────────────────
    {
        "id": "PB-008", "title": "Demand Shock — Demand Sensing and Flexible Capacity Management",
        "category": "Demand Shock", "intervention_type": "monitor",
        "causal_variable": "demand_shock",
        "chunks": [
            {
                "chunk_id": "PB-008-01",
                "text": (
                    "Demand Shock Response — Demand Sensing Protocol: Deploy real-time demand sensing "
                    "using POS data, e-commerce signals, and social sentiment to detect demand shifts "
                    "2–4 weeks earlier than traditional forecasting. "
                    "Demand shock thresholds: upside >20% vs. forecast → expedite orders, activate "
                    "capacity reservations; downside >20% → defer POs, renegotiate MOQs, "
                    "divert inventory to high-velocity markets. "
                    "Weekly S&OP cadence with real-time dashboard for demand signal review. "
                    "Supplier flex capacity agreements: pre-negotiate ±20% volume flex with top suppliers "
                    "at no premium; ±30% at 5% premium."
                ),
                "metadata": {"action_steps": [
                    "Deploy POS and e-commerce demand sensing",
                    "Define ±20% threshold for demand shock response",
                    "Activate capacity reservations on upside shock",
                    "Defer POs and renegotiate MOQs on downside shock",
                    "Pre-negotiate ±20% flex capacity with top suppliers",
                ]}
            },
            {
                "chunk_id": "PB-008-02",
                "text": (
                    "Demand Shock — Inventory Repositioning and Markdown Management: "
                    "Upon confirmed downside demand shock (>20% below forecast for 2 consecutive weeks): "
                    "Step 1: Freeze all discretionary POs immediately. "
                    "Step 2: Reposition excess inventory from slow markets to high-velocity regions. "
                    "Step 3: Initiate structured markdown programme (10% → 20% → 30% over 4-week cadence). "
                    "Step 4: Activate secondary channel (B2B liquidation, off-price retail) for aged stock. "
                    "Step 5: Negotiate inventory return rights with top suppliers (target >30% return acceptance). "
                    "Target: clear excess inventory within 8 weeks to preserve cash and warehouse capacity."
                ),
                "metadata": {"action_steps": [
                    "Freeze discretionary POs on confirmed downside shock",
                    "Reposition inventory from slow to fast markets",
                    "4-week markdown cadence: 10% → 20% → 30%",
                    "Activate B2B liquidation for aged stock",
                    "Negotiate supplier return rights (>30% target)",
                ]}
            },
        ]
    },
 
    # ── 9. REGULATORY / TRADE POLICY ─────────────────────────────────────────
    {
        "id": "PB-009", "title": "Regulatory / Trade Policy — Compliance and Tariff Engineering",
        "category": "Regulatory / Trade Policy", "intervention_type": "escalate",
        "causal_variable": "geopolitical_tension",
        "chunks": [
            {
                "chunk_id": "PB-009-01",
                "text": (
                    "Regulatory / Trade Policy — Compliance Readiness Programme: "
                    "Maintain a live regulatory change calendar covering: US USMCA, EU CBAM, "
                    "UK Global Tariff, China export controls, OFAC sanctions lists. "
                    "Monthly review: trade compliance team reviews all new regulations affecting top-50 SKUs. "
                    "Tariff engineering options: (1) HTS code reclassification, (2) substantial transformation "
                    "in a third country to change country of origin, (3) first-sale valuation for customs. "
                    "Free trade zone (FTZ) strategy: bond inventory in FTZ to defer duty payment "
                    "and enable duty-free re-export. "
                    "Escalation: legal counsel review for any regulation increasing landed cost >3%."
                ),
                "metadata": {"action_steps": [
                    "Maintain live regulatory calendar (USMCA, CBAM, OFAC)",
                    "Monthly compliance review for top-50 SKUs",
                    "Evaluate HTS reclassification and country-of-origin options",
                    "Implement FTZ strategy for duty deferral",
                    "Legal review trigger at >3% landed cost increase",
                ]}
            },
        ]
    },
 
    # ── 10. INVENTORY SHORTAGE ────────────────────────────────────────────────
    {
        "id": "PB-010", "title": "Inventory Shortage — Emergency Procurement and Allocation",
        "category": "Shipping Delay", "intervention_type": "stockpile",
        "causal_variable": "inventory_shortage",
        "chunks": [
            {
                "chunk_id": "PB-010-01",
                "text": (
                    "Inventory Shortage — Emergency Procurement Protocol: "
                    "When inventory coverage drops below 7 days for critical components: "
                    "Step 1: Activate emergency procurement team (buyer + engineering + finance). "
                    "Step 2: Issue RFQ to minimum 3 qualified spot-market suppliers within 4 hours. "
                    "Step 3: Accept price premium up to 40% over standard cost to secure supply. "
                    "Step 4: Arrange expedited freight (air if sea lead time >14 days). "
                    "Step 5: Implement production allocation plan — prioritise highest-margin SKUs. "
                    "Step 6: Customer communication — proactive outreach on lead time revision. "
                    "Recovery target: restore 14-day inventory coverage within 10 business days."
                ),
                "metadata": {"action_steps": [
                    "Activate emergency procurement team at <7 days coverage",
                    "Issue RFQ to 3+ spot-market suppliers within 4 hours",
                    "Accept up to 40% cost premium for critical supply",
                    "Arrange expedited/air freight if sea lead time >14 days",
                    "Allocate production to highest-margin SKUs",
                    "Proactive customer lead time communication",
                ]}
            },
        ]
    },
 
    # ── 11. CYBER / INFRASTRUCTURE ────────────────────────────────────────────
    {
        "id": "PB-011", "title": "Cyber / Infrastructure Risk — Business Continuity and Manual Fallback",
        "category": "Cyber / Infrastructure Risk", "intervention_type": "escalate",
        "causal_variable": "supplier_reliability",
        "chunks": [
            {
                "chunk_id": "PB-011-01",
                "text": (
                    "Cyber / Infrastructure Risk — Supply Chain BCP Protocol: "
                    "Maintain offline backup of all critical supplier contacts, PO data, and "
                    "inventory records (updated weekly, stored in geographically separate data centre). "
                    "Cyber incident response playbook for supply chain: "
                    "(1) isolate affected systems within 1 hour, "
                    "(2) switch to manual PO processing within 4 hours, "
                    "(3) activate backup EDI connections with top-20 suppliers within 8 hours, "
                    "(4) notify customers of potential delay within 24 hours. "
                    "Annual tabletop exercise simulating cyber-induced supply chain disruption. "
                    "Third-party cyber risk scoring for all Tier-1 suppliers (BitSight / SecurityScorecard)."
                ),
                "metadata": {"action_steps": [
                    "Weekly offline backup of supplier and PO data",
                    "1-hour system isolation on cyber incident",
                    "4-hour manual PO processing fallback",
                    "8-hour backup EDI activation with top-20 suppliers",
                    "Annual supply chain cyber tabletop exercise",
                ]}
            },
        ]
    },
]
 
 
def get_all_chunks() -> List[PlaybookChunk]:
    """Flatten all playbook entries into a list of PlaybookChunk objects."""
    chunks: List[PlaybookChunk] = []
    for pb in RAW_PLAYBOOKS:
        for chunk_raw in pb["chunks"]:
            chunks.append(PlaybookChunk(
                chunk_id         = chunk_raw["chunk_id"],
                playbook_id      = pb["id"],
                playbook_title   = pb["title"],
                category         = pb["category"],
                intervention_type= pb["intervention_type"],
                text             = chunk_raw["text"],
                metadata         = chunk_raw.get("metadata", {}),
            ))
    return chunks
 