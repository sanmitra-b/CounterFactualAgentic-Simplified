"""
Layer 6 PDF Report Generator — CFASimplified Risk Intelligence
Generates a comprehensive, easy-to-understand PDF report from all pipeline outputs.

Usage:
    python layer6/layer6_pdf.py --output risk_report.pdf
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether, PageTemplate, Frame
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfgen import canvas


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data")

SEVERITY_COLORS = {
    "CRITICAL": colors.HexColor("#ef4444"),
    "HIGH": colors.HexColor("#f97316"),
    "MEDIUM": colors.HexColor("#eab308"),
    "LOW": colors.HexColor("#22c55e"),
}

SEVERITY_BADGE = {
    "CRITICAL": "🔴 CRITICAL",
    "HIGH": "🟠 HIGH",
    "MEDIUM": "🟡 MEDIUM",
    "LOW": "🟢 LOW",
}


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM STYLES
# ─────────────────────────────────────────────────────────────────────────────

def create_styles():
    """Create custom paragraph styles for the PDF."""
    styles = getSampleStyleSheet()
    
    # Title styles
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor("#1e40af"),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='ReportSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=24,
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))
    
    # Section styles
    styles.add(ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        borderPadding=5
    ))
    
    styles.add(ParagraphStyle(
        name='SubsectionHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    ))
    
    # Body text - use custom name to avoid conflict
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#1f2937"),
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=14
    ))
    
    styles.add(ParagraphStyle(
        name='SmallText',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=4
    ))
    
    # Highlight/callout
    styles.add(ParagraphStyle(
        name='Highlight',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#1e40af"),
        fontName='Helvetica-Bold',
        spaceAfter=6
    ))
    
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_json_file(filename):
    """Safely load JSON file from data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filename}: {e}")
        return None


def extract_config_info():
    """Extract domain, keywords, APIs from layer1 config and input bundle."""
    config = load_json_file("../layer1_data_collection/config.json")
    input_bundle = load_json_file("risk_input_bundle.json")
    
    info = {
        "domain": "Unknown",
        "keywords": [],
        "sources_enabled": [],
        "total_records": 0,
        "records_by_source": {},
    }
    
    if input_bundle:
        info["domain"] = input_bundle.get("domain", "Unknown")
        info["total_records"] = sum([
            len(input_bundle.get("news", [])),
            len(input_bundle.get("social", [])),
            len(input_bundle.get("stocks", [])),
            len(input_bundle.get("weather", [])),
            len(input_bundle.get("commodities", [])),
            len(input_bundle.get("jobs", [])),
        ])
        
        # Count by source
        if "news" in input_bundle:
            info["records_by_source"]["news"] = len(input_bundle["news"])
        if "social" in input_bundle:
            info["records_by_source"]["social"] = len(input_bundle["social"])
        if "stocks" in input_bundle:
            info["records_by_source"]["stocks"] = len(input_bundle["stocks"])
        if "weather" in input_bundle:
            info["records_by_source"]["weather"] = len(input_bundle["weather"])
        if "commodities" in input_bundle:
            info["records_by_source"]["commodities"] = len(input_bundle["commodities"])
        if "jobs" in input_bundle:
            info["records_by_source"]["jobs"] = len(input_bundle["jobs"])
    
    if config and "sources" in config:
        # Extract keywords
        if "news" in config["sources"] and "keywords" in config["sources"]["news"]:
            info["keywords"] = config["sources"]["news"]["keywords"]
        
        # Extract enabled sources
        for source_name, source_config in config["sources"].items():
            if source_config.get("enabled", False):
                info["sources_enabled"].append(source_name.upper())
    
    return info


# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def add_cover_page(elements, styles, config_info):
    """Add cover page to PDF."""
    # Title
    elements.append(Spacer(1, 0.8*inch))
    elements.append(Paragraph("Risk Intelligence Report", styles['ReportTitle']))
    elements.append(Paragraph("CFASimplified Agentic Pipeline", styles['ReportSubtitle']))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Domain highlight
    domain_text = f"<b>Domain:</b> {config_info['domain'].replace('_', ' ').title()}"
    elements.append(Paragraph(domain_text, styles['Highlight']))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Generation timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated: {now}", styles['SmallText']))
    
    elements.append(Spacer(1, 0.5*inch))
    
    # Quick facts
    elements.append(Paragraph("Quick Facts:", styles['SubsectionHeading']))
    
    facts_data = [
        ["Total Records Collected", str(config_info['total_records'])],
        ["Sources Enabled", ", ".join(config_info['sources_enabled'])],
        ["Primary Keywords", ", ".join(config_info['keywords'][:3])],
    ]
    
    facts_table = Table(facts_data, colWidths=[2.5*inch, 2.5*inch])
    facts_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f0f9ff")),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor("#fafafa")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
    ]))
    elements.append(facts_table)
    
    elements.append(PageBreak())


def add_data_collection_section(elements, styles, config_info):
    """Add data collection summary."""
    elements.append(Paragraph("📊 Data Collection Summary", styles['SectionHeading']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Overview
    total_text = f"""
    <b>Total Records Collected:</b> {config_info['total_records']}<br/>
    <b>Collection Completeness:</b> All available sources processed<br/>
    <b>Keywords Used:</b> {', '.join(config_info['keywords'])}
    """
    elements.append(Paragraph(total_text, styles['CustomBody']))
    
    elements.append(Spacer(1, 0.15*inch))
    
    # Records by source table
    elements.append(Paragraph("Records by Source:", styles['SubsectionHeading']))
    
    source_data = [["Source", "Records", "Percentage"]]
    total = config_info['total_records']
    
    for source, count in sorted(config_info['records_by_source'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / total * 100) if total > 0 else 0
        source_data.append([
            source.upper(),
            str(count),
            f"{pct:.1f}%"
        ])
    
    # Add total row
    source_data.append([
        "TOTAL",
        str(total),
        "100.0%"
    ])
    
    source_table = Table(source_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    source_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#f0f9ff")),
        ('TEXTCOLOR', (0, -1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#f9fafb")]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#d1d5db")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(source_table)
    
    elements.append(Spacer(1, 0.2*inch))


def add_risk_analysis_section(elements, styles):
    """Add risk analysis section."""
    risk_report = load_json_file("risk_report.json")
    if not risk_report or "top_risks" not in risk_report:
        elements.append(Paragraph("⚠️ Risk Analysis — No data available", styles['SectionHeading']))
        elements.append(PageBreak())
        return
    
    elements.append(Paragraph("⚠️ Top 5 Identified Risks", styles['SectionHeading']))
    elements.append(Spacer(1, 0.1*inch))
    
    top_risks = risk_report.get("top_risks", [])[:5]
    
    for risk in top_risks:
        rank = risk.get("rank", "?")
        title = risk.get("title", "Unknown")
        severity = risk.get("severity", "UNKNOWN")
        confidence = risk.get("confidence", 0)
        probability = risk.get("probability_next_30d", 0)
        category = risk.get("category", "Other")
        evidence = risk.get("evidence", [])[:2]  # First 2 evidence points
        action = risk.get("recommended_action", "Review and monitor")
        
        # Risk card
        risk_text = f"""
        <b>Risk #{rank}: {title}</b><br/>
        <b>Category:</b> {category}<br/>
        <b>Severity:</b> {SEVERITY_BADGE.get(severity, severity)}<br/>
        <b>Confidence:</b> {confidence*100:.0f}%<br/>
        <b>Probability (30 days):</b> {probability*100:.0f}%
        """
        elements.append(Paragraph(risk_text, styles['CustomBody']))
        
        # Evidence
        elements.append(Paragraph("<b>Key Evidence:</b>", styles['SmallText']))
        for ev in evidence:
            ev_text = f"• {ev[:100]}..." if len(ev) > 100 else f"• {ev}"
            elements.append(Paragraph(ev_text, styles['SmallText']))
        
        # Recommended action
        action_text = f"<b>Recommended Action:</b> {action}"
        elements.append(Paragraph(action_text, styles['Highlight']))
        
        elements.append(Spacer(1, 0.15*inch))
    
    elements.append(PageBreak())


def add_counterfactual_section(elements, styles):
    """Add counterfactual analysis section."""
    cf_bundle = load_json_file("counterfactual_bundle.json")
    if not cf_bundle or "scenarios" not in cf_bundle:
        elements.append(Paragraph("🔄 Counterfactual Analysis — No data available", styles['SectionHeading']))
        elements.append(PageBreak())
        return
    
    elements.append(Paragraph("🔄 Counterfactual Analysis & Interventions", styles['SectionHeading']))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph(
        "This section shows <b>what-if</b> scenarios where we test interventions to reduce risk.",
        styles['CustomBody']
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    scenarios = cf_bundle.get("scenarios", [])
    
    # Group scenarios by risk rank
    scenarios_by_rank = {}
    for scenario in scenarios:
        rank = scenario.get("risk_rank", 0)
        if rank not in scenarios_by_rank:
            scenarios_by_rank[rank] = []
        scenarios_by_rank[rank].append(scenario)
    
    # Show best scenario per risk rank
    for rank in sorted(scenarios_by_rank.keys())[:3]:  # Top 3 risks
        scenarios_for_rank = sorted(
            scenarios_by_rank[rank],
            key=lambda x: abs(x.get("delta_probability", 0)),
            reverse=True
        )
        
        best_scenario = scenarios_for_rank[0]
        risk_title = best_scenario.get("risk_title", "Unknown Risk")
        intervention = best_scenario.get("intervention", "")
        intervention_type = best_scenario.get("intervention_type", "UNKNOWN")
        baseline_prob = best_scenario.get("baseline_probability", 0)
        counterfactual_prob = best_scenario.get("counterfactual_probability", 0)
        delta_prob = best_scenario.get("delta_probability", 0)
        feasibility = best_scenario.get("feasibility", "UNKNOWN")
        cost = best_scenario.get("estimated_cost_usd", "TBD")
        time_to_impact = best_scenario.get("time_to_impact_days", 0)
        
        elements.append(Paragraph(f"<b>Risk #{rank}: {risk_title}</b>", styles['SubsectionHeading']))
        
        # Truncate intervention text only if very long
        intervention_display = intervention[:500] if len(intervention) > 500 else intervention
        
        scenario_text = f"""
        <b>Intervention Type:</b> {intervention_type}<br/>
        <b>Strategy:</b> {intervention_display}<br/>
        <b>Baseline Risk Probability:</b> {baseline_prob*100:.0f}%<br/>
        <b>After Intervention:</b> {counterfactual_prob*100:.0f}% (↓ {abs(delta_prob)*100:.0f}%)<br/>
        <b>Implementation Feasibility:</b> {feasibility}<br/>
        <b>Estimated Cost:</b> {cost}<br/>
        <b>Time to Impact:</b> {time_to_impact} days
        """
        elements.append(Paragraph(scenario_text, styles['CustomBody']))
        elements.append(Spacer(1, 0.15*inch))
    
    elements.append(PageBreak())


def add_solutions_section(elements, styles):
    """Add solutions and mitigations section."""
    solutions_report = load_json_file("solution_mapping_report.json")
    if not solutions_report or "solutions" not in solutions_report:
        elements.append(Paragraph("💡 Recommended Solutions — No data available", styles['SectionHeading']))
        elements.append(PageBreak())
        return
    
    elements.append(Paragraph("💡 Recommended Solutions & Mitigations", styles['SectionHeading']))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph(
        "Based on the risk analysis and counterfactual interventions, here are the top recommended solutions:",
        styles['CustomBody']
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    solutions = solutions_report.get("solutions", [])
    
    # Group solutions by risk
    solutions_by_risk = {}
    for solution in solutions:
        risk_title = solution.get("risk_title", "Unknown")
        if risk_title not in solutions_by_risk:
            solutions_by_risk[risk_title] = []
        solutions_by_risk[risk_title].append(solution)
    
    # Show top 2 solutions per risk (top 3 risks)
    for i, (risk_title, risk_solutions) in enumerate(list(solutions_by_risk.items())[:3]):
        elements.append(Paragraph(f"<b>Risk: {risk_title}</b>", styles['SubsectionHeading']))
        
        for j, solution in enumerate(risk_solutions[:2]):
            sol_title = solution.get("solution_title", "Solution")
            sol_type = solution.get("solution_type", "UNKNOWN")
            description = solution.get("description", "")
            steps = solution.get("implementation_steps", [])
            cost = solution.get("estimated_cost_usd", "TBD")
            risk_reduction = solution.get("risk_reduction_estimate", "TBD")
            time_horizon = solution.get("time_horizon", "MEDIUM")
            relevance = solution.get("relevance_score", 0)
            
            # Truncate description only if very long
            description_display = description[:400] if len(description) > 400 else description
            
            sol_text = f"""
            <b>{j+1}. {sol_title}</b> [{sol_type}]<br/>
            <b>Type:</b> {sol_type} | <b>Time Horizon:</b> {time_horizon}<br/>
            <b>Relevance Score:</b> {relevance:.1%} | <b>Risk Reduction:</b> {risk_reduction}<br/>
            <b>Estimated Cost:</b> {cost}<br/>
            <b>Description:</b> {description_display}
            """
            elements.append(Paragraph(sol_text, styles['CustomBody']))
            
            # Implementation steps (top 3 - show more detail)
            if steps:
                elements.append(Paragraph("<b>Key Implementation Steps:</b>", styles['SmallText']))
                for step in steps[:3]:
                    step_text = f"• {step[:200]}..." if len(step) > 200 else f"• {step}"
                    elements.append(Paragraph(step_text, styles['SmallText']))
            
            elements.append(Spacer(1, 0.1*inch))
        
        if i < 2:  # Add page break between risks
            elements.append(PageBreak())


def add_summary_section(elements, styles, config_info):
    """Add executive summary / conclusion."""
    elements.append(Paragraph("📋 Report Summary", styles['SectionHeading']))
    elements.append(Spacer(1, 0.1*inch))
    
    summary_text = f"""
    This Risk Intelligence Report synthesizes multi-source signals across <b>{', '.join(config_info['sources_enabled'])}</b> 
    to identify, analyze, and recommend mitigations for risks in the <b>{config_info['domain'].replace('_', ' ').title()}</b> domain.
    <br/><br/>
    <b>Pipeline Overview:</b><br/>
    <b>Layer 1:</b> Collected {config_info['total_records']} signals from multiple sources<br/>
    <b>Layer 2:</b> Enriched signals with NLP sentiment, entities, and reliability scoring<br/>
    <b>Layer 3:</b> Identified and ranked top 5 risks using LLM analysis<br/>
    <b>Layer 4:</b> Tested counterfactual interventions to find risk-reducing actions<br/>
    <b>Layer 5:</b> Mapped interventions to practical solutions from knowledge base<br/>
    <br/>
    <b>Next Steps:</b><br/>
    1. Review the top identified risks and prioritize by severity and probability<br/>
    2. Evaluate counterfactual interventions for feasibility and cost-benefit<br/>
    3. Implement recommended solutions in order of priority and time-to-impact<br/>
    4. Monitor KPIs and re-run analysis as conditions evolve<br/>
    """
    elements.append(Paragraph(summary_text, styles['CustomBody']))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer info
    footer_text = """
    <b>Report Generated By:</b> CFASimplified Agentic AI Pipeline<br/>
    <b>Timestamp:</b> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """<br/>
    <b>Disclaimer:</b> This report is for informational purposes and should be reviewed by domain experts before decision-making.
    """
    elements.append(Paragraph(footer_text, styles['SmallText']))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf(output_path=None):
    """Generate comprehensive PDF report."""
    if output_path is None:
        output_path = OUTPUT_DIR / "risk_intelligence_report.pdf"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating PDF report: {output_path}")
    
    # Load data
    config_info = extract_config_info()
    print(f"Domain: {config_info['domain']}")
    print(f"Total records: {config_info['total_records']}")
    print(f"Sources: {', '.join(config_info['sources_enabled'])}")
    
    # Create PDF
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        title="Risk Intelligence Report",
        author="CFASimplified"
    )
    
    # Create styles
    styles = create_styles()
    
    # Build elements
    elements = []
    
    # Cover page
    add_cover_page(elements, styles, config_info)
    
    # Data collection
    add_data_collection_section(elements, styles, config_info)
    
    # Risk analysis
    add_risk_analysis_section(elements, styles)
    
    # Counterfactual analysis
    add_counterfactual_section(elements, styles)
    
    # Solutions
    add_solutions_section(elements, styles)
    
    # Summary
    add_summary_section(elements, styles, config_info)
    
    # Build PDF
    try:
        doc.build(elements)
        print(f"✓ PDF report generated successfully: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"✗ Error generating PDF: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PDF risk intelligence report from CFASimplified pipeline outputs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/risk_intelligence_report.pdf",
        help="Output PDF path (default: data/risk_intelligence_report.pdf)"
    )
    
    args = parser.parse_args()
    
    output_file = generate_pdf(Path(args.output))
    if output_file:
        sys.exit(0)
    else:
        sys.exit(1)
