from __future__ import annotations

from typing import List

from schemas_layer5 import PlaybookChunk


RAW_PLAYBOOKS: List[dict] = [
    {
        "id": "AIJ-001",
        "title": "Job Displacement Risk - Workforce Redeployment Program",
        "category": "Job Displacement Risk",
        "intervention_type": "redeploy",
        "causal_variable": "automation_exposure",
        "chunks": [
            {
                "chunk_id": "AIJ-001-01",
                "text": (
                    "When automation exposure rises in operational or clerical roles, deploy a phased role-redesign "
                    "program instead of immediate role elimination. Build a role adjacency map and move at-risk workers "
                    "to AI-supervision, data quality, customer exception handling, and compliance workflows. "
                    "Use a 90-day redeployment window with weekly manager checkpoints."
                ),
                "metadata": {
                    "action_steps": [
                        "Build role adjacency map for high-risk job families",
                        "Freeze non-critical layoffs during 90-day transition window",
                        "Move workers into AI-supervision and exception workflows",
                        "Run weekly transition checkpoints with managers",
                        "Track redeployment rate and retention by job family",
                    ]
                },
            },
            {
                "chunk_id": "AIJ-001-02",
                "text": (
                    "Establish internal talent marketplace rules to prioritize displaced workers for open positions "
                    "before external hiring. Use skill badges and short practical assessments to match workers with "
                    "adjacent roles."
                ),
                "metadata": {
                    "action_steps": [
                        "Enable internal-first hiring policy for 6 months",
                        "Publish skill-badge pathways for target roles",
                        "Use practical assessments for role matching",
                        "Report internal fill-rate monthly",
                    ]
                },
            },
        ],
    },
    {
        "id": "AIJ-002",
        "title": "Skills Obsolescence Risk - Rapid Reskilling Pathways",
        "category": "Skills Obsolescence Risk",
        "intervention_type": "reskill",
        "causal_variable": "transition_friction",
        "chunks": [
            {
                "chunk_id": "AIJ-002-01",
                "text": (
                    "Launch 6 to 12 week stackable reskilling tracks for workers in roles with high automation "
                    "exposure. Curriculum should cover prompt literacy, human-in-the-loop review, data validation, "
                    "AI governance basics, and role-specific tool usage. Tie completion to guaranteed interview slots."
                ),
                "metadata": {
                    "action_steps": [
                        "Identify top 10 at-risk job roles",
                        "Create role-based 6-12 week reskilling tracks",
                        "Guarantee interviews for graduates",
                        "Set completion KPI above 70%",
                        "Measure post-training placement within 60 days",
                    ]
                },
            },
            {
                "chunk_id": "AIJ-002-02",
                "text": (
                    "Pair formal training with apprenticeship sprints in live AI-enabled workflows so workers gain "
                    "verifiable transition evidence."
                ),
                "metadata": {
                    "action_steps": [
                        "Create 4-week apprenticeship sprints",
                        "Assign mentors for each cohort",
                        "Issue transition-ready portfolio evidence",
                    ]
                },
            },
        ],
    },
    {
        "id": "AIJ-003",
        "title": "Wage Suppression Risk - Wage Insurance and Floor Protection",
        "category": "Wage Suppression Risk",
        "intervention_type": "wage_support",
        "causal_variable": "wage_pressure",
        "chunks": [
            {
                "chunk_id": "AIJ-003-01",
                "text": (
                    "When wage pressure rises in AI-exposed occupations, deploy temporary wage insurance to offset "
                    "income drops during role transitions. Combine this with revised pay bands for AI-augmented roles "
                    "to prevent permanent downward wage drift."
                ),
                "metadata": {
                    "action_steps": [
                        "Define wage insurance eligibility criteria",
                        "Cover a share of wage loss for transition period",
                        "Rebaseline pay bands for AI-augmented roles",
                        "Audit wage outcomes quarterly by role cohort",
                    ]
                },
            }
        ],
    },
    {
        "id": "AIJ-004",
        "title": "Hiring Slowdown Risk - Demand Stimulation and Placement",
        "category": "Hiring Slowdown Risk",
        "intervention_type": "hiring_incentive",
        "causal_variable": "labor_market_demand",
        "chunks": [
            {
                "chunk_id": "AIJ-004-01",
                "text": (
                    "Counter hiring slowdown by funding short-cycle hiring incentives in sectors with measurable AI "
                    "complementarity. Prioritize roles that absorb transitioning workers: implementation specialists, "
                    "AI operations analysts, data quality associates, and compliance reviewers."
                ),
                "metadata": {
                    "action_steps": [
                        "Offer time-bound hiring credits for target roles",
                        "Prioritize sectors with rising AI-complement demand",
                        "Tie incentives to net-new employment outcomes",
                        "Track placement rates monthly",
                    ]
                },
            }
        ],
    },
    {
        "id": "AIJ-005",
        "title": "Public Sector Workforce Transition Risk - Government Reskilling at Scale",
        "category": "Public Sector Workforce Transition Risk",
        "intervention_type": "policy_reform",
        "causal_variable": "policy_support",
        "chunks": [
            {
                "chunk_id": "AIJ-005-01",
                "text": (
                    "For public-sector transition risk, establish agency-wide AI transition frameworks with funded "
                    "reskilling mandates, role redesign guidelines, and mobility pathways across agencies."
                ),
                "metadata": {
                    "action_steps": [
                        "Publish AI transition policy standards",
                        "Fund mandatory reskilling by occupational family",
                        "Enable inter-agency mobility pathways",
                        "Report agency transition KPIs publicly",
                    ]
                },
            }
        ],
    },
    {
        "id": "AIJ-006",
        "title": "Regional Employment Shock Risk - Place-Based Transition Hubs",
        "category": "Regional Employment Shock Risk",
        "intervention_type": "safety_net",
        "causal_variable": "transition_friction",
        "chunks": [
            {
                "chunk_id": "AIJ-006-01",
                "text": (
                    "Where local labor markets face concentrated AI displacement, launch regional transition hubs "
                    "combining career coaching, rapid skilling, childcare support, and employer matching."
                ),
                "metadata": {
                    "action_steps": [
                        "Identify high-shock counties or metros",
                        "Open one-stop transition hubs",
                        "Provide wraparound supports for trainees",
                        "Integrate employers into matching pipeline",
                    ]
                },
            }
        ],
    },
    {
        "id": "AIJ-007",
        "title": "Inequality Amplification Risk - Inclusive Transition Design",
        "category": "Inequality Amplification Risk",
        "intervention_type": "policy_reform",
        "causal_variable": "transition_friction",
        "chunks": [
            {
                "chunk_id": "AIJ-007-01",
                "text": (
                    "Reduce inequality amplification by targeting transition resources to vulnerable cohorts: "
                    "entry-level workers, older workers, and underrepresented groups in high-automation sectors."
                ),
                "metadata": {
                    "action_steps": [
                        "Segment impact by demographic and job level",
                        "Target scholarships and transition stipends",
                        "Monitor placement equity metrics",
                        "Publish equity impact dashboard",
                    ]
                },
            }
        ],
    },
    {
        "id": "AIJ-008",
        "title": "Regulatory Lag Risk - Adaptive AI Labor Policy",
        "category": "Regulatory Lag Risk",
        "intervention_type": "policy_reform",
        "causal_variable": "policy_support",
        "chunks": [
            {
                "chunk_id": "AIJ-008-01",
                "text": (
                    "Create adaptive labor policy cycles for AI by updating occupational standards, reporting "
                    "requirements, and worker protection mechanisms every 6 to 12 months."
                ),
                "metadata": {
                    "action_steps": [
                        "Establish 6-12 month policy refresh cadence",
                        "Update occupational standards for AI-enabled work",
                        "Mandate employer transition disclosures",
                        "Review worker protection triggers quarterly",
                    ]
                },
            }
        ],
    },
    {
        "id": "AIJ-009",
        "title": "AI Governance and Trust Risk - Human Oversight Controls",
        "category": "AI Governance and Trust Risk",
        "intervention_type": "monitor",
        "causal_variable": "ai_adoption_rate",
        "chunks": [
            {
                "chunk_id": "AIJ-009-01",
                "text": (
                    "To avoid governance failures during rapid AI adoption, enforce human oversight checkpoints, "
                    "incident logging, and bias review gates before AI decisions affect employment outcomes."
                ),
                "metadata": {
                    "action_steps": [
                        "Define mandatory human review points",
                        "Log all AI-assisted hiring and evaluation decisions",
                        "Run periodic fairness and bias audits",
                        "Escalate high-risk model behavior within 24 hours",
                    ]
                },
            }
        ],
    },
    {
        "id": "AIJ-010",
        "title": "Education-Pipeline Mismatch Risk - Curriculum Realignment",
        "category": "Education-Pipeline Mismatch Risk",
        "intervention_type": "reskill",
        "causal_variable": "reskilling_capacity",
        "chunks": [
            {
                "chunk_id": "AIJ-010-01",
                "text": (
                    "Align education pipelines with AI-era demand by co-designing curricula with employers, "
                    "embedding practical tool fluency, and shortening credential cycles for fast-moving roles."
                ),
                "metadata": {
                    "action_steps": [
                        "Co-design curricula with hiring employers",
                        "Embed practical AI tool modules",
                        "Shorten credential cycle times",
                        "Track graduate placement in AI-complement jobs",
                    ]
                },
            }
        ],
    },
]


def get_all_chunks() -> List[PlaybookChunk]:
    """Flatten all playbook entries into a list of PlaybookChunk objects."""
    chunks: List[PlaybookChunk] = []
    for pb in RAW_PLAYBOOKS:
        for chunk_raw in pb["chunks"]:
            chunks.append(
                PlaybookChunk(
                    chunk_id=chunk_raw["chunk_id"],
                    playbook_id=pb["id"],
                    playbook_title=pb["title"],
                    category=pb["category"],
                    intervention_type=pb["intervention_type"],
                    text=chunk_raw["text"],
                    metadata=chunk_raw.get("metadata", {}),
                )
            )
    return chunks
