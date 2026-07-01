"""
Job-description spec.

The JD ("Senior AI Engineer — Founding Team") is fixed and known, so we encode it as a
structured object rather than parsing prose at runtime. Two products:

  * jd_requirements.json  — the structured signals consumed by scoring/reasoning.
  * JD_QUERY_TEXT         — a focused, noise-free "ideal candidate" description used as the
                            retrieval query (the real JD is full of culture prose that would
                            pollute an embedding).
"""
from __future__ import annotations

import json
from typing import Dict

from . import config
from .capability_graph import JD_CAPABILITY_IMPORTANCE

# Curated retrieval query: the technical essence of who they want. Used for FAISS retrieval.
JD_QUERY_TEXT = (
    "Senior AI engineer who has built and deployed embeddings-based retrieval, semantic "
    "search, ranking, and recommendation systems to real users in production at a product "
    "company. Strong Python. Experience with vector databases and hybrid search (FAISS, "
    "Pinecone, Elasticsearch, OpenSearch). Designs evaluation frameworks for ranking using "
    "NDCG, MRR, MAP and online A/B testing. Has shipped end-to-end search, ranking, or "
    "recommendation systems at meaningful scale. 6 to 8 years of applied machine-learning "
    "experience. Tilts toward shipping production systems rather than pure research."
)


def build_jd_requirements() -> Dict:
    """Build the structured JD requirements object."""
    return {
        "title": "Senior AI Engineer - Founding Team",
        "company": "Redrob AI",
        "query_text": JD_QUERY_TEXT,
        "experience": {
            "accept_low": config.EXP_ACCEPT_LOW,
            "accept_high": config.EXP_ACCEPT_HIGH,
            "ideal_low": config.EXP_IDEAL_LOW,
            "ideal_high": config.EXP_IDEAL_HIGH,
        },
        "capability_importance": JD_CAPABILITY_IMPORTANCE,
        "must_haves": [
            "embeddings-based retrieval in production",
            "vector database / hybrid search infrastructure",
            "strong Python",
            "ranking evaluation frameworks (NDCG, MRR, MAP, A/B testing)",
        ],
        "nice_to_haves": [
            "LLM fine-tuning (LoRA, QLoRA, PEFT)",
            "learning-to-rank models",
            "HR-tech / recruiting / marketplace products",
            "distributed systems / large-scale inference",
            "open-source contributions",
        ],
        "anti_signals": {
            "title_chaser_job_hopping": True,
            "framework_only_langchain": True,
            "consulting_only_career": True,
            "cv_speech_robotics_only": True,
            "research_only_no_production": True,
            "closed_source_only": True,
        },
        "location": {
            "target_cities": config.TARGET_LOCATIONS,
            "target_country": config.TARGET_COUNTRY,
            "relocation_acceptable": True,
        },
        "notice_period": {
            "ideal_max_days": config.NOTICE_IDEAL_DAYS,
            "buyout_max_days": 30,
        },
        "behavioral_note": (
            "Down-weight perfect-on-paper candidates who are inactive or unresponsive; "
            "availability and engagement matter for hireability."
        ),
    }


def write_jd_requirements(path: str = config.JD_REQUIREMENTS_JSON) -> Dict:
    req = build_jd_requirements()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(req, f, indent=2)
    return req
