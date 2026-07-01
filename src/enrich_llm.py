"""
OPTIONAL LLM enrichment — PRECOMPUTE ONLY (never imported by rank.py).

Runs a Claude pass over the top-N shortlisted candidates to (a) confirm/extend inferred
capabilities and (b) flag profiles that look impossible or stuffed. Results are cached to
artifacts/llm_enrichment.parquet. The deterministic core works fully without this; it is a
quality booster, not a dependency.

Requires:  pip install anthropic  and  ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
from typing import List

import pandas as pd

from . import config
from .data_io import profile_document

MODEL = "claude-haiku-4-5"  # cost-effective for bulk enrichment; swap to claude-opus-4-8 for quality

_SYSTEM = (
    "You are an expert technical recruiter screening for a Senior AI Engineer role focused on "
    "embeddings, retrieval, ranking, recommendation, and production ML. For each candidate, judge "
    "what they can actually DO based on career evidence (not just listed keywords), and whether the "
    "profile looks internally impossible. Respond ONLY with compact JSON."
)


def _prompt(candidate: dict) -> str:
    return (
        "Candidate profile:\n"
        + profile_document(candidate)[:4000]
        + "\n\nReturn JSON: {\"capabilities\":[...up to 6 short tags...], "
        "\"production_evidence\":true|false, \"suspicious\":true|false, "
        "\"fit_0_to_1\":float, \"one_line\":\"...\"}"
    )


def enrich(candidates: List[dict], model: str = MODEL) -> pd.DataFrame:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rows = []
    for cand in candidates:
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=300,
                system=_SYSTEM,
                messages=[{"role": "user", "content": _prompt(cand)}],
            )
            data = json.loads(msg.content[0].text)
        except Exception as e:  # enrichment is best-effort
            data = {"capabilities": [], "production_evidence": None,
                    "suspicious": None, "fit_0_to_1": None, "one_line": f"error: {e}"}
        data["candidate_id"] = cand.get("candidate_id")
        rows.append(data)
    return pd.DataFrame(rows)


def write_enrichment(df: pd.DataFrame, path: str = config.LLM_ENRICHMENT_PARQUET) -> None:
    df.to_parquet(path, index=False)
