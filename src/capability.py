"""
Capability scoring -> capability_scores.parquet.

Per candidate we produce a score in [0,1] for each capability axis, combining:
  * keyword/evidence rules (capability_graph.match_keyword_capabilities)
  * semantic anchor similarity (cosine of candidate embedding vs each capability anchor)

Combined as a saturating max-ish blend, then aggregated into `capability_fit` weighted by
the JD's importance for each capability.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import config
from .capability_graph import (
    CAPABILITIES, JD_CAPABILITY_IMPORTANCE, match_keyword_capabilities,
)
from .data_io import evidence_text, iter_candidates


def build_keyword_capabilities(path: str = config.CANDIDATES_PATH, records=None) -> pd.DataFrame:
    """
    Keyword/evidence-rule capability scores per candidate (from path or records).

    Matches against career-history descriptions only (evidence_text) — what the candidate
    actually did — so stuffed skills lists cannot fabricate capabilities.
    """
    rows: List[Dict] = []
    source = records if records is not None else iter_candidates(path)
    for cand in source:
        caps = match_keyword_capabilities(evidence_text(cand))
        row = {"candidate_id": cand.get("candidate_id")}
        for c in CAPABILITIES:
            row[f"kw_{c}"] = caps[c]
        rows.append(row)
    return pd.DataFrame(rows)


def combine_capabilities(kw_df: pd.DataFrame, sem_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge keyword (career-evidence) and semantic capability scores into final cap_<axis>
    columns plus a weighted `capability_fit` aggregate.

    Keyword career evidence is the PRIMARY, discriminative signal. The embedding model's
    anchor similarities are heavily compressed (tech-adjacent traps score nearly as high as
    real builders), so the semantic layer only contributes a small, top-tail boost: sem_df
    carries per-capability POPULATION PERCENTILES, and only the top decile (>SEM_PCT_FLOOR)
    adds anything. This recovers genuine keyword-free "hidden gems" without floating traps up.

    sem_df (optional): candidate_id + sem_<capability> columns of percentile in [0,1].
    """
    df = kw_df.copy()
    if sem_df is not None:
        df = df.merge(sem_df, on="candidate_id", how="left")

    weights = JD_CAPABILITY_IMPORTANCE
    wsum = sum(weights.values())

    fit = np.zeros(len(df), dtype=float)
    for c in CAPABILITIES:
        kw = df[f"kw_{c}"].fillna(0.0).to_numpy()
        if sem_df is not None and f"sem_{c}" in df.columns:
            sem_pct = df[f"sem_{c}"].fillna(0.0).to_numpy()
            boost = np.clip((sem_pct - config.SEM_PCT_FLOOR) / (1.0 - config.SEM_PCT_FLOOR), 0.0, 1.0)
            combined = 1.0 - (1.0 - kw) * (1.0 - config.SEM_BLEND * boost)
        else:
            combined = kw
        df[f"cap_{c}"] = np.round(combined, 4)
        fit += weights.get(c, 0.0) * combined

    df["capability_fit"] = np.round(fit / wsum, 4)
    keep = ["candidate_id", "capability_fit"] + [f"cap_{c}" for c in CAPABILITIES]
    return df[keep]


def write_capabilities(df: pd.DataFrame, path: str = config.CAPABILITY_PARQUET) -> None:
    df.to_parquet(path, index=False)
