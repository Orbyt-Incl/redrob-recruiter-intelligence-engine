"""
Recruiter Intelligence Score (RIS).

Operates on the shortlist DataFrame (features + capability columns + a `jd_cosine` column
from FAISS retrieval). Produces a `score` in (0,1] plus the component breakdown (kept for
reasoning generation and evaluation).

Blend = weighted positive components MINUS soft penalties. Honeypots are forced to the floor
(the only hard filter). Everything else is a soft penalty, never an exclusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _pct(series: pd.Series) -> pd.Series:
    """Percentile rank within the shortlist (robust to embedding scale)."""
    return series.rank(pct=True)


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if n == 0:
        df["score"] = []
        return df

    # ---- positive components (each in [0,1]) ----
    jd_sim = _pct(df["jd_cosine"]) if "jd_cosine" in df else pd.Series(np.zeros(n), index=df.index)
    # Keyword career capability is the discriminative signal; JD-cosine percentile is a weak
    # secondary nudge (the embedding similarities are compressed and float traps upward).
    cap_fit = df["capability_fit"].fillna(0.0).clip(0, 1)
    capability_component = (0.85 * cap_fit + 0.15 * jd_sim).clip(0, 1)

    # Career evidence is demonstrated capability in real jobs. Product-company credit is GATED
    # on the candidate actually showing capability in their career history, so a keyword-stuffer
    # at a product company earns no free "career evidence".
    cap_present = (cap_fit >= 0.20).astype(float)
    product_gated = df["has_product_experience"].astype(float) * cap_present
    career_evidence = (
        0.55 * cap_fit
        + 0.25 * (df["production_hits"].clip(0, 3) / 3.0)
        + 0.20 * product_gated
    ).clip(0, 1)

    availability = df["availability"].fillna(0.0).clip(0, 1)
    recruiter_demand = df["recruiter_demand"].fillna(0.0).clip(0, 1)
    experience_fit = df["experience_fit"].fillna(0.0).clip(0, 1)

    w = config.RIS_WEIGHTS
    base = (
        w["capability_fit"] * capability_component
        + w["career_evidence"] * career_evidence
        + w["availability"] * availability
        + w["recruiter_demand"] * recruiter_demand
        + w["experience_fit"] * experience_fit
    )

    # ---- soft penalties ----
    p = config.RIS_PENALTIES
    notice_excess = ((df["notice_period_days"] - config.NOTICE_IDEAL_DAYS).clip(lower=0) / 150.0).clip(0, 1)
    location_miss = (~df["location_match"] & ~df["willing_to_relocate"]).astype(float)

    penalty = (
        p["job_hopping"] * df["is_job_hopper"].astype(float)
        + p["consulting_only"] * df["is_consulting_only"].astype(float)
        + p["research_only"] * df["is_research_only"].astype(float)
        + p["cv_speech_only"] * df["is_cv_speech_only"].astype(float)
        + p["framework_only"] * df["framework_only"].astype(float)
        + p["title_mismatch"] * df["title_mismatch"].astype(float)
        + p["location_miss"] * location_miss
        + p["notice_long"] * notice_excess
    )

    score = (base - penalty).clip(lower=0.0)

    # ---- honeypots: hard filter to the floor ----
    score = score.where(~df["is_honeypot"], other=config.HONEYPOT_SCORE_FLOOR)

    # normalize positives into (0,1]; keep honeypots pinned at the floor
    max_s = float(score[~df["is_honeypot"]].max()) if (~df["is_honeypot"]).any() else 1.0
    if max_s > 0:
        norm = (score / max_s).clip(0, 1)
        score = norm.where(~df["is_honeypot"], other=config.HONEYPOT_SCORE_FLOOR)

    df["capability_component"] = capability_component.round(4)
    df["career_evidence"] = career_evidence.round(4)
    df["availability_component"] = availability.round(4)
    df["recruiter_demand_component"] = recruiter_demand.round(4)
    df["penalty_total"] = penalty.round(4)
    df["score"] = score.round(6)
    return df
