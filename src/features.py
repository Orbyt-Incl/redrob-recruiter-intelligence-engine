"""
Feature extraction -> features.parquet.

Two stages:
  1. extract_raw(candidate): per-candidate scalars needing no population statistics.
  2. build_features(path):   assemble the DataFrame, then add population-normalized
                             composites (recruiter demand percentiles, availability).
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config, honeypot
from .data_io import all_titles, career_text, full_text_lower, iter_candidates


def _parse_date(s) -> date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def _count_hits(text: str, keywords: List[str]) -> int:
    return sum(1 for k in keywords if k in text)


def _is_consulting_company(company: str, industry: str) -> bool:
    company = (company or "").lower()
    industry = (industry or "").lower()
    if any(f in company for f in config.CONSULTING_FIRMS):
        return True
    if any(i in industry for i in config.CONSULTING_INDUSTRIES):
        return True
    return False


def _experience_fit(yoe: float) -> float:
    """1.0 inside the ideal 6-8 band; tapering to 0 outside the acceptable 5-9 band."""
    lo_a, hi_a = config.EXP_ACCEPT_LOW, config.EXP_ACCEPT_HIGH
    lo_i, hi_i = config.EXP_IDEAL_LOW, config.EXP_IDEAL_HIGH
    if lo_i <= yoe <= hi_i:
        return 1.0
    if yoe < lo_i:
        # ramp from lo_a-2 (0) .. lo_i (1)
        return float(np.clip((yoe - (lo_a - 2.0)) / ((lo_i) - (lo_a - 2.0)), 0.0, 1.0))
    # yoe > hi_i: ramp down to 0 at hi_a+4
    return float(np.clip(1.0 - (yoe - hi_i) / ((hi_a + 4.0) - hi_i), 0.0, 1.0))


def extract_raw(candidate: dict, ref: date) -> Dict:
    prof = candidate.get("profile", {}) or {}
    sig = candidate.get("redrob_signals", {}) or {}
    roles = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    text = full_text_lower(candidate)
    evidence = career_text(candidate).lower()  # what they actually DID (titles + descriptions)
    yoe = float(prof.get("years_of_experience", 0) or 0)

    # --- tenure / job hopping ---
    completed = [r for r in roles if not r.get("is_current")]
    durations = [float(r.get("duration_months", 0) or 0) for r in roles]
    total_tenure = float(sum(durations))
    short_stints = sum(
        1 for r in completed if (r.get("duration_months", 0) or 0) < config.SHORT_STINT_MONTHS
    )
    n_completed = max(len(completed), 1)
    job_hop_ratio = short_stints / n_completed
    is_job_hopper = (len(completed) >= 3) and (job_hop_ratio > config.JOB_HOP_RATIO_PENALTY)

    # --- product vs consulting ---
    consulting_roles = sum(
        1 for r in roles if _is_consulting_company(r.get("company"), r.get("industry"))
    )
    # include current company/industry from profile too
    cur_consulting = _is_consulting_company(prof.get("current_company"), prof.get("current_industry"))
    n_roles = max(len(roles), 1)
    consulting_ratio = consulting_roles / n_roles
    is_consulting_only = (consulting_roles == len(roles) and len(roles) > 0) or (
        consulting_ratio >= 0.8 and cur_consulting
    )
    has_product_experience = consulting_roles < len(roles)

    # --- title signals ---
    titles = all_titles(candidate)
    cur_title = (prof.get("current_title", "") or "").lower()
    is_non_software_eng = any(k in cur_title for k in config.NON_SOFTWARE_ENGINEER_TITLES)
    is_engineering_title = (
        any(k in cur_title for k in config.ENGINEERING_TITLE_KEYWORDS) and not is_non_software_eng
    )
    is_non_eng_title = (
        is_non_software_eng
        or (any(k in cur_title for k in config.NON_ENGINEERING_TITLES) and not is_engineering_title)
    )

    # --- skills / AI-core ---
    skill_names_lc = [(s.get("name", "") or "").lower() for s in skills]
    ai_core_skill_count = sum(
        1 for n in skill_names_lc if any(k == n or k in n for k in config.AI_CORE_SKILL_NAMES)
    )
    has_ai_signal = ai_core_skill_count > 0 or _count_hits(text, config.CORE_AI_KEYWORDS) >= 2

    # keyword-stuffer trap: lots of AI on a non-engineering title
    title_mismatch = is_non_eng_title and has_ai_signal

    # skill trust: do claimed skills have endorsement+duration backing?
    trusts = []
    for s in skills:
        n = (s.get("name", "") or "").lower()
        if not any(k == n or k in n for k in config.AI_CORE_SKILL_NAMES):
            continue
        end = min(1.0, (s.get("endorsements", 0) or 0) / 10.0)
        dur = min(1.0, (s.get("duration_months", 0) or 0) / 24.0)
        trusts.append(0.5 * end + 0.5 * dur)
    skill_trust = float(np.mean(trusts)) if trusts else 0.0

    # --- domain: NLP/IR vs CV/speech/robotics ---
    cv_hits = _count_hits(text, config.CV_SPEECH_ROBOTICS_KEYWORDS)
    nlp_hits = _count_hits(text, config.NLP_IR_KEYWORDS)
    is_cv_speech_only = cv_hits >= 2 and nlp_hits == 0

    # --- research vs production (judged on what they DID, not stuffed skills) ---
    research_hits = _count_hits(evidence, config.RESEARCH_KEYWORDS)
    production_hits = _count_hits(evidence, config.PRODUCTION_KEYWORDS)
    is_research_only = research_hits >= 2 and production_hits == 0

    # --- framework-only (recent langchain, shallow ml) ---
    framework_hits = _count_hits(text, config.FRAMEWORK_ONLY_KEYWORDS)
    deep_ml = _count_hits(text, ["learning to rank", "ranking", "retrieval", "embedding",
                                 "recommendation", "production", "deployed"])
    framework_only = framework_hits >= 1 and deep_ml == 0 and yoe < 4

    # --- location ---
    loc = (prof.get("location", "") or "").lower()
    country = (prof.get("country", "") or "").lower()
    location_match = (
        any(c in loc for c in config.TARGET_LOCATIONS) or config.TARGET_COUNTRY in country
    )
    willing_to_relocate = bool(sig.get("willing_to_relocate", False))

    # --- behavioral / signals ---
    la = _parse_date(sig.get("last_active_date"))
    last_active_days = (ref - la).days if la else 365
    notice = int(sig.get("notice_period_days", 180) or 180)
    response_rate = float(sig.get("recruiter_response_rate", 0) or 0)
    interview_cr = float(sig.get("interview_completion_rate", 0) or 0)
    offer_ar = float(sig.get("offer_acceptance_rate", 0) or 0)  # -1 if none
    open_to_work = bool(sig.get("open_to_work_flag", False))
    github = float(sig.get("github_activity_score", -1) or -1)

    return {
        "candidate_id": candidate.get("candidate_id"),
        "yoe": yoe,
        "num_roles": len(roles),
        "total_tenure_months": total_tenure,
        "avg_tenure_months": total_tenure / n_roles,
        "short_stints": short_stints,
        "job_hop_ratio": round(job_hop_ratio, 3),
        "is_job_hopper": is_job_hopper,
        "consulting_ratio": round(consulting_ratio, 3),
        "is_consulting_only": is_consulting_only,
        "has_product_experience": has_product_experience,
        "is_engineering_title": is_engineering_title,
        "title_mismatch": title_mismatch,
        "ai_core_skill_count": ai_core_skill_count,
        "has_ai_signal": has_ai_signal,
        "skill_trust": round(skill_trust, 3),
        "cv_hits": cv_hits,
        "nlp_hits": nlp_hits,
        "is_cv_speech_only": is_cv_speech_only,
        "research_hits": research_hits,
        "production_hits": production_hits,
        "is_research_only": is_research_only,
        "framework_only": framework_only,
        "location_match": location_match,
        "willing_to_relocate": willing_to_relocate,
        "notice_period_days": notice,
        "last_active_days": last_active_days,
        "recruiter_response_rate": response_rate,
        "interview_completion_rate": interview_cr,
        "offer_acceptance_rate": offer_ar,
        "open_to_work": open_to_work,
        "github_activity_score": github,
        "saved_by_recruiters_30d": int(sig.get("saved_by_recruiters_30d", 0) or 0),
        "search_appearance_30d": int(sig.get("search_appearance_30d", 0) or 0),
        "profile_views_received_30d": int(sig.get("profile_views_received_30d", 0) or 0),
        "experience_fit": round(_experience_fit(yoe), 3),
        # current title/company for reasoning
        "current_title": prof.get("current_title", ""),
        "current_company": prof.get("current_company", ""),
        "location": prof.get("location", ""),
        "country": prof.get("country", ""),
    }


def _availability(row: pd.Series) -> float:
    """Composite hireability score in [0,1] from behavioral signals."""
    recency = float(np.clip(1.0 - row["last_active_days"] / 180.0, 0.0, 1.0))
    notice = float(np.clip(1.0 - row["notice_period_days"] / 180.0, 0.0, 1.0))
    resp = float(np.clip(row["recruiter_response_rate"], 0.0, 1.0))
    interview = float(np.clip(row["interview_completion_rate"], 0.0, 1.0))
    otw = 1.0 if row["open_to_work"] else 0.4
    return float(
        0.30 * recency + 0.25 * resp + 0.20 * otw + 0.15 * notice + 0.10 * interview
    )


def build_features(
    path: str = config.CANDIDATES_PATH,
    reference_date: str | None = None,
    records=None,
) -> pd.DataFrame:
    """Build features from a jsonl path, or from an in-memory iterable of records."""
    ref = _parse_date(reference_date or config.REFERENCE_DATE) or date(2026, 6, 14)
    source = records if records is not None else iter_candidates(path)

    rows: List[Dict] = []
    for cand in source:
        raw = extract_raw(cand, ref)
        is_hp, hp_score, hp_reasons = honeypot.detect(cand, reference_date)
        raw["is_honeypot"] = is_hp
        raw["honeypot_score"] = hp_score
        raw["honeypot_reasons"] = "; ".join(hp_reasons[:3])
        rows.append(raw)

    df = pd.DataFrame(rows)

    # --- population-normalized recruiter-demand composite (log + percentile rank) ---
    def _pct_log(col: str) -> pd.Series:
        v = np.log1p(df[col].clip(lower=0).astype(float))
        return v.rank(pct=True)

    demand = (
        0.45 * _pct_log("saved_by_recruiters_30d")
        + 0.35 * _pct_log("search_appearance_30d")
        + 0.20 * _pct_log("profile_views_received_30d")
    )
    df["recruiter_demand"] = demand.round(4)
    df["availability"] = df.apply(_availability, axis=1).round(4)

    return df


def write_features(df: pd.DataFrame, path: str = config.FEATURES_PARQUET) -> None:
    df.to_parquet(path, index=False)
