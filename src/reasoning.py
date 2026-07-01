"""
Reasoning generation — factual, offline, no LLM.

Stage-4 manual review checks each reasoning for: specific facts, JD connection, honest
concerns, NO hallucination, variation across rows, and tone matching the rank. We build each
string purely from fields that exist in the candidate's own record, vary phrasing by a
per-candidate seed, and attach the single most relevant honest concern.
"""
from __future__ import annotations

import pandas as pd

from .capability_graph import CAPABILITIES

# capability key -> human phrase
_CAP_PHRASE = {
    "retrieval": "retrieval",
    "ranking": "ranking",
    "recommendation": "recommendation systems",
    "embeddings": "embeddings",
    "search_ir": "search/IR",
    "evaluation": "ranking evaluation",
    "production_ml": "production ML",
    "nlp": "NLP",
    "llm": "LLMs",
}

_LEAD = ["{title} with {y} experience", "{title}, {y} of experience", "{y} as a {title}"]
_STRENGTH = [
    "career shows evidence of {caps}",
    "demonstrated {caps} in past roles",
    "background covers {caps}",
]


def _years(yoe: float) -> str:
    return f"{yoe:.1f} yrs" if yoe and yoe > 0 else "unspecified experience"


def _top_caps(row: pd.Series, n: int = 2, thresh: float = 0.45) -> list[str]:
    scored = [(c, float(row.get(f"cap_{c}", 0.0))) for c in CAPABILITIES]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [_CAP_PHRASE[c] for c, s in scored[:n] if s >= thresh]


def _seed(candidate_id: str) -> int:
    digits = "".join(ch for ch in str(candidate_id) if ch.isdigit())
    return int(digits) if digits else 0


def _concern(row: pd.Series) -> str | None:
    """The single most material honest concern, if any."""
    if row.get("is_consulting_only"):
        return "career is entirely at IT-services/consulting firms"
    if row.get("title_mismatch"):
        return f"current title ({row.get('current_title')}) is outside engineering despite AI skills"
    if row.get("is_research_only"):
        return "profile reads research-only with little production signal"
    if row.get("is_cv_speech_only"):
        return "domain is CV/speech rather than NLP/IR"
    if row.get("is_job_hopper"):
        return "frequent short stints suggest job-hopping"
    notice = int(row.get("notice_period_days", 0) or 0)
    if notice > 60:
        return f"{notice}-day notice period"
    if int(row.get("last_active_days", 0) or 0) > 90:
        return f"inactive ~{int(row['last_active_days'])} days"
    rr = float(row.get("recruiter_response_rate", 0) or 0)
    if rr < 0.2:
        return f"low recruiter response rate ({rr:.0%})"
    if not row.get("location_match") and not row.get("willing_to_relocate"):
        loc = row.get("location") or "outside target geography"
        return f"based in {loc}, not open to relocation"
    return None


def generate(row: pd.Series, rank: int) -> str:
    seed = _seed(row.get("candidate_id", ""))
    yoe = float(row.get("yoe", 0) or 0)
    title = (row.get("current_title") or "candidate").strip()
    company = (row.get("current_company") or "").strip()

    lead = _LEAD[seed % len(_LEAD)].format(y=_years(yoe), title=title)
    if company:
        lead += f" at {company}"

    caps = _top_caps(row)
    parts = [lead]
    if caps:
        parts.append(_STRENGTH[seed % len(_STRENGTH)].format(caps=" and ".join(caps)))

    # supporting positive facts
    pos = []
    if row.get("has_product_experience"):
        pos.append("product-company experience")
    rr = float(row.get("recruiter_response_rate", 0) or 0)
    if rr >= 0.5:
        pos.append(f"strong recruiter response ({rr:.0%})")
    if int(row.get("last_active_days", 999) or 999) <= 30:
        pos.append("recently active")
    n_ai = int(row.get("ai_core_skill_count", 0) or 0)
    if n_ai >= 3 and rank <= 60:
        pos.append(f"{n_ai} core AI skills")
    if pos:
        parts.append(", ".join(pos[:2]))

    text = "; ".join(parts) + "."

    concern = _concern(row)
    if concern:
        # tone: lower ranks lead harder with the concern
        if rank > 70:
            text = text[:-1] + f" — but {concern}."
        else:
            text = text[:-1] + f". Minor concern: {concern}."

    # rank-tone framing at the extremes
    if rank > 85 and not concern:
        text = text[:-1] + "; adjacent fit, included near the cutoff."

    return " ".join(text.split())  # collapse whitespace, single line


def add_reasoning(df: pd.DataFrame) -> pd.DataFrame:
    """df must be sorted in final rank order with a 'rank' column."""
    df = df.copy()
    df["reasoning"] = [generate(row, int(row["rank"])) for _, row in df.iterrows()]
    return df
