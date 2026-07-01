"""
Honeypot detection — the system's ONLY hard filter.

The dataset hides ~80 candidates with subtly *impossible* profiles (forced to relevance
tier 0). Examples from the spec: "8 years at a company founded 3 years ago", "expert in 10
skills with 0 months used", "1 year experience but 10 years skill usage".

A false positive here *removes a real candidate*, so we favor PRECISION: flag only on
near-certain impossibilities, mostly internal arithmetic contradictions.
"""
from __future__ import annotations

from datetime import date
from typing import List, Tuple

from . import config


def _parse_date(s) -> date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def _months_between(a: date, b: date) -> float:
    return (b.year - a.year) * 12 + (b.month - a.month) + (b.day - a.day) / 30.0


def detect(candidate: dict, reference_date: str | None = None) -> Tuple[bool, float, List[str]]:
    """
    Returns (is_honeypot, score, reasons).

    score accumulates evidence weight; is_honeypot is True if any HARD contradiction fires
    or the cumulative score crosses a threshold. Reasons are human-readable for debugging.
    """
    ref = _parse_date(reference_date or config.REFERENCE_DATE) or date(2026, 6, 14)
    reasons: List[str] = []
    score = 0.0

    prof = candidate.get("profile", {}) or {}
    yoe = float(prof.get("years_of_experience", 0) or 0)
    yoe_months = yoe * 12.0

    # The designed honeypots are EGREGIOUS ("1yr exp, 10yr skill usage"; "expert in 10 skills,
    # 0 months used"). The synthetic data also has mild internal noise (a skill listed a bit
    # longer than tenure) that is NOT a honeypot. So we weight by severity and require the
    # cumulative score to clear HONEYPOT_THRESHOLD (multiple or one egregious signal).

    # --- Skills: usage far exceeding the whole career ---
    skills = candidate.get("skills", []) or []
    if yoe_months > 0:
        excesses = [
            (s.get("name"), (s.get("duration_months", 0) or 0) - yoe_months)
            for s in skills if s.get("duration_months") is not None
        ]
        big = [(n, e) for n, e in excesses if e >= 48]      # 4+ yrs beyond career: egregious
        mild = [(n, e) for n, e in excesses if 18 <= e < 48]  # mild: synthetic noise
        if big:
            score += 1.5
            n, e = max(big, key=lambda x: x[1])
            reasons.append(f"skill '{n}' used {e + yoe_months:.0f}mo >> career {yoe_months:.0f}mo")
        if len(mild) >= 3:                                   # several mild excesses together
            score += 1.0
            reasons.append(f"{len(mild)} skills used longer than the whole career")

    # "expert" with zero usage time (the spec's canonical honeypot)
    expert_zero = sum(
        1 for s in skills
        if (s.get("proficiency") == "expert") and (s.get("duration_months", 0) or 0) == 0
    )
    if expert_zero >= 5:
        score += 1.5
        reasons.append(f"{expert_zero} 'expert' skills with 0 months used")
    elif expert_zero >= 3:
        score += 0.75
        reasons.append(f"{expert_zero} 'expert' skills with 0 months used")

    # --- Career history arithmetic ---
    roles = candidate.get("career_history", []) or []
    total_role_months = 0.0
    for r in roles:
        dm = r.get("duration_months", 0) or 0
        total_role_months += dm
        if yoe_months > 0 and dm > yoe_months + 24:          # a single role >2yr beyond career
            score += 1.5
            reasons.append(f"role '{r.get('title')}' {dm}mo >> career {yoe_months:.0f}mo")
        sd = _parse_date(r.get("start_date"))
        ed = _parse_date(r.get("end_date"))
        if sd and ed and sd > ed:                            # hard logical contradiction
            score += 1.5
            reasons.append(f"role '{r.get('title')}' start>end")
        if sd and sd > ref:                                  # role starts in the future
            score += 1.5
            reasons.append(f"role '{r.get('title')}' starts in the future")

    # Serial tenure far exceeding total experience (allow genuine overlap)
    if yoe_months > 0 and total_role_months > yoe_months * 2.0 + 24:
        score += 1.0
        reasons.append(f"summed tenure {total_role_months:.0f}mo >> career {yoe_months:.0f}mo")

    # Claims far more experience than the earliest job allows
    starts = [d for d in (_parse_date(r.get("start_date")) for r in roles) if d]
    if starts and yoe > 0:
        earliest = min(starts)
        career_span_years = (ref.year - earliest.year) + (ref.month - earliest.month) / 12.0
        if yoe > career_span_years + 4.0:
            score += 1.5
            reasons.append(f"claims {yoe:.1f}y but earliest job only {career_span_years:.1f}y ago")

    # --- Education arithmetic (hard contradiction) ---
    for e in candidate.get("education", []) or []:
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            score += 1.5
            reasons.append("education end_year < start_year")

    is_honeypot = score >= config.HONEYPOT_THRESHOLD
    return is_honeypot, round(score, 3), reasons
