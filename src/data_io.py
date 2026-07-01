"""
Loading and text-extraction helpers for candidate records.

Handles both plain `.jsonl` and gzipped `.jsonl.gz`. Streaming iteration keeps memory
flat over the 100k / ~465 MB pool.
"""
from __future__ import annotations

import gzip
import io
import json
from typing import Dict, Iterator, List


def open_candidates(path: str) -> io.TextIOBase:
    """Open a candidates file transparently whether it is gzipped or plain text."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str) -> Iterator[dict]:
    """Yield one candidate dict per line. Skips blank lines."""
    with open_candidates(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates(path: str) -> List[dict]:
    """Load all candidates into a list (use only when full materialization is acceptable)."""
    return list(iter_candidates(path))


# --------------------------------------------------------------------------------------
# Text assembly — used for both embeddings and keyword/capability matching.
# --------------------------------------------------------------------------------------

def career_text(candidate: dict) -> str:
    """Concatenate all career-history titles + descriptions."""
    parts: List[str] = []
    for role in candidate.get("career_history", []) or []:
        title = role.get("title", "") or ""
        desc = role.get("description", "") or ""
        company = role.get("company", "") or ""
        parts.append(f"{title} at {company}. {desc}")
    return "\n".join(parts)


def skills_text(candidate: dict) -> str:
    """Space-joined skill names."""
    return " ".join((s.get("name", "") or "") for s in candidate.get("skills", []) or [])


def profile_document(candidate: dict) -> str:
    """
    Canonical text "document" for EMBEDDING / retrieval.

    Deliberately EXCLUDES the raw skills list: keyword-stuffer traps pack the skills array
    with AI tokens their actual work never touched, which would inflate semantic similarity.
    We embed what a recruiter reads as narrative — headline, summary, and career history.
    """
    prof = candidate.get("profile", {}) or {}
    segments = [
        prof.get("headline", "") or "",
        prof.get("current_title", "") or "",
        prof.get("summary", "") or "",
        career_text(candidate),
    ]
    return "\n".join(s for s in segments if s.strip())


def evidence_text(candidate: dict) -> str:
    """
    Text used for CAPABILITY inference: career-history titles + descriptions only.

    This is *what the candidate actually did*. Capability keywords are matched here rather
    than against the skills list or summary, so a stuffed skills array cannot manufacture
    capabilities the career history does not support. Lowercased.
    """
    return career_text(candidate).lower()


def full_text_lower(candidate: dict) -> str:
    """Lowercased blob of everything textual — for substring keyword matching."""
    prof = candidate.get("profile", {}) or {}
    blob = " ".join([
        prof.get("headline", "") or "",
        prof.get("summary", "") or "",
        prof.get("current_title", "") or "",
        prof.get("current_industry", "") or "",
        career_text(candidate),
        skills_text(candidate),
    ])
    return blob.lower()


def all_titles(candidate: dict) -> List[str]:
    """Current title plus every career-history title, lowercased."""
    prof = candidate.get("profile", {}) or {}
    titles = [(prof.get("current_title", "") or "").lower()]
    for role in candidate.get("career_history", []) or []:
        titles.append((role.get("title", "") or "").lower())
    return [t for t in titles if t]


def all_companies(candidate: dict) -> List[str]:
    """Current company plus every career-history company, lowercased."""
    prof = candidate.get("profile", {}) or {}
    comps = [(prof.get("current_company", "") or "").lower()]
    for role in candidate.get("career_history", []) or []:
        comps.append((role.get("company", "") or "").lower())
    return [c for c in comps if c]
