#!/usr/bin/env python3
"""
Phase 2 — Ranking (CPU only, no network, < 5 minutes).

Loads precomputed artifacts, retrieves a shortlist via FAISS, scores with the Recruiter
Intelligence Score, generates factual reasoning, and writes the top-100 submission CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src import config, reasoning, retrieval, scoring


def build_submission() -> pd.DataFrame:
    feat = pd.read_parquet(config.FEATURES_PARQUET)
    cap = pd.read_parquet(config.CAPABILITY_PARQUET)

    index, ids, jd_emb = retrieval.load_artifacts()
    sl = retrieval.shortlist(index, ids, jd_emb, cap)

    df = sl.merge(feat, on="candidate_id", how="left").merge(cap, on="candidate_id", how="left")
    df = scoring.compute_scores(df)

    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    top = df.head(config.TOP_K_OUTPUT).copy()
    top["rank"] = range(1, len(top) + 1)
    top = reasoning.add_reasoning(top)
    return top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=config.CANDIDATES_PATH,
                    help="Path to candidates.jsonl (artifacts are precomputed from it).")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--validate", action="store_true", help="Run validate_submission.py after writing.")
    args = ap.parse_args()

    t0 = time.time()
    top = build_submission()
    out = top[["candidate_id", "rank", "score", "reasoning"]]
    out.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Wrote {args.out}: {len(out)} rows in {time.time() - t0:.1f}s", flush=True)

    if args.validate:
        import subprocess
        import sys
        subprocess.run([sys.executable, "validate_submission.py", args.out], check=False)


if __name__ == "__main__":
    main()
