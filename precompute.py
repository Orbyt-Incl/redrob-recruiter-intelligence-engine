#!/usr/bin/env python3
"""
Phase 1 — Precompute (offline, unlimited time).

Reads candidates.jsonl and produces all ranking artifacts under artifacts/:
    jd_requirements.json
    features.parquet           (+ honeypot flags)
    capability_scores.parquet
    candidate_embeddings.faiss
    id_index.parquet
    jd_embedding.npy, anchor_embeddings.npy
    embedding_model/           (local copy of the sentence-transformer)

Usage:
    python precompute.py [--candidates ./candidates.jsonl] [--no-embeddings]
"""
from __future__ import annotations

import argparse
import os
import time

from src import capability, config, features, jd_spec


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=config.CANDIDATES_PATH)
    ap.add_argument("--no-embeddings", action="store_true",
                    help="Skip the FAISS/embedding step (keyword-only capabilities).")
    args = ap.parse_args()

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    t0 = time.time()

    # 1. JD requirements
    req = jd_spec.write_jd_requirements()
    _log(f"Wrote jd_requirements.json ({len(req['must_haves'])} must-haves)", t0)

    # 2. Features + honeypots
    feat_df = features.build_features(args.candidates)
    features.write_features(feat_df)
    n_hp = int(feat_df["is_honeypot"].sum())
    _log(f"Wrote features.parquet: {len(feat_df)} candidates, {n_hp} honeypots flagged", t0)

    # 3. Keyword capabilities
    kw_df = capability.build_keyword_capabilities(args.candidates)
    _log("Built keyword capability scores", t0)

    # 4. Embeddings + FAISS + semantic capabilities
    sem_df = None
    if not args.no_embeddings:
        from src import embeddings
        sem_df, _jd_emb, _names, _anchors = embeddings.build_embeddings(
            args.candidates, query_text=req["query_text"]
        )
        _log("Built FAISS index + semantic capability scores", t0)
    else:
        _log("Skipped embeddings (--no-embeddings)", t0)

    # 5. Combine capability scores
    cap_df = capability.combine_capabilities(kw_df, sem_df)
    capability.write_capabilities(cap_df)
    _log(f"Wrote capability_scores.parquet (mean fit={cap_df['capability_fit'].mean():.3f})", t0)

    _log("Precompute complete.", t0)


if __name__ == "__main__":
    main()
