#!/usr/bin/env python3
"""
Proxy evaluation — there is NO public ground truth, so this is a sanity/tuning harness,
clearly labelled as a proxy.

It builds a heuristic relevance label from the JD's *explicit* definition of a good
candidate (a different blend + hard caps than the ranker, to reduce circularity), reports
NDCG@10/50, MAP, P@10 of a submission against it, and runs TRAP REGRESSION checks that
encode the dataset's known failure modes:

  * 0 honeypots in the top 100 (hard DQ in the real eval).
  * keyword-stuffers with non-engineering titles do not flood the top.
  * the top is dominated by product-company experience (not consulting-only).
  * "hidden gems" (strong career capability, few literal AI keywords) reach the top.

Usage:
    python -m eval.proxy_eval [--submission submission.csv]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config


# ----------------------------- metrics -----------------------------
def dcg(rels: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(ranked_rels: np.ndarray, all_rels: np.ndarray, k: int) -> float:
    ideal = np.sort(all_rels)[::-1][:k]
    idcg = dcg(ideal)
    return dcg(ranked_rels[:k]) / idcg if idcg > 0 else 0.0


def average_precision(ranked_rel_bool: np.ndarray, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    hits, ap = 0, 0.0
    for i, r in enumerate(ranked_rel_bool, start=1):
        if r:
            hits += 1
            ap += hits / i
    return ap / min(total_relevant, len(ranked_rel_bool))


# ----------------------------- proxy relevance -----------------------------
def proxy_relevance(feat: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    df = feat.merge(cap, on="candidate_id", how="left").fillna(0.0)
    raw = (
        1.5 * df["has_product_experience"].astype(float)
        + 2.0 * df["capability_fit"]
        + 1.0 * df["availability"]
        + 0.5 * df["experience_fit"]
        - 2.0 * df["title_mismatch"].astype(float)
        - 1.5 * df["is_consulting_only"].astype(float)
        - 1.0 * df["is_research_only"].astype(float)
        - 1.0 * df["is_cv_speech_only"].astype(float)
    )
    raw = raw.where(~df["is_honeypot"], other=-1e9)

    # map to 0..4 tiers by quantile (honeypots pinned to 0)
    valid = raw[raw > -1e8]
    q = valid.quantile([0.75, 0.90, 0.97, 0.995]).values  # tiers 1,2,3,4 cutoffs
    tier = np.zeros(len(df), dtype=int)
    tier = np.where(raw >= q[0], 1, tier)
    tier = np.where(raw >= q[1], 2, tier)
    tier = np.where(raw >= q[2], 3, tier)
    tier = np.where(raw >= q[3], 4, tier)
    tier = np.where(df["is_honeypot"].values, 0, tier)
    df["proxy_tier"] = tier
    return df[["candidate_id", "proxy_tier"]]


# ----------------------------- evaluation -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission.csv")
    args = ap.parse_args()

    feat = pd.read_parquet(config.FEATURES_PARQUET)
    cap = pd.read_parquet(config.CAPABILITY_PARQUET)
    rel = proxy_relevance(feat, cap)
    sub = pd.read_csv(args.submission).sort_values("rank")
    sub = sub.merge(rel, on="candidate_id", how="left").merge(feat, on="candidate_id", how="left")

    all_rels = rel["proxy_tier"].to_numpy().astype(float)
    ranked_rels = sub["proxy_tier"].to_numpy().astype(float)
    total_relevant = int((all_rels >= 3).sum())
    ranked_rel_bool = ranked_rels >= 3

    print("=== PROXY METRICS (heuristic label — NOT ground truth) ===")
    print(f"  NDCG@10 : {ndcg_at_k(ranked_rels, all_rels, 10):.4f}")
    print(f"  NDCG@50 : {ndcg_at_k(ranked_rels, all_rels, 50):.4f}")
    print(f"  MAP@100 : {average_precision(ranked_rel_bool, total_relevant):.4f}")
    print(f"  P@10    : {ranked_rel_bool[:10].mean():.4f}")

    print("\n=== TRAP REGRESSION CHECKS ===")
    checks = []
    hp = int(sub["is_honeypot"].sum())
    checks.append((f"0 honeypots in top-100 (found {hp})", hp == 0))
    tm = int(sub["title_mismatch"].sum())
    checks.append((f"<=2 title-mismatch keyword-stuffers in top-100 (found {tm})", tm <= 2))
    prod = float(sub["has_product_experience"].mean())
    checks.append((f">=70% top-100 have product experience (got {prod:.0%})", prod >= 0.70))
    cons = float(sub["is_consulting_only"].mean())
    checks.append((f"<=15% top-100 consulting-only (got {cons:.0%})", cons <= 0.15))
    # hidden gem: strong career capability but few literal AI skills, present in top-100
    gems = sub[(sub["capability_fit"] >= 0.5) & (sub["ai_core_skill_count"] <= 2)]
    checks.append((f">=3 hidden gems in top-100 (found {len(gems)})", len(gems) >= 3))

    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    n_fail = sum(1 for _, ok in checks if not ok)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed.")


if __name__ == "__main__":
    main()
