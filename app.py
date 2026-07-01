#!/usr/bin/env python3
"""
Gradio demo (sandbox deliverable).

Upload a small candidate sample (<=100, JSON array or JSONL); the app runs the full ranking
pipeline end-to-end on CPU and shows the ranked table with scores, reasoning, and
hidden-gem / honeypot flags. Mirrors rank.py but self-contained for small inputs (direct
cosine instead of a prebuilt FAISS index).

Run locally:   python app.py
Deploy:        HuggingFace Spaces / Streamlit Cloud / Replit (see README).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import capability, config, embeddings, features, jd_spec, reasoning, scoring

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = embeddings.get_model()
    return _MODEL


def _parse_upload(file_obj) -> list[dict]:
    if file_obj is None:
        # fall back to the bundled sample
        path = "sample_candidates.json"
    else:
        path = file_obj.name
    text = open(path, "r", encoding="utf-8").read().strip()
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return records[: config.TOP_K_OUTPUT]


def rank_records(records: list[dict]) -> pd.DataFrame:
    model = _get_model()
    feat = features.build_features(records=records)
    kw = capability.build_keyword_capabilities(records=records)

    docs = [embeddings.profile_document(r) for r in records]
    ids = [r.get("candidate_id") for r in records]
    cand_emb = embeddings._embed(model, docs)
    jd_emb = embeddings.embed_jd(model, jd_spec.JD_QUERY_TEXT)
    _names, anchor_emb = embeddings.embed_anchors(model)

    sims = cand_emb @ anchor_emb.T
    from src.capability_graph import CAPABILITIES
    sem = {"candidate_id": ids}
    sdf = pd.DataFrame(sims, columns=[f"sem_{c}" for c in CAPABILITIES])
    for col in sdf.columns:
        sem[col] = sdf[col].rank(pct=True).to_numpy()  # percentile within the uploaded sample
    cap = capability.combine_capabilities(kw, pd.DataFrame(sem))

    jd_cos = pd.DataFrame({"candidate_id": ids, "jd_cosine": cand_emb @ jd_emb})
    df = jd_cos.merge(feat, on="candidate_id").merge(cap, on="candidate_id")
    df = scoring.compute_scores(df)
    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df = reasoning.add_reasoning(df)

    df["hidden_gem"] = (df["capability_fit"] >= 0.5) & (df["ai_core_skill_count"] <= 2)
    df["flag"] = np.where(df["is_honeypot"], "honeypot",
                          np.where(df["hidden_gem"], "hidden gem", ""))
    return df[["rank", "candidate_id", "score", "current_title", "flag", "reasoning"]]


def _run(file_obj):
    records = _parse_upload(file_obj)
    out = rank_records(records)
    return out


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Redrob — Intelligent Candidate Ranking") as demo:
        gr.Markdown(
            "# Redrob — Intelligent Candidate Ranking\n"
            "Ranks candidates for the **Senior AI Engineer** JD using inferred capabilities, "
            "career evidence, behavioral signals, and availability — not keyword overlap. "
            "Upload a small sample (JSON array or JSONL, ≤100 candidates) or run the bundled sample."
        )
        with gr.Row():
            inp = gr.File(label="candidates sample (.json / .jsonl)", file_types=[".json", ".jsonl"])
            btn = gr.Button("Rank", variant="primary")
        out = gr.Dataframe(label="Ranking", wrap=True)
        btn.click(_run, inputs=inp, outputs=out)
        gr.Markdown(
            "**Flags:** `honeypot` = impossible profile (hard-filtered to the bottom); "
            "`hidden gem` = strong career capability with few literal AI keywords."
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
