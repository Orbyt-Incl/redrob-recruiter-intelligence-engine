"""
Embeddings + FAISS index.

At precompute we:
  * load a small CPU sentence-transformer and SAVE IT LOCALLY (so ranking needs no network),
  * embed every candidate's profile document, L2-normalize, and build a FAISS inner-product
    index (cosine similarity),
  * embed the JD query and the capability anchors,
  * derive per-candidate *semantic* capability scores (cosine vs each anchor).

Persisted: candidate_embeddings.faiss, id_index.parquet, jd_embedding.npy,
anchor_embeddings.npy. The candidate matrix itself lives inside the FAISS index.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import config
from .capability_graph import CAPABILITIES, CAPABILITY_ANCHORS
from .data_io import iter_candidates, profile_document


def get_model():
    """Load the embedding model from the local cache if present, else download + save it."""
    from sentence_transformers import SentenceTransformer

    if os.path.isdir(config.EMBEDDING_MODEL_DIR) and os.listdir(config.EMBEDDING_MODEL_DIR):
        return SentenceTransformer(config.EMBEDDING_MODEL_DIR, device="cpu")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
    os.makedirs(config.EMBEDDING_MODEL_DIR, exist_ok=True)
    model.save(config.EMBEDDING_MODEL_DIR)
    return model


def _embed(model, texts: List[str], batch_size: int = 256) -> np.ndarray:
    """Encode + L2-normalize to float32 (cosine via inner product)."""
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


def embed_jd(model, query_text: str) -> np.ndarray:
    return _embed(model, [config.BGE_QUERY_PREFIX + query_text])[0]


def embed_anchors(model) -> Tuple[List[str], np.ndarray]:
    names = list(CAPABILITIES)
    anchors = [CAPABILITY_ANCHORS[c] for c in names]
    return names, _embed(model, anchors)


def build_embeddings(path: str = config.CANDIDATES_PATH, query_text: str | None = None):
    """
    Build the FAISS index and semantic capability scores.
    Returns (sem_df, jd_embedding, anchor_names, anchor_embeddings).
    """
    import faiss

    model = get_model()

    ids: List[str] = []
    docs: List[str] = []
    for cand in iter_candidates(path):
        ids.append(cand.get("candidate_id"))
        docs.append(profile_document(cand))

    cand_emb = _embed(model, docs)  # (N, D) normalized
    dim = cand_emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(cand_emb)
    faiss.write_index(index, config.FAISS_INDEX_PATH)

    pd.DataFrame({"candidate_id": ids}).to_parquet(config.ID_INDEX_PARQUET, index=False)

    # JD + anchors
    jd_emb = embed_jd(model, query_text or "")
    anchor_names, anchor_emb = embed_anchors(model)
    np.save(config.JD_EMBEDDING_PATH, jd_emb)
    np.save(config.ANCHOR_EMBEDDING_PATH, anchor_emb)

    # semantic capability scores as POPULATION PERCENTILES per capability (the raw cosines are
    # too compressed to use directly; percentile picks out the genuine top tail).
    sims = cand_emb @ anchor_emb.T  # (N, C)
    sem = {"candidate_id": ids}
    sims_df = pd.DataFrame(sims, columns=[f"sem_{c}" for c in anchor_names])
    for col in sims_df.columns:
        sem[col] = sims_df[col].rank(pct=True).round(4).to_numpy()
    sem_df = pd.DataFrame(sem)

    return sem_df, jd_emb, anchor_names, anchor_emb
