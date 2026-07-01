"""
FAISS retrieval -> shortlist (100k -> ~1500).

Primary recall is JD-semantic similarity. We then UNION in the strongest capability-graph
matches so a "hidden gem" — someone who built recommendation/ranking systems but whose
wording doesn't lexically resemble the JD — still makes the shortlist.

Returns a DataFrame of [candidate_id, jd_cosine] for the shortlist.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from . import config


def load_artifacts():
    import faiss

    index = faiss.read_index(config.FAISS_INDEX_PATH)
    ids = pd.read_parquet(config.ID_INDEX_PARQUET)["candidate_id"].tolist()
    jd_emb = np.load(config.JD_EMBEDDING_PATH).astype(np.float32)
    return index, ids, jd_emb


def shortlist(
    index,
    ids: List[str],
    jd_emb: np.ndarray,
    cap_df: pd.DataFrame,
    k: int = config.SHORTLIST_SIZE,
) -> pd.DataFrame:
    k = min(k, len(ids))
    q = jd_emb.reshape(1, -1)
    sims, idxs = index.search(q, k)          # inner product == cosine (normalized vectors)
    sims, idxs = sims[0], idxs[0]

    rows = {ids[i]: float(s) for i, s in zip(idxs, sims)}

    # capability union: top capability_fit candidates not already retrieved
    id_to_row = {cid: r for r, cid in enumerate(ids)}
    extra_n = max(k // 5, 200)
    cap_top = cap_df.sort_values("capability_fit", ascending=False).head(extra_n)
    extras = [cid for cid in cap_top["candidate_id"] if cid not in rows]
    for cid in extras:
        r = id_to_row.get(cid)
        if r is None:
            continue
        vec = index.reconstruct(r)            # IndexFlatIP supports reconstruct
        rows[cid] = float(np.dot(vec, jd_emb))

    return pd.DataFrame(
        {"candidate_id": list(rows.keys()), "jd_cosine": list(rows.values())}
    )
