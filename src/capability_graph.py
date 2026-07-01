"""
Capability graph — the system's core differentiator.

A recruiter reads "built a recommendation system at a product company" and infers the
person can do retrieval, ranking, embeddings, and evaluation — even if those exact words
never appear. We encode that inference explicitly:

  EVIDENCE_RULES : phrase/pattern in a profile  ->  latent capabilities it implies
  CAPABILITY_ANCHORS : a canonical sentence per capability, embedded so we can also score
                       capability presence by *semantic* similarity (keyword-free).

The capabilities are the axes the JD actually cares about for a Senior AI Engineer on a
retrieval/ranking product.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Canonical capability axes (aligned to JD must-haves).
CAPABILITIES = [
    "retrieval",         # dense/sparse retrieval, vector search, semantic search
    "ranking",           # ranking/re-ranking, learning-to-rank, relevance ordering
    "recommendation",    # recommender / personalization systems
    "embeddings",        # representation learning, embedding models
    "search_ir",         # search / information-retrieval systems (BM25, ES, etc.)
    "evaluation",        # NDCG/MRR/MAP, offline+online eval, A/B testing
    "production_ml",     # deployed ML at scale, serving, latency, pipelines
    "nlp",               # NLP / language understanding
    "llm",               # LLMs, fine-tuning, RAG
]

# Each rule: (compiled-pattern, {capability: weight}). Patterns match the lowercased blob.
# Weights are evidence strength in [0,1]; multiple rules accumulate (saturating) per capability.
_RAW_RULES: List[Tuple[str, Dict[str, float]]] = [
    # --- recommendation: the headline "hidden gem" signal ---
    (r"recommend(ation|er)?\s*(system|engine|model)?", {
        "recommendation": 1.0, "ranking": 0.7, "retrieval": 0.6, "embeddings": 0.5, "evaluation": 0.5}),
    (r"personaliz(ation|ed)", {"recommendation": 0.8, "ranking": 0.6, "evaluation": 0.4}),
    (r"\b(feed|news\s*feed|home\s*feed)\s*ranking", {"ranking": 1.0, "recommendation": 0.6, "evaluation": 0.5}),
    (r"collaborative filtering|matrix factorization|two-tower|candidate generation", {
        "recommendation": 1.0, "retrieval": 0.8, "embeddings": 0.7}),

    # --- search / ranking / IR ---
    (r"search\s*(ranking|relevance|engine|quality)", {"search_ir": 1.0, "ranking": 0.9, "retrieval": 0.7, "evaluation": 0.6}),
    (r"information retrieval|\bir\b", {"search_ir": 1.0, "retrieval": 0.8, "ranking": 0.6}),
    (r"learning\s*to\s*rank|\bltr\b|lambdamart|ranknet", {"ranking": 1.0, "evaluation": 0.7, "production_ml": 0.4}),
    (r"\b(bm25|elasticsearch|opensearch|solr|lucene)\b", {"search_ir": 1.0, "retrieval": 0.8}),
    (r"re-?rank|reranking", {"ranking": 1.0, "retrieval": 0.5}),
    (r"ctr|click-?through|click\s*model", {"ranking": 0.8, "recommendation": 0.6, "evaluation": 0.5}),

    # --- retrieval / vector / embeddings ---
    (r"\b(faiss|pinecone|weaviate|qdrant|milvus|annoy|hnsw|scann)\b", {"retrieval": 1.0, "embeddings": 0.7, "production_ml": 0.4}),
    (r"vector\s*(search|database|db|store|index)", {"retrieval": 1.0, "embeddings": 0.8, "search_ir": 0.6}),
    (r"semantic\s*search", {"retrieval": 1.0, "search_ir": 0.8, "embeddings": 0.7, "nlp": 0.5}),
    (r"embedding(s)?|sentence-?transformer|word2vec|\bbge\b|\be5\b|\bsbert\b", {"embeddings": 1.0, "retrieval": 0.6, "nlp": 0.5}),
    (r"representation learning|contrastive learning", {"embeddings": 0.9, "retrieval": 0.5}),

    # --- NLP / LLM ---
    (r"\bnlp\b|natural language", {"nlp": 1.0, "embeddings": 0.4}),
    (r"\b(bert|roberta|transformer|t5|gpt|llama|mistral)\b", {"nlp": 0.8, "llm": 0.6, "embeddings": 0.4}),
    (r"\b(llm|large language model)\b", {"llm": 1.0, "nlp": 0.6}),
    (r"\brag\b|retrieval-augmented", {"llm": 0.9, "retrieval": 0.9, "nlp": 0.5}),
    (r"fine-?tun|lora|qlora|peft|instruction tun", {"llm": 1.0, "nlp": 0.5, "production_ml": 0.3}),

    # --- evaluation / experimentation ---
    (r"\bndcg\b|\bmrr\b|\bmap@|mean average precision|recall@|precision@", {"evaluation": 1.0, "ranking": 0.6}),
    (r"a/?b\s*test|split\s*test|online experiment|experimentation", {"evaluation": 1.0, "production_ml": 0.4}),
    (r"offline (metric|eval)|offline-to-online", {"evaluation": 0.9}),

    # --- production ML / scale ---
    (r"production|deployed|in production|serving|model serving", {"production_ml": 1.0}),
    (r"real-?time|low-?latency|throughput|qps|requests per second", {"production_ml": 0.8}),
    (r"at scale|millions of (users|items|requests)|billions of", {"production_ml": 0.9}),
    (r"feature store|ml pipeline|training pipeline|mlops|kubeflow|airflow.*model", {"production_ml": 0.7}),
    (r"a/b|monitoring|model drift|retrain", {"production_ml": 0.5, "evaluation": 0.4}),
]

EVIDENCE_RULES: List[Tuple["re.Pattern[str]", Dict[str, float]]] = [
    (re.compile(pat, re.IGNORECASE), caps) for pat, caps in _RAW_RULES
]

# Canonical sentence per capability — embedded once at precompute, used for semantic
# (keyword-free) capability scoring of each candidate document.
CAPABILITY_ANCHORS: Dict[str, str] = {
    "retrieval": "Built dense and sparse retrieval systems using vector search and nearest-neighbor indexes to fetch relevant candidates from large corpora.",
    "ranking": "Designed and trained ranking and re-ranking models that order results by relevance, including learning-to-rank approaches.",
    "recommendation": "Built recommendation and personalization systems that suggest items to users based on behavior and content.",
    "embeddings": "Trained and deployed embedding models and representation learning to encode text and items into dense vectors.",
    "search_ir": "Built search and information-retrieval systems with BM25, inverted indexes, and relevance tuning.",
    "evaluation": "Designed evaluation frameworks for ranking systems using NDCG, MRR, MAP, and online A/B testing.",
    "production_ml": "Deployed machine-learning systems to production at scale with low latency, serving, and monitoring.",
    "nlp": "Worked on natural language processing, language understanding, and text models in production.",
    "llm": "Worked with large language models, fine-tuning, and retrieval-augmented generation.",
}

# How JD must-haves map onto capability axes (used to compute capability_fit).
# Weighted importance of each capability for THIS job.
JD_CAPABILITY_IMPORTANCE: Dict[str, float] = {
    "retrieval": 1.0,
    "ranking": 1.0,
    "embeddings": 0.9,
    "search_ir": 0.8,
    "recommendation": 0.8,
    "evaluation": 0.9,
    "production_ml": 0.85,
    "nlp": 0.6,
    "llm": 0.5,
}


def match_keyword_capabilities(text_lower: str) -> Dict[str, float]:
    """
    Apply EVIDENCE_RULES to a lowercased profile blob.
    Returns {capability: score in [0,1]} via saturating accumulation:
        score = 1 - prod(1 - w_i) over matched rules contributing w_i.
    """
    inv: Dict[str, float] = {c: 1.0 for c in CAPABILITIES}
    for pattern, caps in EVIDENCE_RULES:
        if pattern.search(text_lower):
            for cap, w in caps.items():
                inv[cap] *= (1.0 - w)
    return {c: round(1.0 - inv[c], 4) for c in CAPABILITIES}
