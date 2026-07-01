"""
Central configuration for the Redrob ranking system.

Everything that encodes a *judgment call* derived from the job description lives here
(firm lists, keyword sets, scoring weights, thresholds) so tuning happens in one place
and both the precompute and ranking phases stay consistent.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
EMBEDDING_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "embedding_model")  # local copy, no network at rank time

CANDIDATES_PATH = os.path.join(ROOT, "candidates.jsonl")

FEATURES_PARQUET = os.path.join(ARTIFACTS_DIR, "features.parquet")
CAPABILITY_PARQUET = os.path.join(ARTIFACTS_DIR, "capability_scores.parquet")
JD_REQUIREMENTS_JSON = os.path.join(ARTIFACTS_DIR, "jd_requirements.json")
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "candidate_embeddings.faiss")
ID_INDEX_PARQUET = os.path.join(ARTIFACTS_DIR, "id_index.parquet")        # row<->candidate_id alignment
JD_EMBEDDING_PATH = os.path.join(ARTIFACTS_DIR, "jd_embedding.npy")
ANCHOR_EMBEDDING_PATH = os.path.join(ARTIFACTS_DIR, "anchor_embeddings.npy")
LLM_ENRICHMENT_PARQUET = os.path.join(ARTIFACTS_DIR, "llm_enrichment.parquet")  # optional

# --------------------------------------------------------------------------------------
# Embedding model (small, CPU-friendly, runs offline once cached locally)
# --------------------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-dim; good quality/size tradeoff on CPU
EMBEDDING_DIM = 384
# bge models recommend a query instruction prefix for asymmetric search (jd=query, candidate=passage)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------------------
# Retrieval / shortlist
# --------------------------------------------------------------------------------------
SHORTLIST_SIZE = 1500           # 100k -> 1500 candidates carried into full scoring
TOP_K_OUTPUT = 100              # final submission size

# Semantic anchor capability is supplied as a POPULATION PERCENTILE per capability. The
# embedding model barely separates tech-adjacent traps from real builders, so only the top
# tail contributes, as a bounded boost on top of keyword career evidence.
SEM_PCT_FLOOR = 0.90     # only candidates above this anchor-similarity percentile get a boost
SEM_BLEND = 0.35         # max boost weight from the semantic tail

# --------------------------------------------------------------------------------------
# Reference date for recency calculations.
# Set from the data (max last_active_date) at precompute time; this is the fallback.
# --------------------------------------------------------------------------------------
REFERENCE_DATE = "2026-06-14"

# --------------------------------------------------------------------------------------
# Domain knowledge derived from the JD ("Senior AI Engineer — Founding Team")
# --------------------------------------------------------------------------------------

# Consulting / IT-services firms. The JD explicitly down-weights candidates whose ENTIRE
# career is at these. Matched case-insensitively as substrings of company names.
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
    "deloitte", "ibm global services", "dxc", "hexaware", "birlasoft",
    "persistent systems", "l&t infotech", "ust global", "virtusa", "zensar",
    "syntel", "igate", "nttdata", "ntt data", "atos",
]
# Industries that signal a services/consulting shop rather than a product company.
CONSULTING_INDUSTRIES = ["it services", "consulting", "staffing", "outsourcing", "information technology and services"]

# Target geography (JD: Pune/Noida preferred; Hyderabad/Mumbai/Delhi NCR welcome; relocation ok).
TARGET_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon",
    "gurugram", "ncr", "ghaziabad", "faridabad", "navi mumbai", "thane",
]
TARGET_COUNTRY = "india"

# Experience band (JD: 5-9, ideal 6-8).
EXP_IDEAL_LOW = 6.0
EXP_IDEAL_HIGH = 8.0
EXP_ACCEPT_LOW = 5.0
EXP_ACCEPT_HIGH = 9.0

# Notice period (JD: sub-30 ideal; can buy out <=30; 30+ in scope but higher bar).
NOTICE_IDEAL_DAYS = 30

# Job-hopping: JD penalizes ~1.5yr switching ("title-chasers").
SHORT_STINT_MONTHS = 18          # a completed role shorter than this counts as a "short stint"
JOB_HOP_RATIO_PENALTY = 0.5      # if > half of completed roles are short stints -> hopper

# --- Keyword inventories (lowercase, substring matched against profile text) ---

# Core AI/IR skills the JD actually wants (embeddings, retrieval, ranking, eval, vector search).
CORE_AI_KEYWORDS = [
    "embedding", "retrieval", "ranking", "rank ", "recommendation", "recommender",
    "search", "information retrieval", "vector", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "bm25", "semantic search",
    "rag", "nlp", "natural language", "transformer", "bert", "sentence-transformer",
    "learning to rank", "ltr", "ndcg", "mrr", "map@", "relevance", "a/b test",
    "experimentation", "fine-tun", "lora", "qlora", "peft", "llm",
]

# Skills the dataset uses as "AI core" markers (matches the skills[].name vocabulary).
AI_CORE_SKILL_NAMES = [
    "nlp", "information retrieval", "recommendation systems", "learning to rank",
    "semantic search", "vector search", "embeddings", "ranking", "search ranking",
    "rag", "llm", "fine-tuning llms", "transformers", "deep learning",
    "machine learning", "ml", "pytorch", "tensorflow", "a/b testing",
]

# Anti-domain: CV/speech/robotics WITHOUT NLP/IR is explicitly down-weighted.
CV_SPEECH_ROBOTICS_KEYWORDS = [
    "computer vision", "image classification", "object detection", "segmentation",
    "speech recognition", "speech-to-text", "asr", "robotics", "slam", "lidar",
    "autonomous", "opencv", "ocr", "face recognition", "pose estimation",
]
NLP_IR_KEYWORDS = [
    "nlp", "natural language", "information retrieval", "search", "ranking",
    "retrieval", "recommendation", "embedding", "semantic", "text", "language model",
    "transformer", "bert", "llm", "rag",
]

# Research-only signal (academic / no production).
RESEARCH_KEYWORDS = [
    "research scientist", "phd researcher", "postdoc", "post-doctoral", "research fellow",
    "research assistant", "academic", "publication", "paper", "thesis", "lab",
]
PRODUCTION_KEYWORDS = [
    "production", "deployed", "deploy", "shipped", "launched", "real-time", "scale",
    "users", "latency", "throughput", "serving", "pipeline", "platform", "api",
]

# "Recent LangChain-only" framework-enthusiast signal (penalized unless deep ML history).
FRAMEWORK_ONLY_KEYWORDS = ["langchain", "llamaindex", "auto-gpt", "autogpt"]

# Title mismatch: keyword-stuffer with a non-engineering current title (the JD's explicit trap).
NON_ENGINEERING_TITLES = [
    "marketing", "sales", "recruiter", "hr ", "human resources", "content writer",
    "copywriter", "account manager", "business development", "customer success",
    "operations manager", "project manager", "program manager", "designer",
    "accountant", "finance", "teacher", "lecturer", "professor", "consultant",
]
ENGINEERING_TITLE_KEYWORDS = [
    "engineer", "developer", "scientist", "ml ", "machine learning", "ai ",
    "architect", "programmer", "sde", "researcher", "data",
]
# "Engineer" titles that are NOT software/ML engineering — must not count as an engineering fit.
NON_SOFTWARE_ENGINEER_TITLES = [
    "mechanical engineer", "civil engineer", "electrical engineer", "chemical engineer",
    "industrial engineer", "structural engineer", "hardware engineer", "sales engineer",
    "biomedical engineer", "aerospace engineer", "automobile engineer", "production engineer",
]

# --------------------------------------------------------------------------------------
# Recruiter Intelligence Score weights (component blend, tuned against proxy eval)
# --------------------------------------------------------------------------------------
RIS_WEIGHTS = {
    "capability_fit": 0.34,     # do they actually have the capabilities (graph + embedding)
    "career_evidence": 0.26,    # demonstrated in real jobs at product companies, at scale
    "availability": 0.22,       # behaviorally hireable (active, responsive, reasonable notice)
    "recruiter_demand": 0.08,   # recruiters already showing interest
    "experience_fit": 0.10,     # within the 5-9 (ideal 6-8) band
}

# Soft penalties (subtracted from the blended score, each in [0,1] * weight).
RIS_PENALTIES = {
    "job_hopping": 0.10,
    "consulting_only": 0.12,
    "research_only": 0.10,
    "cv_speech_only": 0.12,
    "framework_only": 0.08,
    "title_mismatch": 0.18,     # strongest soft penalty: keyword-stuffer with wrong title
    "location_miss": 0.05,
    "notice_long": 0.04,
}

# Honeypots are the ONLY hard filter.
HONEYPOT_SCORE_FLOOR = 1e-6     # forced to the very bottom; effectively excluded from top-100
# Require multiple/egregious impossibility signals before hard-filtering (favor precision:
# a false positive removes a real candidate). Single mild inconsistencies are synthetic noise.
HONEYPOT_THRESHOLD = 2.0
