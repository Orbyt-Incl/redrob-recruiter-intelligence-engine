---
title: Redrob Recruiter Ranker
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: mit
---

# Redrob — Intelligent Candidate Ranking (Sandbox)

Hosted demo for the Redrob hackathon submission (team **Orbyt**). Upload a small candidate
sample (**≤100** candidates, JSON array or JSONL) and the app runs the full ranking pipeline
end-to-end on **CPU** — capability inference from career evidence, behavioral/availability
signals, the Recruiter Intelligence Score, honeypot hard-filter, and factual reasoning — then
returns a ranked table with `hidden gem` / `honeypot` flags. With no upload it ranks a bundled
50-candidate sample.

This mirrors `rank.py` but is self-contained for small inputs (direct cosine similarity instead
of a prebuilt FAISS index), so it needs no committed artifacts and stays well within the
≤5-minute CPU budget. Source: <https://github.com/Orbyt-Incl/redrob-recruiter-intelligence-engine>.

---

## This folder is the HuggingFace Space card

The YAML front-matter above is the Space configuration. **This file becomes the Space's
`README.md`.** The actual app code (`app.py`, `src/`, `requirements.txt`, `sample_candidates.json`)
lives at the repo root and is copied into the Space at deploy time.

## Deploy (one-time, requires a HuggingFace account + token)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                      # paste a write token from hf.co/settings/tokens

# 1. Create the Space (Gradio SDK). Name it to match submission_metadata.yaml:
huggingface-cli repo create redrob-recruiter-ranker --type space --space_sdk gradio

# 2. Populate it from a clone of the GitHub repo:
git clone https://huggingface.co/spaces/Orbyt-Incl/redrob-recruiter-ranker space && cd space
cp ../app.py ../requirements.txt ../sample_candidates.json .
cp -r ../src .
cp ../deploy/huggingface/README.md ./README.md      # <-- the Space card (this file)
git add -A && git commit -m "Deploy Redrob ranker sandbox" && git push
```

The Space builds automatically, installs `requirements.txt` (the `--extra-index-url` line pulls
the CPU torch wheel), downloads the `BAAI/bge-small-en-v1.5` embedding model from the Hub on first
boot, and launches `app.py`. Once it shows **Running**, put the public URL in
`submission_metadata.yaml` → `sandbox_link` and in the repo `README.md`.

> If HuggingFace rejects `sdk_version: 6.18.0`, change it to the latest version it offers — the
> app has no hard dependency on that exact Gradio release.

### Alternative sandboxes

Any platform in submission_spec section 10.5 is acceptable. The same `app.py` runs unmodified on
Streamlit Cloud / Replit, or locally with `python app.py`. A `docker run` recipe is also viable:
`pip install -r requirements.txt && python app.py`.
