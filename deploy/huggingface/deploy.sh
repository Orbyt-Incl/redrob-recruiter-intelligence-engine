#!/usr/bin/env bash
# One-shot HuggingFace Space deploy for the Redrob ranker sandbox.
#
# Prereqs (once):
#   pip install -U huggingface_hub
#   Create a WRITE token at https://huggingface.co/settings/tokens
#
# Usage:
#   HF_TOKEN=hf_xxx bash deploy/huggingface/deploy.sh <hf_owner> [space_name]
# Example:
#   HF_TOKEN=hf_xxx bash deploy/huggingface/deploy.sh Orbyt-Incl redrob-recruiter-ranker
#
# If `hf` is not on your PATH (pip warned about this on Windows), point HF_BIN at it, e.g.
#   HF_BIN="/c/Users/anujn/AppData/Roaming/Python/Python313/Scripts/hf.exe" \
#     HF_TOKEN=hf_xxx bash deploy/huggingface/deploy.sh Orbyt-Incl
set -euo pipefail

OWNER="${1:?Pass your HuggingFace username/org as the first argument}"
NAME="${2:-redrob-recruiter-ranker}"
HF="${HF_BIN:-hf}"
REPO="$OWNER/$NAME"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="$(mktemp -d)/space"
mkdir -p "$STAGE"

# Stage exactly what the Space needs (no big artifacts; the model downloads on first boot).
cp "$ROOT/app.py" "$ROOT/requirements.txt" "$ROOT/sample_candidates.json" "$STAGE/"
cp -r "$ROOT/src" "$STAGE/src"
cp "$ROOT/deploy/huggingface/README.md" "$STAGE/README.md"   # <- Space config card

echo ">> Authenticating"
if [ -n "${HF_TOKEN:-}" ]; then "$HF" auth login --token "$HF_TOKEN" >/dev/null; fi

echo ">> Creating Space $REPO (ok if it already exists)"
"$HF" repo create "$REPO" --type space --space-sdk gradio || true

echo ">> Uploading files"
"$HF" upload "$REPO" "$STAGE" --repo-type space --commit-message "Deploy Redrob ranker sandbox"

echo ""
echo ">> Done. Space building at: https://huggingface.co/spaces/$REPO"
echo ">> When it shows 'Running', put that URL in submission_metadata.yaml -> sandbox_link"
