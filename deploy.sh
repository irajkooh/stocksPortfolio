#!/usr/bin/env bash
# deploy.sh — commit everything and push to GitHub → triggers CI/CD → HF Space
set -euo pipefail

# ── Load GitHub token ─────────────────────────────────────────────────────────
TOKEN=$(grep 'ghp_' _secrets/GH_HF_secrets.md | head -1 | awk '{print $1}')
if [[ -z "$TOKEN" ]]; then
  echo "ERROR: GitHub token not found in _secrets/GH_HF_secrets.md" >&2
  exit 1
fi

REMOTE="https://${TOKEN}@github.com/irajkooh/stocksPortfolio.git"

# ── Stage & commit (skip if nothing changed) ─────────────────────────────────
git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit — pushing current HEAD as-is."
else
  MSG="${1:-deploy: $(date '+%Y-%m-%d %H:%M')}"
  git commit -m "$MSG"
  echo "Committed: $MSG"
fi

# ── Push ──────────────────────────────────────────────────────────────────────
git push "$REMOTE" main
echo ""
echo "Pushed. CI/CD running at:"
echo "  https://github.com/irajkooh/stocksPortfolio/actions"
echo ""
echo "HF Space:"
echo "  https://huggingface.co/spaces/irajkoohi/stocksPortfolio"
