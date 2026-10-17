#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="data/checkpoints/nightly-$(date +%Y%m%d).log"

./scripts/sync_vault.sh >> "$LOG" 2>&1
sbt ingest >> "$LOG" 2>&1
sbt embed >> "$LOG" 2>&1

if [ "$(date +%u)" -eq 7 ]; then
  sbt train --epochs 2 >> "$LOG" 2>&1
fi

find data/checkpoints -maxdepth 1 -name 'sbt-*' -mtime +21 -exec rm -rf {} +