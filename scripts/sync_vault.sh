#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .env

rsync -a --delete \
  --exclude ".obsidian/" \
  --exclude ".trash/" \
  "$SBT_VAULT_PATH/" data/raw/vault/

echo "synced $(find data/raw/vault -name '*.md' | wc -l) notes"