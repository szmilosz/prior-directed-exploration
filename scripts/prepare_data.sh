#!/usr/bin/env bash
set -euo pipefail

# Decompress the vendored data assets in place (data/**/*.gz -> the sibling
# uncompressed file the training/eval scripts read) and verify checksums.
# Idempotent: existing decompressed files are only rebuilt with --force.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/data"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; fi

shopt -s nullglob
for gz in openings/*.gz test/*.gz; do
  out="${gz%.gz}"
  if [[ "$FORCE" -eq 0 && -s "$out" ]]; then
    echo "keep      $out"
    continue
  fi
  echo "unpack    $gz -> $out"
  gunzip -kf "$gz"
done

echo ""
echo "Verifying checksums (data/SHA256SUMS) ..."
sha256sum -c SHA256SUMS
echo "All data assets ready."
