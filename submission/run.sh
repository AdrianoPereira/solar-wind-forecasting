#!/usr/bin/env bash
set -euo pipefail

T0="${1:?FATAL: no timestamp argument supplied by platform}"
echo "[run] t0=${T0} submission=${SUBMISSION_ID:-unknown}"

OUT=/tmp/forecast.json

# Hybrid forecast: fetches ACE and AIA data available at t0, writes 72 hourly speeds as JSON.
python3 /app/forecast.py "$T0" > "$OUT"

echo "[run] produced $(wc -c < "$OUT") bytes"
/app/upload.sh "$OUT"
echo "[run] done"
