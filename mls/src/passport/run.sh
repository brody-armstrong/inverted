#!/usr/bin/env bash
# Rebuild the Timbers2 readiness passport end to end: fetch -> DuckDB -> models -> page -> tests.
# Usage: bash run.sh <work_dir> <site_repo> <private_dir>
# The 2027 prediction list names 356 players, many of them teenagers, so it is NOT committed. It is kept in
# <private_dir>/predictions_2027_frozen.csv; only its SHA-256 hash is committed (results/passport/
# predictions_2027_frozen.sha256), which proves the list was not changed. It is published when scored after 2027.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:?work dir}"; SITE="${2:?site repo}"; PRIV="${3:?private dir (outside any repo)}"
export PASSPORT_RAW="$WORK/raw" PASSPORT_DB="$WORK/passport.duckdb" PASSPORT_OUT="$WORK/out" SITE_REPO="$SITE" PASSPORT_PRIVATE="$PRIV"
RES="$HERE/../../results/passport"
python3 -I "$HERE/fetch.py" "$PASSPORT_RAW"
python3 -I "$HERE/db.py" "$PASSPORT_RAW" "$PASSPORT_DB"
python3 -I "$HERE/models.py" "$PASSPORT_DB" "$HERE/../../results/pathway_model.json" "$PASSPORT_OUT"
mkdir -p "$RES" "$PRIV"
FROZEN="$PRIV/predictions_2027_frozen.csv"; HASH="$RES/predictions_2027_frozen.sha256"
if [ -f "$HASH" ]; then
  [ -f "$FROZEN" ] || { echo "frozen 2027 list missing from $PRIV"; exit 1; }
  ( cd "$PRIV" && sha256sum -c "$HASH" )          # the private list must match the committed hash
  cp "$FROZEN" "$PASSPORT_OUT/predictions_2027.csv"
else
  cp "$PASSPORT_OUT/predictions_2027.csv" "$FROZEN"
  ( cd "$PRIV" && sha256sum predictions_2027_frozen.csv ) > "$HASH"
fi
for f in passport_players passport_kpis passport_timeline reliability backtest_a backtest_a_zero_mls_exploratory calibration_test hazard_coefficients_train kpi_scale_2026; do
  cp "$PASSPORT_OUT/$f.csv" "$RES/$f.csv"; done
cp "$PASSPORT_OUT/backtest_b.json" "$PASSPORT_OUT/summary.json" "$RES/"
python3 -I "$HERE/page.py" "$PASSPORT_OUT" "$PASSPORT_DB" "$SITE"
python3 -m pytest -q "$HERE/test_pipeline.py"
