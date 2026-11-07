#!/bin/bash
# Run every system on a page set, one after the other (resumable: done pages are skipped),
# then score. CPU etiquette for a shared machine: 2 threads, nice 10.
#   PAGES=/path/to/hazefira OUT=/path/to/out TESSDATA=/path/to/heb_rivok/tessdata bench/run_all.sh
# PAGES: page set (manifest.json, images/, refs/); OUT: outputs, kept out of git;
# TESSDATA: folder with heb_rivok*.traineddata (skip tess_lite/tess_rivok if unset);
# PY: the Python with docling and this plugin installed (default: python3).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PAGES:?set PAGES to the page-set folder}" "${OUT:?set OUT to an output folder}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 DOCLING_NUM_THREADS=2 BENCH_THREADS=2
PY="${PY:-python3}"
log() { echo "$(date '+%F %T') $*"; }
systems="tesseract hebrew"
[ -n "${TESSDATA:-}" ] && systems="tesseract tess_lite tess_rivok hebrew"
for s in $systems; do
  log "start $s"
  nice -n 10 "$PY" -I bench/run_bench.py run --pages "$PAGES" --out "$OUT" --system "$s" ${TESSDATA:+--tessdata "$TESSDATA"} 2>&1 \
    | grep -v --line-buffered "Loading weights\|allow_external_plugins=false\|^\s*$" || true
  log "done $s"
done
nice -n 10 "$PY" -I bench/run_bench.py score --pages "$PAGES" --out "$OUT"
log ALL_DONE
