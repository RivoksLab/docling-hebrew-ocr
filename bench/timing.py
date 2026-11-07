#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Seconds per page, measured fairly: every system converts the same pages, interleaved
page by page, so all of them see the same machine load. run_bench.py runs the systems
one after another, and on a shared machine their timings are not comparable.

    python bench/timing.py --pages DIR --out OUT [--tessdata RELEASE_TESSDATA] [--n 4]
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bench import make_converter  # noqa: E402
from scoring import load_pages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, required=True, help="page set: manifest.json, images/, refs/")
    ap.add_argument("--out", type=Path, required=True, help="scratch folder (for the merged tessdata)")
    ap.add_argument("--n", type=int, default=4, help="pages, evenly spaced")
    ap.add_argument("--models", type=Path, default=os.environ.get("DOCLING_HEBREW_OCR_MODELS"))
    ap.add_argument("--tessdata", type=Path, help="folder with heb_rivok*.traineddata (adds tess_lite, tess_rivok)")
    a = ap.parse_args()
    from docling.datamodel.base_models import InputFormat

    systems = ["hebrew", "tesseract"] + (["tess_lite", "tess_rivok"] if a.tessdata else [])
    convs = {}
    for s in systems:
        convs[s] = make_converter(s, a.out, Path(a.models) if a.models else None, a.tessdata)
        convs[s].initialize_pipeline(InputFormat.IMAGE)
    pages = load_pages(a.pages)
    step = max(1, len(pages) // a.n)
    times = {}
    for e in pages[::step][: a.n]:
        for s in systems:
            t0 = time.time()
            res = convs[s].convert(e["image"])
            wall = time.time() - t0
            ocr = sum(res.timings["ocr"].times) if "ocr" in res.timings else 0.0
            times.setdefault(s, []).append((wall, ocr))
            print(f"  {e['id']} {s}: wall {wall:.1f}s ocr {ocr:.1f}s", flush=True)
    print(f"\n{'system':12s} {'pages':>5s} {'wall s/p':>9s} {'ocr s/p':>8s} {'median ocr':>10s}")
    for s, v in times.items():
        print(f"{s:12s} {len(v):5d} {statistics.mean(w for w, _ in v):9.1f} "
              f"{statistics.mean(o for _, o in v):8.1f} {statistics.median(o for _, o in v):10.1f}")


if __name__ == "__main__":
    main()
