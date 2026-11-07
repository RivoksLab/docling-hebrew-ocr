#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Docling with different OCR engines over a page set and score the Markdown.

The page set is a folder with manifest.json, images/ and refs/ (one reference text per
page). The HaZefira set used in bench/RESULTS.md (34 scans of the Hebrew newspaper
HaZefira, 1874 and 1887; OmiLab's transcriptions, evaluation only) is built by the
heb_rivok repository (https://github.com/RivoksLab/tesseract-heb-rivok):

    python -I eval/hazefira.py build --data-dir DIR

Outputs (Markdown, document JSON, timings) go to --out. Keep them out of git: they hold
text derived from the references.

    python bench/run_bench.py run --pages DIR --out OUT --system hebrew
    python bench/run_bench.py run --pages DIR --out OUT --system tess_rivok --tessdata RELEASE_TESSDATA
    python bench/run_bench.py score --pages DIR --out OUT

Systems (all CPU, $BENCH_THREADS threads, default 2), all on page images, FULL_PAGE:
  hebrew       this plugin
  tesseract    Docling's Tesseract CLI engine, heb+eng (the system's models)
  tess_lite    Tesseract CLI, heb_rivok_lite+eng   (--tessdata: a folder with the release files)
  tess_rivok   Tesseract CLI, heb_rivok            (same)
Images carry no DPI metadata, so Docling takes them as 72 dpi; scale=1.0 sends every
engine the image at its own resolution.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import load_pages, reference, tokens  # noqa: E402

THREADS = int(os.environ.get("BENCH_THREADS", "2"))
SYSTEMS = ["hebrew", "tesseract", "tess_lite", "tess_rivok"]
TESS_LANGS = {"tesseract": ["heb", "eng"], "tess_lite": ["heb_rivok_lite", "eng"], "tess_rivok": ["heb_rivok"]}


def system_tessdata() -> Path:
    """The folder of the installed Tesseract models (from `tesseract --list-langs`)."""
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    m = re.search(r'"(.+?)"', out.stdout + out.stderr)
    if not m:
        raise SystemExit("cannot find Tesseract's tessdata folder (tesseract --list-langs)")
    return Path(m.group(1))


def merged_tessdata(release: Path, out: Path) -> Path:
    """A tessdata folder holding the release models next to the system's own files.

    Docling lists languages with `tesseract --list-langs` without --tessdata-dir, so the
    folder must also be TESSDATA_PREFIX. It needs osd (Docling's per-page OSD call),
    configs/ (the `tsv` output Docling asks for), eng, and the stock heb so that the
    `tesseract` system still works in the same process.
    """
    system = system_tessdata()
    d = out / "_tessdata"
    d.mkdir(parents=True, exist_ok=True)
    links = {f"{m}.traineddata": release / f"{m}.traineddata" for m in ("heb_rivok", "heb_rivok_lite")}
    links |= {f"{m}.traineddata": system / f"{m}.traineddata" for m in ("osd", "eng", "heb")}
    links["configs"] = system / "configs"
    for name, src in links.items():
        if not src.exists():
            raise SystemExit(f"missing {src}")
        if not (d / name).exists():
            (d / name).symlink_to(src.resolve())
    return d


def make_converter(system: str, out: Path, models: Path | None, tessdata: Path | None):
    import torch

    torch.set_num_threads(THREADS)
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, TesseractCliOcrOptions
    from docling.datamodel.settings import settings
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    settings.debug.profile_pipeline_timings = True
    po = PdfPipelineOptions()
    po.accelerator_options = AcceleratorOptions(num_threads=THREADS, device="cpu")
    po.generate_parsed_pages = True  # to count OCR cells afterwards
    if system == "hebrew":
        from docling_hebrew_ocr import HebrewOcrOptions

        po.allow_external_plugins = True
        po.ocr_options = HebrewOcrOptions(models_dir=models, threads=THREADS, mode=OcrMode.FULL_PAGE, scale=1.0)
    elif system == "tesseract":
        po.ocr_options = TesseractCliOcrOptions(lang=TESS_LANGS[system], mode=OcrMode.FULL_PAGE, scale=1.0)
    elif system in ("tess_lite", "tess_rivok"):
        if tessdata is None:
            raise SystemExit(f"{system} needs --tessdata (a folder with heb_rivok*.traineddata)")
        d = merged_tessdata(tessdata, out)
        os.environ["TESSDATA_PREFIX"] = str(d)
        po.ocr_options = TesseractCliOcrOptions(lang=TESS_LANGS[system], path=str(d), mode=OcrMode.FULL_PAGE, scale=1.0)
    else:
        raise ValueError(system)
    fmt = {InputFormat.PDF: PdfFormatOption(pipeline_options=po), InputFormat.IMAGE: ImageFormatOption(pipeline_options=po)}
    return DocumentConverter(format_options=fmt)


def run(system: str, pages_dir: Path, out: Path, models: Path | None, tessdata: Path | None, limit: int | None) -> None:
    pages = load_pages(pages_dir)[:limit]
    d = out / system
    d.mkdir(parents=True, exist_ok=True)
    todo = [e for e in pages if not (d / f"{e['id']}.json").exists()]
    print(f"{system}: {len(todo)} of {len(pages)} pages to do", flush=True)
    if not todo:
        return
    from docling.datamodel.base_models import InputFormat

    conv = make_converter(system, out, models, tessdata)
    conv.initialize_pipeline(InputFormat.IMAGE)  # load the models before the clock starts
    for e in todo:
        t0 = time.time()
        try:
            res = conv.convert(e["image"])
        except Exception:
            traceback.print_exc()
            print(f"  FAILED {e['id']}", flush=True)
            continue
        wall = time.time() - t0
        timings = {k: round(sum(v.times), 2) for k, v in res.timings.items()}
        cells = [c for p in res.pages if p.parsed_page is not None for c in p.parsed_page.textline_cells]
        doc = res.document
        rec = {
            "page": e["id"],
            "system": system,
            "wall_sec": round(wall, 2),
            "ocr_sec": timings.get("ocr", 0.0),
            "timings": timings,
            "n_cells": len(cells),
            "n_ocr_cells": sum(1 for c in cells if c.from_ocr),
            "ocr_chars": sum(len(c.text) for c in cells if c.from_ocr),
            "status": str(res.status),
            "markdown": doc.export_to_markdown(),
        }
        (d / f"{e['id']}.doc.json").write_text(json.dumps(doc.export_to_dict(), ensure_ascii=False), encoding="utf-8")
        (d / f"{e['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {e['id']}: wall {wall:.1f}s ocr {rec['ocr_sec']:.1f}s, {rec['n_ocr_cells']} OCR cells", flush=True)


def score(pages_dir: Path, out: Path, per_page: bool, common: bool = False) -> None:
    pages = [e for e in load_pages(pages_dir) if reference(e) is not None]
    ref_tokens = {e["id"]: tokens(reference(e)) for e in pages}
    refs = {k: collections.Counter(t) for k, t in ref_tokens.items()}
    bigrams = {k: collections.Counter(zip(t, t[1:])) for k, t in ref_tokens.items()}
    dirs = sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if common:  # only the pages every system has finished
        pages = [e for e in pages if all((p / f"{e['id']}.json").exists() for p in dirs)]
    print(f"{len(pages)} pages")
    print(f"{'system':12s} {'pages':>5s} {'recall':>7s} {'prec':>7s} {'F1':>6s} {'pageF1':>7s} {'numbers':>8s} "
          f"{'bigramF1':>8s} {'wall s/p':>9s} {'ocr s/p':>8s}")
    for sysdir in dirs:
        hit = nref = npred = nh = nn = bhit = bref = bpred = 0
        wall = ocr = 0.0
        f1s = []
        for e in pages:
            f = sysdir / f"{e['id']}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            pred_tokens = tokens(r["markdown"])
            ref, pred = refs[e["id"]], collections.Counter(pred_tokens)
            rb, pb = bigrams[e["id"]], collections.Counter(zip(pred_tokens, pred_tokens[1:]))
            bhit += sum((rb & pb).values())
            bref += sum(rb.values())
            bpred += sum(pb.values())
            h = sum((ref & pred).values())
            numref = collections.Counter({w: c for w, c in ref.items() if re.search(r"\d", w)})
            hit += h
            nref += sum(ref.values())
            npred += sum(pred.values())
            nh += sum((numref & pred).values())
            nn += sum(numref.values())
            rc, pc = h / max(1, sum(ref.values())), h / max(1, sum(pred.values()))
            f1s.append(2 * rc * pc / max(1e-9, rc + pc))
            wall += r["wall_sec"]
            ocr += r["ocr_sec"]
            if per_page:
                print(f"   {sysdir.name:12s} {e['id']:36s} F1 {100 * f1s[-1]:5.1f}  wall {r['wall_sec']:6.1f}s")
        n = len(f1s)
        if not n:
            continue
        rec, prec = hit / max(1, nref), hit / max(1, npred)
        f1 = 2 * rec * prec / max(1e-9, rec + prec)
        brec, bprec = bhit / max(1, bref), bhit / max(1, bpred)
        bf1 = 2 * brec * bprec / max(1e-9, brec + bprec)
        print(f"{sysdir.name:12s} {n:5d} {100 * rec:6.1f}% {100 * prec:6.1f}% {100 * f1:5.1f}% "
              f"{100 * sum(f1s) / n:6.1f}% {100 * nh / max(1, nn):7.1f}% {100 * bf1:7.1f}% {wall / n:9.1f} {ocr / n:8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--pages", type=Path, required=True, help="page set: manifest.json, images/, refs/")
    ap.add_argument("--out", type=Path, required=True, help="output folder (keep it out of git)")
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--models", type=Path, default=os.environ.get("DOCLING_HEBREW_OCR_MODELS"),
                    help="paddleocr-hebrew model folder (default: $DOCLING_HEBREW_OCR_MODELS, else download)")
    ap.add_argument("--tessdata", type=Path, help="folder with heb_rivok.traineddata and heb_rivok_lite.traineddata")
    ap.add_argument("--limit", type=int, default=None, help="first N pages only")
    ap.add_argument("--per-page", action="store_true", help="score: print every page")
    ap.add_argument("--common", action="store_true", help="score: only pages every system finished")
    a = ap.parse_args()
    if a.cmd == "score":
        return score(a.pages, a.out, a.per_page, a.common)
    if not a.system:
        ap.error("run needs --system")
    if shutil.which("tesseract") is None and a.system != "hebrew":
        raise SystemExit("tesseract is not on PATH")
    run(a.system, a.pages, a.out, Path(a.models) if a.models else None, a.tessdata, a.limit)


if __name__ == "__main__":
    main()
