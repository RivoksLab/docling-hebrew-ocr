# docling-hebrew-ocr

A Hebrew OCR engine for [Docling](https://github.com/docling-project/docling), backed by
[paddleocr-hebrew](https://github.com/RivoksLab/paddleocr-hebrew). Use it for scanned
Hebrew documents, and for PDFs whose Hebrew text layer is broken (reversed, garbled or
missing), with `OcrMode.FULL_PAGE`.
How the models were built and how they score: the
[Hebrew OCR article](https://rivoklabs.com/research/hebrew-ocr) by [Rivok Labs](https://rivoklabs.com).

Docling's built-in engines and Hebrew: only Tesseract (`heb`) reads it. RapidOCR has no
Hebrew model, EasyOCR ships a Hebrew character list but no Hebrew model, and Nemotron OCR
has none either.

> 0.1.0, the first release. Tested with Docling 2.148 on CPU (Linux).

## Install

```bash
pip install docling-hebrew-ocr        # pulls docling and paddleocr-hebrew
```

The models (about 190 MB) are downloaded from the Hugging Face Hub
(`Rivok/paddleocr-hebrew`) into Docling's cache on first use, unless you point
`models_dir` (or `$DOCLING_HEBREW_OCR_MODELS`) at a local copy. Files needed:
`config.json`, `charset_v2f.txt`, `word-det/*`, `server-svtrv2/*` (plus `line-det/*`
for `pipeline="line"`).

## Use

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling_hebrew_ocr import HebrewOcrOptions

opts = PdfPipelineOptions()
opts.allow_external_plugins = True                    # third-party engines are opt-in
opts.ocr_options = HebrewOcrOptions(mode=OcrMode.FULL_PAGE)
converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
    InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
})
print(converter.convert("scan.pdf").document.export_to_markdown())
```

Command line:

```bash
docling --allow-external-plugins --ocr-engine hebrew --ocr-mode full_page scan.pdf
```

### Which mode

| input | mode | why |
|---|---|---|
| scanned PDF, photo, page image | `OcrMode.FULL_PAGE` | the detector sees the whole page; best accuracy |
| PDF with a broken Hebrew text layer | `OcrMode.FULL_PAGE` | every page is OCR'd and the text layer is ignored. Docling has no check for a reversed or garbled Hebrew layer |
| born-digital PDF with a good text layer | Docling's default (`OcrMode.DEFAULT`) | the text layer is used; only bitmaps and regions with no text are OCR'd |

In the default mode on a scan, Docling hands the engine one crop per layout region. That
works, but the word detector then sees text at another scale than on the whole page and
reads a little worse, so prefer `FULL_PAGE` for scans.

**Image files without DPI metadata** (most PNG scans) are taken by Docling as 72 dpi,
so the default `scale=3.0` would triple them. Pass `scale=1.0` for such images, or rely
on `max_image_side` (default 6000 px), which lowers the scale for very large pages.

### Options

| option | default | meaning |
|---|---|---|
| `models_dir` | `None` | local model folder; else `$DOCLING_HEBREW_OCR_MODELS`, then `<artifacts_path>/paddleocr-hebrew`, then a download |
| `pipeline` | `"word"` | `"word"`: word detector (best on clean pages). `"line"`: line detector (degraded scans, dense layouts) |
| `rec_mode` | `"cascade"` | `"cascade"`, `"ctc"` (fastest) or `"nrtr"` (slowest) |
| `threads` | Docling's `num_threads` | ONNX Runtime threads |
| `split_columns` | `True` | split each row of words at column gutters and table gaps |
| `gutter_factor` | `2.0` | a gap wider than this many word heights splits a row |
| `emit_word_cells` | `True` | also add OCR word cells, which Docling's table model uses as table tokens |
| `confidence` | `1.0` | constant written on every cell (see below) |
| `lang` | `["heb", "eng"]` | validation only: `heb`, `eng`, `iso:he`, `iso:en`; anything else is an error |
| `mode`, `scale`, `replace_broken_text_layer` | Docling's | as for every Docling OCR engine |

## What the engine emits

- **One `TextCell` per text line**, `from_ocr=True`, with the line's text in **logical
  order** (the order Hebrew is typed and stored; apply BiDi only to display it).
  English words and numbers inside a Hebrew line keep their own order:
  `אוניברסיטת חיפה 2024 University of Haifa`.
- **Right-to-left reading order:** rows from top to bottom; within a row, the pieces
  separated by a column gutter go right to left. Docling keeps this order inside each
  layout cluster (paragraph, list item, table), so every paragraph reads correctly.
- **Page coordinates:** boxes in PDF points, top-left origin, mapped back from the
  rendered image (including the crop offset in the region modes).
- **Word cells** (with `emit_word_cells`) for Docling's table structure model.
- **Confidence:** paddleocr-hebrew 0.1.0 reports no confidences. Every cell carries the
  constant `confidence` (1.0), and the page's `ocr_score` in Docling's confidence report
  is set to NaN (unknown), so Docling's quality grade does not count it.

## Known limits

- **Column order across the page is Docling's.** Docling's rule-based reading-order model
  reads side-by-side blocks left to right, so on a two-column Hebrew page the left column
  comes before the right one (83-93% of side-by-side block pairs on the HaZefira scans,
  with every OCR engine). Lines inside each block are in the right order. A fix belongs in
  Docling (an RTL option for the reading-order model).
- **Tables come out mirrored:** Docling numbers table columns from the left, so a Markdown
  table starts with the column a Hebrew reader reads last. The text inside each cell is in
  logical order (this engine feeds the table model words in reading order).
- **No confidences** (see above).
- Vertical text (rotated margin notes) is dropped; the recognizer reads horizontal text
  only.
- Small punctuation that the detector boxes on its own (a full stop after a gap, a comma
  between numbers) can be lost, and a prefix glued to a number can lose the number
  (`כ-46%` read as `כ-%`). Both come from paddleocr-hebrew's word detector.
- Speed: CPU-only ONNX. Inside Docling on 2 CPU threads the OCR stage takes about 3x as
  long as Tesseract's (55 s against 18 s per newspaper page on a shared 4-core desktop);
  a whole page takes about 2.8x as long
  ([bench/RESULTS.md](https://github.com/RivoksLab/docling-hebrew-ocr/blob/main/bench/RESULTS.md)).

## Benchmark

[bench/run_bench.py](https://github.com/RivoksLab/docling-hebrew-ocr/blob/main/bench/run_bench.py)
runs Docling with this engine and with Docling's Tesseract engine on a page set and scores
the Markdown: word F1 ignores word order, bigram F1 (pairs of neighbouring words) counts it.
On 34 scans of the 19th-century Hebrew newspaper HaZefira (OmiLab's transcriptions;
details, timings and what Docling's layout model does with right-to-left pages in
[bench/RESULTS.md](https://github.com/RivoksLab/docling-hebrew-ocr/blob/main/bench/RESULTS.md)):

| Docling + | word F1 | numbers found | bigram F1 | OCR stage, s/page |
|---|---:|---:|---:|---:|
| this engine | **81.2** | **83.0** | 63.8 | 55 |
| Tesseract [`heb_rivok`](https://github.com/RivoksLab/tesseract-heb-rivok) | 80.8 | 74.9 | **67.8** | 16 |
| Tesseract [`heb_rivok_lite`](https://github.com/RivoksLab/tesseract-heb-rivok)`+eng` | 77.1 | 74.3 | 62.5 | 16 |
| Tesseract `heb+eng` (Ubuntu's models) | 70.3 | 61.7 | 53.1 | 18 |

On these pages Tesseract with `heb_rivok`, our fine-tuned Tesseract model, comes close to
this engine and is faster. This engine finds more of the numbers, and it reads lines that
mix Hebrew and English in Hebrew order, which Tesseract does not do for short lines.

### Tesseract with heb_rivok in Docling

Download `heb_rivok.traineddata` from
[tesseract-heb-rivok](https://github.com/RivoksLab/tesseract-heb-rivok/releases) into a
folder that also holds `osd.traineddata` and `configs/` from your Tesseract installation
(links are enough), then:

```python
import os
from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, TesseractCliOcrOptions

os.environ["TESSDATA_PREFIX"] = "/path/to/tessdata"   # Docling lists languages without --tessdata-dir
opts = PdfPipelineOptions()
opts.ocr_options = TesseractCliOcrOptions(lang=["heb_rivok"], path="/path/to/tessdata", mode=OcrMode.FULL_PAGE)
```

## Tests

```bash
pip install -e ".[test]"
pytest -m "not slow"                                   # seconds, no models
DOCLING_HEBREW_OCR_MODELS=/path/to/models pytest       # also the end-to-end tests (minutes on CPU)
```

`tests/data/sample_page.png` comes from paddleocr-hebrew (Apache-2.0).
`tests/data/two_column.png` is synthetic, made by `tests/make_fixtures.py`.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.
