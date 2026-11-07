# Benchmark results, 2026-11-07

Docling 2.148.0 (docling-parse 7.24.2, docling-ibm-models 4.0.4), CPU only, 2 threads,
on a shared 4-core Xeon E5-1630 v4. paddleocr-hebrew 0.1.0 (word pipeline, cascade).
Tesseract 5.5.0. Commands: `bench/run_all.sh` (or `bench/run_bench.py run` per system),
then `bench/run_bench.py score`, `bench/rtl_layout.py` and `bench/timing.py`.

**Page set:** 34 scans of the Hebrew newspaper HaZefira (1874 and 1887), with OmiLab's
transcriptions as references (evaluation only; no licence for redistribution, so neither
the pages nor the outputs are in this repository). Build it with
`python -I eval/hazefira.py build --data-dir DIR` from
[tesseract-heb-rivok](https://github.com/RivoksLab/tesseract-heb-rivok).

**Scoring** (`bench/scoring.py`): order-free word F1, pooled over all pages, on Docling's
Markdown (`export_to_markdown()`, body only, as a RAG pipeline gets it). Markup, niqqud
and edge punctuation are removed and Hebrew/digit runs split. "bigram F1" scores pairs of
neighbouring words: an order-sensitive supplement.

## Docling with each OCR engine (page images, `OcrMode.FULL_PAGE`, scale 1.0)

| Docling + | word F1 | recall | precision | numbers found | bigram F1 |
|---|---:|---:|---:|---:|---:|
| this engine | **81.2** | **81.8** | 80.6 | **83.0** | 63.8 |
| Tesseract `heb_rivok` | 80.8 | 80.8 | **80.8** | 74.9 | **67.8** |
| Tesseract `heb_rivok_lite+eng` | 77.1 | 77.1 | 77.1 | 74.3 | 62.5 |
| Tesseract `heb+eng` (Ubuntu's models) | 70.3 | 70.8 | 69.8 | 61.7 | 53.1 |

`heb_rivok` and `heb_rivok_lite` are the fine-tuned Tesseract models from
[tesseract-heb-rivok](https://github.com/RivoksLab/tesseract-heb-rivok), run through Docling's
Tesseract CLI engine (`heb_rivok_lite` with `+eng`; `heb_rivok` reads English itself).
The `heb_rivok_lite` run was made under its development name; the file is the same model.
These runs use `heb_rivok` 1.0; 1.0.1 only makes Tesseract lay its pages out right to left
and reads the same text.

`heb_rivok` 1.0 makes Tesseract lay out the page left to right (its character set has
more Latin than Hebrew letters), which outside Docling puts the left column first. Inside
Docling it makes no difference: Docling's layout model groups the words into blocks and
its reading-order model orders the blocks (left first for every engine, next sections).

RapidOCR and EasyOCR cannot take part: RapidOCR (3.10, PP-OCRv4/v5/v6) has no Hebrew
model (`OcrLanguageNotSupportedError` for `iso:he`), and EasyOCR 1.7.2 ships a Hebrew
character list but no Hebrew model (`he` is not in its language list). Nemotron OCR has
no Hebrew either. OcrMac (macOS Vision) was not tested.

The same engines outside Docling (same pages, same scorer): this engine 81.3, `heb_rivok`
81.3, `heb_rivok_lite` 76.6, Tesseract `heb+eng` 67.6, so Docling loses almost no words on
these pages.

## Speed

The sequential runs above shared the machine with other jobs (load up to 50), so their
timings are not comparable. Interleaved timing (`bench/timing.py`: 4 evenly spaced pages, every system converts each page in turn, 2 threads each, machine load 6-9), seconds
per page:

| Docling + | whole page | OCR stage |
|---|---:|---:|
| this engine | 57.5 | 55.0 |
| Tesseract `heb_rivok` | 17.9 | 16.0 |
| Tesseract `heb_rivok_lite+eng` | 17.5 | 15.6 |
| Tesseract `heb+eng` | 20.3 | 17.6 |

The OCR stage of this engine is about 3x slower than Tesseract's (which includes Docling's
per-page OSD call); a whole page is about 2.8x slower. Inside Docling, `heb_rivok` runs at
the stock model's speed. The rest of the page time is Docling's layout model. Page times
vary a lot with the amount of text: from 5 s to 127 s per page for this engine.

## What Docling's layout and reading-order models do with right-to-left pages

`bench/rtl_layout.py` over the document JSON of every run:

| Docling + | side-by-side block pairs | read left block first | pages read left-first | headings per page | words in 1-2-word items |
|---|---:|---:|---:|---:|---:|
| this engine | 192 | 83% | 21/27 | 1.4 | 0.3% |
| Tesseract `heb_rivok` | 1642 | 93% | 34/34 | 2.2 | 1.5% |
| Tesseract `heb_rivok_lite+eng` | 1389 | 92% | 31/33 | 2.3 | 1.4% |
| Tesseract `heb+eng` | 2455 | 90% | 32/34 | 2.2 | 2.3% |

- **Column order: left to right.** The rule-based reading-order model sorts blocks that sit
  side by side left first (`PageElement.__lt__`: `self.l < other.l`), with no RTL option.
  Two-column Hebrew pages come out left column first, whatever the OCR engine. Within each
  block the order is the OCR engine's, so this engine's paragraphs read correctly; the
  block order does not. On the synthetic page `tests/data/two_column.png` the engine emits
  the right column's line first on every row, and Docling still prints the left column's
  paragraph first.
- **Fragments.** Cells that overlap no layout cluster become one text item each, and those
  items are also ordered left to right. With Tesseract (one cell per word), lines the layout
  model missed turn into reversed one-word items. This engine's cells are whole lines, so
  an orphan line stays a readable line.
- **Tables are mirrored.** In every table, column 0 is the leftmost column, so the
  Markdown table starts with the column a Hebrew reader reads last. Cell text comes from
  the table model's tokens in token order. In our tests on born-digital Hebrew reports,
  Docling's own text layer (docling-parse) reversed the word order of about half the
  multi-word Hebrew table cells and split words and lines into pieces (the regression of
  docling#4018); this engine's cells read in order. No HaZefira page got a table.
- **Headings** are found visually and work on Hebrew, all at one level (heading hierarchy
  is off by default). The 1874 mastheads and boxed price lists are read as text.
