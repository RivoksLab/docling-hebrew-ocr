# SPDX-License-Identifier: Apache-2.0
"""Docling end to end: plugin registration, scans, full-page OCR over a bad text layer.

The tests marked ``slow`` run the OCR and Docling's layout models on a CPU (about a
minute each) and need $DOCLING_HEBREW_OCR_MODELS; the others take a second.
"""

import ctypes
from difflib import SequenceMatcher
from pathlib import Path

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.factories import get_ocr_factory
from PIL import Image

from docling_hebrew_ocr import HebrewOcrOptions
from docling_hebrew_ocr.model import HebrewOcrModel

DATA = Path(__file__).resolve().parent / "data"
MIXED_OK = (
    "אוניברסיטת חיפה 2024 University of Haifa",
    "אוניברסיטת חיפה University of Haifa 2024",
)
TWO_COL_RIGHT = [
    "בטור הימני השורה הראשונה מספרת",
    "על הספרייה העירונית שנפתחה",
    "מחדש אחרי שיפוץ ארוך בשנת 2023",
    "ובה אלפי ספרים חדשים לילדים",
]
TWO_COL_LEFT = [
    "בטור השמאלי השורה הראשונה",
    "עוסקת בגינה הקהילתית שבה",
    "מגדלים ירקות ופרחים לאורך",
    "כל ימות השנה בעזרת מתנדבים",
]


def test_plugin_is_registered_under_kind_hebrew():
    factory = get_ocr_factory(allow_external_plugins=True)
    assert "hebrew" in factory.registered_kind
    assert factory.create_options(kind="hebrew").__class__ is HebrewOcrOptions


def test_unsupported_language_is_an_error():
    opts = HebrewOcrOptions(lang=["deu"])
    with pytest.raises(OcrLanguageNotSupportedError):
        HebrewOcrModel(enabled=True, artifacts_path=None, options=opts,
                       accelerator_options=AcceleratorOptions(device="cpu"))


def converter(models_dir, threads, **ocr):
    po = PdfPipelineOptions()
    po.allow_external_plugins = True
    po.generate_parsed_pages = True
    po.do_table_structure = False
    po.accelerator_options = AcceleratorOptions(num_threads=threads, device="cpu")
    po.ocr_options = HebrewOcrOptions(models_dir=models_dir, threads=threads, **ocr)
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=po),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=po),
    })


def image_pdf(png: Path, out: Path, dpi: int = 150) -> Path:
    """An image-only PDF (a scan): the page is one bitmap, no text layer."""
    Image.open(png).convert("RGB").save(out, resolution=dpi)
    return out


def pdf_with_bad_text_layer(png: Path, out: Path, dpi: int = 150) -> Path:
    """The page image plus an invisible text layer of garbage, as a broken OCR layer would be."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    pil = Image.open(png).convert("RGB")
    w, h = pil.width * 72 / dpi, pil.height * 72 / dpi
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(w, h)
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(pil))
    image.set_matrix(pdfium.PdfMatrix().scale(w, h))
    page.insert_obj(image)
    font = raw.FPDFText_LoadStandardFont(pdf, b"Helvetica")
    for i, junk in enumerate(["tisharb arb mihole", "xjq brokenlayer zzv", "qqq garbledtext www"]):
        text = raw.FPDFPageObj_CreateTextObj(pdf, font, 14.0)
        buf = (junk + "\x00").encode("utf-16-le")
        raw.FPDFText_SetText(text, ctypes.cast(ctypes.c_char_p(buf), ctypes.POINTER(raw.FPDF_WCHAR)))
        raw.FPDFTextObj_SetTextRenderMode(text, raw.FPDF_TEXTRENDERMODE_INVISIBLE)
        raw.FPDFPageObj_Transform(text, 1, 0, 0, 1, w * 0.4, h - 40 - 30 * i)
        raw.FPDFPage_InsertObject(page, text)
    page.gen_content()
    pdf.save(str(out))
    return out


@pytest.mark.slow
def test_scanned_pdf_reads_mixed_line_in_logical_order(models_dir, threads, tmp_path):
    pdf = image_pdf(DATA / "sample_page.png", tmp_path / "scan.pdf")
    res = converter(models_dir, threads, mode=OcrMode.FULL_PAGE).convert(pdf)
    md = res.document.export_to_markdown()
    assert any(m in md for m in MIXED_OK), md
    assert "בראשית ברא אלהים את השמים ואת הארץ" in md

    page = res.pages[0]
    cells = page.parsed_page.textline_cells
    assert cells and all(c.from_ocr for c in cells)
    texts = [c.text for c in cells]
    assert texts[0] == "בראשית"  # the heading, top right, comes first
    assert any(t in MIXED_OK for t in texts)  # one cell per line
    for c in cells:  # page coordinates, inside the page
        box = c.rect.to_bounding_box().to_top_left_origin(page.size.height)
        assert 0 <= box.l < box.r <= page.size.width + 1
        assert 0 <= box.t < box.b <= page.size.height + 1
    assert page.parsed_page.word_cells and all(w.from_ocr for w in page.parsed_page.word_cells)


@pytest.mark.slow
def test_layout_region_mode_maps_crops_back_to_the_page(models_dir, threads, tmp_path):
    """Docling's default mode OCRs layout regions; boxes must land where the text is.

    Text is not compared exactly: on a crop the detector works at another scale than on
    the whole page, and reads a little worse (one reason FULL_PAGE is recommended).
    """
    pdf = image_pdf(DATA / "sample_page.png", tmp_path / "scan.pdf")  # 1100 px at 150 dpi = 528 pt
    res = converter(models_dir, threads).convert(pdf)
    assert "אוניברסיטת חיפה 2024 University" in res.document.export_to_markdown()
    page = res.pages[0]
    cells = {c.text: c.rect.to_bounding_box().to_top_left_origin(page.size.height)
             for c in page.parsed_page.textline_cells}
    assert len(cells) == 7  # the heading and six lines
    for text, box in cells.items():  # right-aligned text: every line ends near x = 1060 px
        assert abs(box.r - 1060 * 72 / 150) < 8, (text, box)
    mixed = next(box for text, box in cells.items() if "University" in text)
    assert abs((mixed.t + mixed.b) / 2 - 376 * 72 / 150) < 8, mixed  # line centre at y = 376 px


@pytest.mark.slow
def test_two_column_image_never_mixes_columns(models_dir, threads):
    res = converter(models_dir, threads, mode=OcrMode.FULL_PAGE, scale=1.0).convert(DATA / "two_column.png")
    page = res.pages[0]
    mid = page.size.width / 2
    cells = []
    for c in page.parsed_page.textline_cells:
        box = c.rect.to_bounding_box().to_top_left_origin(page.size.height)
        cells.append((c.text, box))
    body = [(t, b) for t, b in cells if "השכונה" not in t]  # the centred title spans both
    assert all(b.r < mid or b.l > mid for _, b in body), "a line cell crosses the gutter"
    right = [t for t, b in body if b.l > mid]
    left = [t for t, b in body if b.r < mid]
    # every line of each column is its own cell, read top to bottom (allowing OCR slips)
    assert len(right) == len(TWO_COL_RIGHT) and len(left) == len(TWO_COL_LEFT), cells
    for got, want in [*zip(right, TWO_COL_RIGHT), *zip(left, TWO_COL_LEFT)]:
        assert SequenceMatcher(None, got, want).ratio() > 0.85, (got, want)
    # on each row the right column's line comes first
    order = [t for t, _ in body]
    assert all(order.index(r) < order.index(lt) for r, lt in zip(right, left)), order

    md = res.document.export_to_markdown()
    for column in (right, left):  # each column stays one run of text in the Markdown
        assert " ".join(column) in md, md


@pytest.mark.slow
def test_full_page_mode_replaces_a_bad_text_layer(models_dir, threads, tmp_path):
    pdf = pdf_with_bad_text_layer(DATA / "sample_page.png", tmp_path / "bad_layer.pdf")
    md = converter(models_dir, threads, mode=OcrMode.FULL_PAGE).convert(pdf).document.export_to_markdown()
    assert "brokenlayer" not in md and "garbledtext" not in md
    assert any(m in md for m in MIXED_OK), md
