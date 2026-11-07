# SPDX-License-Identifier: Apache-2.0
"""``HebrewOcrModel``: paddleocr-hebrew as a Docling OCR engine."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from pathlib import Path
from unittest import mock

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.datamodel.settings import settings
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.ocr_language import OcrLanguage, OcrLanguageSupport
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from paddleocr_hebrew.heb_postprocess import fix_geresh_yod_token

from .lines import build_lines
from .options import HebrewOcrOptions

_log = logging.getLogger(__name__)

HF_REPO = "Rivok/paddleocr-hebrew"
MODELS_ENV = "DOCLING_HEBREW_OCR_MODELS"
_MODEL_FOLDER = "paddleocr-hebrew"
_COMMON_FILES = ["config.json", "charset_v2f.txt", "server-svtrv2/*"]
_DETECTOR = {"word": "word-det", "line": "line-det"}

# Native codes the user may write in `lang`, and the tag each stands for
_NATIVE = {"heb": "he", "he": "he", "eng": "en", "en": "en"}


def allow_patterns(pipeline: str = "word") -> list[str]:
    """Files of the Hugging Face repo that one pipeline needs."""
    return [*_COMMON_FILES, f"{_DETECTOR[pipeline]}/*"]


class HebrewOcrModel(BaseOcrModel):
    """Hebrew (plus Latin and digits) OCR with paddleocr-hebrew.

    Emits one ``TextCell`` per text line, in reading order for a right-to-left page:
    rows top to bottom, the pieces of a row (columns, table gaps) right to left, and
    the words of a line in logical order. Docling keeps that order inside each layout
    cluster. With ``emit_word_cells`` the words are added as OCR word cells as well,
    which Docling's table model prefers for table tokens.
    """

    multiple_languages = True

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: HebrewOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: HebrewOcrOptions
        self.scale = self.options.scale
        self.reader = None
        if not self.enabled:
            return

        self.resolve_ocr_languages()  # raises on a language the model cannot read
        models = self._models_dir(artifacts_path)
        threads = self.options.threads or accelerator_options.num_threads
        self.reader = _load_reader(models, self.options.pipeline, threads, _providers(accelerator_options))

    # ---------------------------------------------------------------- languages
    def supported_ocr_languages(self) -> OcrLanguageSupport:
        return OcrLanguageSupport(bcp47=["en", "he"], native=sorted(_NATIVE))

    def map_ocr_language(self, language: OcrLanguage) -> str | list[str]:
        if language.is_passthrough():
            code = _NATIVE.get((language.native or "").lower())
            if code is not None:
                return code
        elif language.short_tag() in ("he", "en"):
            return language.short_tag()
        raise OcrLanguageNotSupportedError(
            self._engine_name,
            language.tag(),
            supported=self.supported_ocr_languages(),
            detail="paddleocr-hebrew reads Hebrew, Latin letters and digits only.",
        )

    # ---------------------------------------------------------------- models
    def _models_dir(self, artifacts_path: Path | None) -> Path:
        if self.options.models_dir is not None:
            return Path(self.options.models_dir)
        if os.environ.get(MODELS_ENV):
            return Path(os.environ[MODELS_ENV])
        if artifacts_path is not None and (Path(artifacts_path) / _MODEL_FOLDER).is_dir():
            return Path(artifacts_path) / _MODEL_FOLDER
        return self.download_models(pipeline=self.options.pipeline)

    @staticmethod
    def download_models(local_dir: Path | None = None, pipeline: str = "word") -> Path:
        """Download the model files one pipeline needs (about 190 MB for 'word')."""
        from huggingface_hub import snapshot_download

        local_dir = Path(local_dir or settings.cache_dir / "models" / _MODEL_FOLDER)
        if not _missing_files(local_dir, pipeline):
            return local_dir  # already there: no network call
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(HF_REPO, local_dir=str(local_dir), allow_patterns=allow_patterns(pipeline))
        return local_dir

    # ---------------------------------------------------------------- OCR
    def _scale_for(self, rect: BoundingBox) -> float:
        long_side = max(rect.width, rect.height)
        if long_side * self.scale <= self.options.max_image_side:
            return self.scale
        return self.options.max_image_side / long_side

    def _ocr_rect(self, page: Page, rect: BoundingBox) -> list[tuple[TextCell, list[TextCell]]]:
        """OCR one region: a line cell and its word cells per line, in page coordinates."""
        assert page._backend is not None and page.size is not None
        if rect.coord_origin != CoordOrigin.TOPLEFT:
            rect = rect.to_top_left_origin(page_height=page.size.height)
        scale = self._scale_for(rect)
        image = page._backend.get_page_image(scale=scale, cropbox=rect)
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        # Pixel -> page points; the image may be a pixel off rect*scale after rounding
        sx = rect.width / image.width if image.width else 1 / scale
        sy = rect.height / image.height if image.height else 1 / scale
        result = self.reader.read_array(bgr, rec_mode=self.options.rec_mode)
        lines = build_lines(
            result["words"],
            split_columns=self.options.split_columns,
            gutter_factor=self.options.gutter_factor,
        )

        def cell(text: str, box) -> TextCell:
            x1, y1, x2, y2 = box
            bbox = BoundingBox(
                l=rect.l + x1 * sx,
                t=rect.t + y1 * sy,
                r=rect.l + x2 * sx,
                b=rect.t + y2 * sy,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            return TextCell(
                index=0,  # set by the caller
                text=text,
                orig=text,
                from_ocr=True,
                confidence=self.options.confidence,
                rect=BoundingRectangle.from_bounding_box(bbox),
            )

        out = []
        for line in lines:
            words = []
            if self.options.emit_word_cells:
                words = [cell(fix_geresh_yod_token(w["text"]), w["bbox"]) for w in line.words]
            out.append((cell(line.text, line.bbox), words))
        return out

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                lines: list[tuple[TextCell, list[TextCell]]] = []
                for rect in ocr_rects:
                    if rect.area() == 0:
                        continue
                    lines.extend(self._ocr_rect(page, rect))
                line_cells = [line for line, _ in lines]
                for i, line in enumerate(line_cells):
                    line.index = i

                self.post_process_cells(line_cells, page, conv_res)
                self._add_word_cells(page, lines)
                # The library reports no confidences, so the constant on the cells says
                # nothing; leave the page's OCR score unknown rather than perfect.
                if line_cells:
                    conv_res.confidence.pages[page.page_no].ocr_score = math.nan

            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @staticmethod
    def _add_word_cells(page: Page, lines: list[tuple[TextCell, list[TextCell]]]) -> None:
        """Add the words of every OCR line that survived the merge as OCR word cells.

        In full-page mode every line survives. In the other modes a line that overlaps
        native text may have been dropped, and its words are dropped with it.
        """
        if page.parsed_page is None:
            return
        kept = {id(c) for c in page.parsed_page.textline_cells}
        words = [w for line, ws in lines if id(line) in kept for w in ws]
        if not words:
            return
        start = len(page.parsed_page.word_cells)
        for i, w in enumerate(words):
            w.index = start + i
        page.parsed_page.word_cells = [*page.parsed_page.word_cells, *words]
        page.parsed_page.has_words = True

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return HebrewOcrOptions


_REQUIRED = ["charset_v2f.txt", "server-svtrv2/ctc.onnx", "server-svtrv2/nrtr-encoder.onnx",
             "server-svtrv2/nrtr-decstep.onnx"]


def _missing_files(models_dir: Path, pipeline: str) -> list[str]:
    need = [*_REQUIRED, f"{_DETECTOR[pipeline]}/det.onnx"]
    return [rel for rel in need if not (Path(models_dir) / rel).is_file()]


def _providers(accelerator_options: AcceleratorOptions) -> list[str]:
    import onnxruntime as ort

    device = str(accelerator_options.device).lower()
    if ("cuda" in device or "auto" in device) and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_reader(models_dir: Path, pipeline: str, threads: int, providers: list[str]):
    """Build a HebrewOCR with a thread limit on every ONNX session.

    paddleocr-hebrew 0.1.0 takes no SessionOptions, so the limit is injected by
    wrapping ``onnxruntime.InferenceSession`` while the sessions are created.
    """
    import onnxruntime as ort
    from paddleocr_hebrew import HebrewOCR

    missing = _missing_files(models_dir, pipeline)
    if missing:
        raise FileNotFoundError(
            f"{models_dir / missing[0]} is missing. Point HebrewOcrOptions.models_dir at a copy"
            f" of the Hugging Face repo {HF_REPO} (patterns: {allow_patterns(pipeline)})."
        )

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(threads))
    so.inter_op_num_threads = 1
    base = ort.InferenceSession

    class _Session(base):
        def __init__(self, path, sess_options=None, providers=None, **kw):
            super().__init__(path, sess_options or so, providers=providers, **kw)

    with mock.patch.object(ort, "InferenceSession", _Session):
        factory = HebrewOCR.line if pipeline == "line" else HebrewOCR.word
        return factory(str(models_dir), providers=providers)
