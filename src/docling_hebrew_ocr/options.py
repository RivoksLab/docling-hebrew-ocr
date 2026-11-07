# SPDX-License-Identifier: Apache-2.0
"""Options for the Hebrew OCR engine (``kind = "hebrew"``)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from docling.datamodel.pipeline_options import OcrOptions
from pydantic import ConfigDict, Field


class HebrewOcrOptions(OcrOptions):
    """paddleocr-hebrew inside Docling.

    One model reads Hebrew, Latin letters and digits together, so ``lang`` only
    validates the request: ``heb``/``eng`` (or ``iso:he``/``iso:en``) are accepted,
    anything else is an error.
    """

    kind: ClassVar[Literal["hebrew"]] = "hebrew"

    lang: list[str] = ["heb", "eng"]

    models_dir: Path | None = Field(
        default=None,
        description=(
            "A local copy of the Hugging Face repo Rivok/paddleocr-hebrew (needs"
            " config.json, charset_v2f.txt, word-det/ and server-svtrv2/; line-det/ for"
            " pipeline='line'). When unset: $DOCLING_HEBREW_OCR_MODELS, then"
            " <artifacts_path>/paddleocr-hebrew, then a download into Docling's cache."
        ),
    )
    pipeline: Literal["word", "line"] = Field(
        default="word",
        description=(
            "'word' (default): word-level detector, best on clean pages. 'line':"
            " line-level detector, better on degraded scans and dense layouts."
        ),
    )
    rec_mode: Literal["cascade", "ctc", "nrtr"] = Field(
        default="cascade",
        description=(
            "'cascade' (default): fast CTC head, NRTR fallback for crops with digits,"
            " Latin or low CTC confidence. 'ctc': fastest, worse on mixed text."
            " 'nrtr': slowest, every crop through NRTR."
        ),
    )
    threads: int | None = Field(
        default=None,
        ge=1,
        description="ONNX Runtime intra-op threads. Default: accelerator_options.num_threads.",
    )
    split_columns: bool = Field(
        default=True,
        description=(
            "Split each row of words at wide gaps (column gutters, table gaps) so one"
            " line cell never spans two columns."
        ),
    )
    gutter_factor: float = Field(
        default=2.0,
        gt=0,
        description="A gap wider than this many median word heights splits a row.",
    )
    emit_word_cells: bool = Field(
        default=True,
        description=(
            "Also add OCR word cells to the page. Docling's table model matches words"
            " to table cells, so tables get word-level tokens instead of whole lines."
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description=(
            "Confidence written on every cell. paddleocr-hebrew 0.1.0 reports no"
            " confidences, so this is a constant, and the page's ocr_score is set to"
            " NaN (unknown) instead of the mean of these constants."
        ),
    )
    max_image_side: int = Field(
        default=6000,
        ge=256,
        description=(
            "Upper limit, in pixels, for the long side of an image sent to the"
            " recognizer. An image file without DPI metadata is taken as 72 dpi, so"
            " scale=3 would triple a 300 dpi scan; the limit lowers the scale instead."
        ),
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
