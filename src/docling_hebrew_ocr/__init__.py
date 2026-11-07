# SPDX-License-Identifier: Apache-2.0
"""Hebrew OCR engine for Docling, backed by paddleocr-hebrew.

    from docling_hebrew_ocr import HebrewOcrOptions
    pipeline_options.allow_external_plugins = True
    pipeline_options.ocr_options = HebrewOcrOptions(mode=OcrMode.FULL_PAGE)
"""

from .options import HebrewOcrOptions

__version__ = "0.1.0"
__all__ = ["HebrewOcrModel", "HebrewOcrOptions"]


def __getattr__(name):
    if name == "HebrewOcrModel":  # imports the models' runtime only when asked for
        from .model import HebrewOcrModel

        return HebrewOcrModel
    raise AttributeError(name)
