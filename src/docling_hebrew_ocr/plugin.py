# SPDX-License-Identifier: Apache-2.0
"""Docling plugin entry point (``[project.entry-points."docling"]``)."""


def ocr_engines():
    from .model import HebrewOcrModel

    return {"ocr_engines": [HebrewOcrModel]}
