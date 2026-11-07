# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """The paddleocr-hebrew model files, from $DOCLING_HEBREW_OCR_MODELS (tests skip without)."""
    path = os.environ.get("DOCLING_HEBREW_OCR_MODELS")
    if not path or not (Path(path) / "word-det" / "det.onnx").is_file():
        pytest.skip("set DOCLING_HEBREW_OCR_MODELS to a copy of Rivok/paddleocr-hebrew")
    return Path(path)


@pytest.fixture(scope="session")
def threads() -> int:
    return int(os.environ.get("TEST_THREADS", "2"))
