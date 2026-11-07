# SPDX-License-Identifier: Apache-2.0
"""Turn recognized words into text lines in reading order. Pure functions, no models.

A *word* is ``{"bbox": [x1, y1, x2, y2], "text": str}`` in image pixels (top-left
origin), its text in logical Unicode order, as ``HebrewOCR.read_array(...)["words"]``
returns it.

``paddleocr_hebrew.HebrewOCR.read_array`` also returns lines, but it groups words into
page-wide rows: on a two-column page one "line" holds the same row of both columns.
Docling assigns each line cell to the one layout cluster it overlaps most, so such a
line would put text of the left column into the right one. Here every row is first
split at wide horizontal gaps (column gutters, table gaps), the same rule as
``ocrmypdf_hebrew.layout``; the word order inside each piece comes from
``paddleocr_hebrew.heb_postprocess.order_row`` unchanged.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from paddleocr_hebrew.heb_postprocess import (
    cluster_rows,
    fix_geresh_yod_text,
    order_row,
    row_directions,
)

# A gap wider than this many median word heights splits a row into separate lines
# (column gutter, table gap). Normal Hebrew word spacing is about 0.3-0.7 heights.
GUTTER_FACTOR = 2.0

# Boxes at least this many times taller than wide hold vertical text (margin notes).
# The recognizer reads horizontally and returns garbage for them, so they are dropped
# unless they hold a single character (a narrow letter or digit is tall and thin).
VERTICAL_RATIO = 2.5

_HEB = re.compile(r"[֐-׿]")


@dataclass
class Line:
    """One output line: logical-order text, its pixel box, and its words in logical order."""

    text: str
    bbox: tuple[float, float, float, float]
    words: list[dict] = field(default_factory=list)


def is_vertical_garbage(word: dict) -> bool:
    x1, y1, x2, y2 = word["bbox"]
    w, h = max(1, x2 - x1), y2 - y1
    return h >= VERTICAL_RATIO * w and len(word["text"].replace(" ", "")) >= 2


def split_at_gutters(row: list[dict], median_h: float, factor: float = GUTTER_FACTOR) -> list[list[dict]]:
    """Split one page-wide row wherever the horizontal gap between neighbouring words
    is wider than ``factor`` x the median word height. Segments come back left to right."""
    xs = sorted(row, key=lambda w: w["bbox"][0])
    segs, cur = [], [xs[0]]
    right = xs[0]["bbox"][2]
    for w in xs[1:]:
        if w["bbox"][0] - right > factor * median_h:
            segs.append(cur)
            cur = [w]
        else:
            cur.append(w)
        right = max(right, w["bbox"][2])
    segs.append(cur)
    return segs


def build_lines(
    words: list[dict],
    *,
    split_columns: bool = True,
    gutter_factor: float = GUTTER_FACTOR,
    row_tolerance: float = 0.5,
    drop_vertical: bool = True,
) -> list[Line]:
    """Group words into lines, in reading order for a right-to-left page.

    Rows run top to bottom. A row that holds Hebrew lists its pieces right to left
    (right column first); a row without Hebrew lists them left to right. Inside each
    piece the words are in logical order, with embedded English or number runs kept
    in their own order (``order_row``).
    """
    words = [w for w in words if w.get("text", "").strip()]
    if drop_vertical:
        words = [w for w in words if not is_vertical_garbage(w)]
    if not words:
        return []
    median_h = statistics.median(w["bbox"][3] - w["bbox"][1] for w in words)
    rows = cluster_rows(words, row_tolerance)
    lines: list[Line] = []
    for row, row_dir in zip(rows, row_directions(rows)):
        segs = split_at_gutters(row, median_h, gutter_factor) if split_columns else [row]
        rtl_row = any(_HEB.search(w["text"]) for w in row) and row_dir != "ltr"
        if rtl_row:
            segs.reverse()
        for seg in segs:
            ordered = order_row(seg, row_dir)
            text = fix_geresh_yod_text(" ".join(w["text"] for w in ordered).strip())
            if not text:
                continue
            bbox = (
                min(w["bbox"][0] for w in ordered),
                min(w["bbox"][1] for w in ordered),
                max(w["bbox"][2] for w in ordered),
                max(w["bbox"][3] for w in ordered),
            )
            lines.append(Line(text=text, bbox=bbox, words=ordered))
    return lines

