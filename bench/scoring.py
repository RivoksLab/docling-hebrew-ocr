# SPDX-License-Identifier: Apache-2.0
"""Word scoring for the page benchmark: the text is cut into words the same way for the
reference and for Docling's Markdown, then compared as word multisets (order-free F1)
and as word pairs (bigram F1, an order-sensitive supplement).

Words: HTML and Markdown markup removed, bidi controls and niqqud dropped, ׳ ״ ־ – —
folded to ' " - - -, a Hebrew letter followed by a digit (or the reverse) split, and
punctuation stripped from the edges of each word.
"""

import html
import json
import re
import unicodedata
from pathlib import Path

CTRL = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩﻿"))
FOLD = str.maketrans({"׳": "'", "״": '"', "־": "-", "–": "-", "—": "-"})


def tokens(text):
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))           # HTML tables
    t = re.sub(r"[|#*`]+|\$+", " ", t)                                 # markdown
    t = unicodedata.normalize("NFD", t.translate(CTRL))
    t = "".join(c for c in t if not unicodedata.category(c) == "Mn")
    t = unicodedata.normalize("NFC", t).translate(FOLD)
    t = re.sub(r"(?<=[א-ת])(?=\d)|(?<=\d)(?=[א-ת])", " ", t)   # טלפון0775 -> טלפון 0775
    out = []
    for w in t.split():
        w = w.strip(".,:;!?()[]{}\"'-–_/\\₪%*•")
        if w:
            out.append(w)
    return out


def load_pages(pages_dir):
    """The page set in `pages_dir`: manifest.json, images/ and refs/ (files found by name)."""
    d = Path(pages_dir)
    manifest = d / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"{d} has no manifest.json (see bench/run_bench.py for how to build the set)")
    pages = []
    for e in json.loads(manifest.read_text(encoding="utf-8")):
        pages.append({
            "id": e["id"],
            "image": d / "images" / Path(e["image"]).name,
            "ref": d / "refs" / Path(e["ref_txt"]).name if e.get("ref_txt") else None,
        })
    return pages


def reference(page):
    """The reference text of a page from load_pages(), or None."""
    return page["ref"].read_text(encoding="utf-8") if page["ref"] and page["ref"].exists() else None
