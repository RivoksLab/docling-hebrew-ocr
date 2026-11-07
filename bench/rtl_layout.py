#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What Docling's layout and reading-order models do with right-to-left pages.

Reads the document JSON files run_bench.py saved and reports, per set and system:
  - column order: of all pairs of text blocks that sit side by side (vertical overlap,
    no horizontal overlap), the share Docling reads left block first. A right-to-left
    reader would read the right block first (share near 0); a left-to-right one gives 1.
  - tables: how many, and whether table column 0 is the leftmost column (a mirrored
    table for Hebrew, whose first column is on the right).
  - headings: section headers and titles per page.
  - fragments: the share of words that sit in text items of one or two words. Cells
    that fall outside every layout cluster become one item each, and a word-level
    engine's orphan words then read as single-word items in left-to-right order.
  - table cell text (sets with references only): of the table cells holding two or more
    Hebrew words, how many read as a phrase of the reference in the reference's word
    order, and how many only with their words reversed.

    python bench/rtl_layout.py --out OUT [--pages DIR]
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docling_core.types.doc import DoclingDocument, DocItemLabel, TableItem, TextItem

BLOCKS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM,
          DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE, DocItemLabel.CAPTION,
          DocItemLabel.FOOTNOTE}


def boxes_in_order(doc: DoclingDocument):
    """(label, top-left box) of every text block, in Docling's reading order."""
    out = []
    for item, _ in doc.iterate_items():
        if isinstance(item, TextItem) and item.label in BLOCKS and item.prov:
            p = item.prov[0]
            h = doc.pages[p.page_no].size.height
            out.append((item.label, p.bbox.to_top_left_origin(h)))
    return out


def fragment_words(doc: DoclingDocument):
    """(words in items of at most two words, all words) over the text items."""
    small = total = 0
    for item, _ in doc.iterate_items():
        if isinstance(item, TextItem) and item.label in BLOCKS:
            n = len(item.text.split())
            total += n
            small += n if n <= 2 else 0
    return small, total


def side_by_side_pairs(boxes):
    left_first = right_first = 0
    for i, (_, a) in enumerate(boxes):
        for _, b in boxes[i + 1:]:
            v_overlap = min(a.b, b.b) - max(a.t, b.t)
            if v_overlap <= 0.3 * min(a.height, b.height):
                continue
            if a.r <= b.l:
                left_first += 1
            elif b.r <= a.l:
                right_first += 1
    return left_first, right_first


def table_stats(doc: DoclingDocument):
    n = mirrored = 0
    for item, _ in doc.iterate_items():
        if not isinstance(item, TableItem) or item.data.num_cols < 2:
            continue
        n += 1
        cols = collections.defaultdict(list)
        for c in item.data.table_cells:
            if c.bbox is not None and c.col_span == 1:
                cols[c.start_col_offset_idx].append((c.bbox.l + c.bbox.r) / 2)
        if 0 in cols and item.data.num_cols - 1 in cols:
            first = sum(cols[0]) / len(cols[0])
            last = sum(cols[item.data.num_cols - 1]) / len(cols[item.data.num_cols - 1])
            mirrored += first < last  # column 0 is the leftmost
    return n, mirrored


HEB_WORD = re.compile(r"[\u05d0-\u05ea]{2,}")


def cell_word_order(doc: DoclingDocument, ref: str):
    """(cells in reference order, cells only reversed) over multi-word Hebrew table cells."""
    ref_words = " " + " ".join(HEB_WORD.findall(ref)) + " "
    ok = rev = 0
    for item, _ in doc.iterate_items():
        if not isinstance(item, TableItem):
            continue
        for c in item.data.table_cells:
            words = HEB_WORD.findall(c.text)
            if len(words) < 2:
                continue
            fwd = f" {' '.join(words)} " in ref_words
            bwd = f" {' '.join(reversed(words))} " in ref_words
            ok += fwd
            rev += bwd and not fwd
    return ok, rev


def references(pages_dir) -> dict:
    if pages_dir is None:
        return {}
    from scoring import load_pages, reference

    return {e["id"]: reference(e) for e in load_pages(pages_dir) if reference(e) is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="run_bench.py's output folder")
    ap.add_argument("--pages", type=Path, help="the page set, for the table-cell word order check")
    a = ap.parse_args()
    refs = references(a.pages)
    print(f"{'set/system':24s} {'pages':>5s} {'pairs':>6s} {'left-first':>10s} {'pages L1st':>10s} "
          f"{'tables':>6s} {'col0=left':>9s} {'headers/p':>9s} {'cells fwd/rev':>13s} {'fragments':>9s}")
    for set_dir in [a.out]:
        for sys_dir in sorted(p for p in set_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            pages = lf = rf = pages_lf = pages_pairs = tables = mirrored = headers = fwd = rev = 0
            frag = words = 0
            for f in sorted(sys_dir.glob("*.doc.json")):
                doc = DoclingDocument.model_validate(json.loads(f.read_text(encoding="utf-8")))
                boxes = boxes_in_order(doc)
                l1, r1 = side_by_side_pairs(boxes)
                lf += l1
                rf += r1
                if l1 + r1:
                    pages_pairs += 1
                    pages_lf += l1 > r1
                page_id = f.name[: -len(".doc.json")]
                if page_id in refs:
                    o, r = cell_word_order(doc, refs[page_id])
                    fwd += o
                    rev += r
                fs, fw = fragment_words(doc)
                frag += fs
                words += fw
                t, m = table_stats(doc)
                tables += t
                mirrored += m
                headers += sum(1 for lab, _ in boxes if lab in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE))
                pages += 1
            if not pages:
                continue
            share = f"{100 * lf / (lf + rf):.0f}%" if lf + rf else "-"
            print(f"{set_dir.name + '/' + sys_dir.name:24s} {pages:5d} {lf + rf:6d} {share:>10s} "
                  f"{pages_lf:>4d}/{pages_pairs:<5d} {tables:6d} {mirrored:9d} {headers / pages:9.1f} {fwd:>6d}/{rev:<6d} {100 * frag / max(1, words):8.1f}%")


if __name__ == "__main__":
    main()
