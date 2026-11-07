# SPDX-License-Identifier: Apache-2.0
"""Line assembly on fixed word boxes: no models, runs in milliseconds."""

from docling_hebrew_ocr.lines import build_lines

MIXED_OK = {
    "אוניברסיטת חיפה 2024 University of Haifa",
    "אוניברסיטת חיפה University of Haifa 2024",
}


def words(spec, y=100, h=30):
    """(text, x1, x2) as the words sit on the page -> word dicts."""
    return [{"bbox": [x1, y, x2, y + h], "text": t} for t, x1, x2 in spec]


def test_mixed_line_keeps_logical_order():
    # The boxes of the sample page's line "אוניברסיטת חיפה 2024 University of Haifa"
    row = words([("University", 470, 600), ("of", 610, 640), ("Haifa", 650, 730),
                 ("2024", 740, 800), ("חיפה", 810, 880), ("אוניברסיטת", 890, 1060)])
    lines = build_lines(row)
    assert len(lines) == 1
    assert lines[0].text in MIXED_OK


def test_two_columns_become_two_lines_right_column_first():
    right = [("בטור", 1060, 1160), ("הימני", 950, 1045), ("מספרת", 840, 940)]
    left = [("בטור", 460, 560), ("השמאלי", 330, 450), ("הראשונה", 160, 320)]
    lines = build_lines(words(right + left))
    assert [ln.text for ln in lines] == ["בטור הימני מספרת", "בטור השמאלי הראשונה"]
    assert lines[0].bbox[0] > lines[1].bbox[2]  # the first line lies right of the second


def test_two_column_rows_interleave_but_stay_in_column_order():
    page = []
    for i, y in enumerate((100, 160, 220)):
        page += words([(f"ימין{i}", 900, 1000), ("טקסט", 780, 880)], y=y)
        page += words([(f"שמאל{i}", 400, 500), ("טקסט", 280, 380)], y=y)
    texts = [ln.text for ln in build_lines(page)]
    assert texts == ["ימין0 טקסט", "שמאל0 טקסט", "ימין1 טקסט", "שמאל1 טקסט", "ימין2 טקסט", "שמאל2 טקסט"]


def test_without_column_split_a_row_is_one_line():
    right = [("ימין", 900, 1000)]
    left = [("שמאל", 100, 200)]
    lines = build_lines(words(right + left), split_columns=False)
    assert [ln.text for ln in lines] == ["ימין שמאל"]


def test_english_row_pieces_run_left_to_right():
    row = words([("Total", 100, 180), ("due", 190, 240), ("Page", 900, 970), ("3", 980, 1000)])
    assert [ln.text for ln in build_lines(row)] == ["Total due", "Page 3"]


def test_vertical_margin_text_is_dropped():
    page = words([("שורה", 300, 400), ("רגילה", 180, 290)])
    page.append({"bbox": [20, 50, 40, 400], "text": "Group D"})  # tall and thin
    assert [ln.text for ln in build_lines(page)] == ["שורה רגילה"]


def test_geresh_yod_repair_applies_to_the_line():
    row = words([("עמי", 300, 360), ("131", 200, 260)])
    assert build_lines(row)[0].text == "עמ' 131"
