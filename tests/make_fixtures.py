#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate tests/data/two_column.png, a synthetic two-column Hebrew page.

The text is written for this test (no third-party content). Needs Pillow built with
libraqm (right-to-left shaping) and a font with Hebrew, e.g. DejaVu Sans or Noto
Sans Hebrew. The PNG is committed, so the tests themselves need neither.

    python tests/make_fixtures.py [--font /path/to/font.ttf]
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

HERE = Path(__file__).resolve().parent
TITLE = "עלון השכונה"
RIGHT = [  # read first on a Hebrew page
    "בטור הימני השורה הראשונה מספרת",
    "על הספרייה העירונית שנפתחה",
    "מחדש אחרי שיפוץ ארוך בשנת 2023",
    "ובה אלפי ספרים חדשים לילדים",
]
LEFT = [
    "בטור השמאלי השורה הראשונה",
    "עוסקת בגינה הקהילתית שבה",
    "מגדלים ירקות ופרחים לאורך",
    "כל ימות השנה בעזרת מתנדבים",
]
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default=next((f for f in FONTS if Path(f).exists()), None))
    a = ap.parse_args()
    if not features.check("raqm"):
        raise SystemExit("Pillow without libraqm cannot shape right-to-left text")
    W, H = 1240, 900  # 150 dpi, 8.3 in wide
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(a.font, 44)
    body = ImageFont.truetype(a.font, 30)
    d.text((W // 2, 90), TITLE, font=big, fill="black", anchor="mm", direction="rtl")
    margin, gutter = 80, 120
    col_w = (W - 2 * margin - gutter) // 2
    right_edge = {"right": W - margin, "left": margin + col_w}
    for col, lines in (("right", RIGHT), ("left", LEFT)):
        y = 200
        for line in lines:
            d.text((right_edge[col], y), line, font=body, fill="black", anchor="ra", direction="rtl")
            assert d.textlength(line, font=body, direction="rtl") <= col_w, line
            y += 56
    out = HERE / "data" / "two_column.png"
    out.parent.mkdir(exist_ok=True)
    img.save(out, dpi=(150, 150), optimize=True)
    print(out, img.size)


if __name__ == "__main__":
    main()
