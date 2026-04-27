"""Stitch a 3-row comparison grid:
    Row 1: procedural render
    Row 2: FLUX.1 Kontext [dev] photoreal target (run on bender)
    Row 3: Hunyuan3D-2mv reconstruction (rendered from the same views)

Uses 4 views per row: front, left, back, iso (iso for sanity check; the
Hunyuan3D-2mv pipeline conditions only on front/left/back).
"""

from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont


VIEWS = ["front", "left", "back", "iso"]
FACE_VIEWS = ["face_front", "face_3q", "face_left", "face_iso"]
LABELS = {
    "front": "Front", "left": "Left", "back": "Back", "iso": "3/4 Iso",
    "face_front": "Face — Front", "face_3q": "Face — 3/4",
    "face_left": "Face — Left", "face_iso": "Face — Iso",
}

ROWS = [
    # filename suffix, row label, set: "body" or "face"
    ("",            "Procedural (ours)",            "body"),
    ("_photoreal",  "FLUX Kontext target",          "body"),
    ("_hy3d",       "Hunyuan3D-2mv reconstruction", "body"),
    ("",            "Procedural face",              "face"),
    ("_photoreal",  "FLUX Kontext face",            "face"),
]


def _font(size):
    try:
        return ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _label(img, text, sub=None):
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    d.rectangle([(0, 0), (out.width, 56)], fill=(0, 0, 0))
    d.text((14, 8), text, fill=(255, 255, 255), font=_font(28))
    if sub:
        d.text((14, out.height - 30), sub, fill=(220, 220, 220), font=_font(18))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="prefix", required=True,
                    help="prefix used by triview_capture (e.g. screenshots/triview/c3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-height", type=int, default=512)
    args = ap.parse_args()

    rendered_rows = []
    for tag, row_label, kind in ROWS:
        view_set = FACE_VIEWS if kind == "face" else VIEWS
        panels = []
        for v in view_set:
            # Hunyuan3D row uses the same view names but only generates from front/left/back.
            # iso is rendered too via tools.glb_render and shares the suffix.
            path = f"{args.prefix}{tag}_{v}.png" if tag.startswith("_hy3d") \
                else f"{args.prefix}_{v}{tag}.png"
            if not os.path.exists(path):
                # Skip missing photoreal-iso (we don't generate iso for hy3d-input pipeline)
                # Substitute with a blank labelled panel so the grid alignment stays.
                blank = Image.new("RGB", (args.row_height, args.row_height), (240, 240, 240))
                d = ImageDraw.Draw(blank)
                d.text((20, 20), f"missing\n{path}", fill=(120, 120, 120), font=_font(18))
                img = blank
            else:
                img = Image.open(path)
                ratio = args.row_height / img.height
                img = img.resize((int(img.width * ratio), args.row_height))
            panels.append(_label(img, LABELS[v], sub=row_label))
        rendered_rows.append(panels)

    n_cols = max(len(r) for r in rendered_rows)
    panel_w = max(p.width for r in rendered_rows for p in r)
    grid = Image.new("RGB",
                     (panel_w * n_cols, args.row_height * len(ROWS)),
                     (255, 255, 255))
    for r, row in enumerate(rendered_rows):
        for c, p in enumerate(row):
            grid.paste(p, (c * panel_w, r * args.row_height))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    grid.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
