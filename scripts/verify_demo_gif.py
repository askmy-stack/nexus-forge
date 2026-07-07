#!/usr/bin/env python3
"""Verify scene 2 CLI frame has no text overlap in the center gap."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_demo_gif import (  # noqa: E402
    SCENE2_GAP_X0,
    SCENE2_GAP_X1,
    SCENE2_PANEL_H,
    SCENE2_PANEL_Y,
    TEXT_PAD,
    TEXT_SOURCE,
    TEXT_TOP,
    frame_text_summarize,
)

OUT = ROOT / "docs" / "assets" / "demo-frame-cli.png"


def _is_dark_text_pixel(r: int, g: int, b: int) -> bool:
    """Detect light body text pixels; ignore green arrow in gap."""
    if g > 170 and g > r + 60 and g > b + 30:
        return False
    return r > 130 and g > 130 and b > 130


def verify_cli_frame(frame: Image.Image) -> bool:
    rgb = frame.convert("RGB")
    text_y_start = SCENE2_PANEL_Y + TEXT_TOP
    text_y_end = SCENE2_PANEL_Y + SCENE2_PANEL_H - TEXT_PAD

    gap_text_pixels = 0
    for y in range(text_y_start, text_y_end):
        for x in range(SCENE2_GAP_X0, SCENE2_GAP_X1):
            r, g, b = rgb.getpixel((x, y))
            if _is_dark_text_pixel(r, g, b):
                gap_text_pixels += 1

    print("Scene 2 gap verification (x=580..620, y>=180 panel text zone):")
    print(f"  dark text pixels in gap: {gap_text_pixels} (expect 0)")
    print("  panel dimensions: 540x280 at left=(40,180) right=(620,180)")

    if gap_text_pixels == 0:
        print("PASS — no text overlap in center gap")
        return True

    print("FAIL — text detected in center gap")
    return False


def main() -> int:
    frame = frame_text_summarize(len(TEXT_SOURCE), summary_alpha=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.save(OUT)
    print(f"Wrote {OUT}")
    return 0 if verify_cli_frame(frame) else 1


if __name__ == "__main__":
    raise SystemExit(main())
