#!/usr/bin/env python3
"""Generate docs/assets/demo.png for GitHub README display."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 525
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo.png"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), "#0f172a")
    draw = ImageDraw.Draw(img)

    # Background gradient bands
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(15 + t * (30 - 15))
        g = int(23 + t * (58 - 23))
        b = int(42 + t * (95 - 42))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    title_font = _font(34, bold=True)
    sub_font = _font(18)
    label_font = _font(15, bold=True)
    mono_font = _font(16)
    badge_font = _font(13)
    small_font = _font(11)

    margin = 60
    draw.text((margin, 52), "SummarizeHub — Live Demo", fill="#e2e8f0", font=title_font)
    draw.text(
        (margin, 96),
        "Extractive & abstractive summarization in one API",
        fill="#94a3b8",
        font=sub_font,
    )

    panel_y, panel_h = 130, 300
    input_w, output_w = 500, 460
    gap = 80
    input_x = margin
    output_x = input_x + input_w + gap

    def panel(x: int, w: int, label: str, label_color: str) -> None:
        draw.rounded_rectangle((x, panel_y, x + w, panel_y + panel_h), radius=16, fill="#111827", outline="#334155", width=2)
        draw.text((x + 20, panel_y + 24), label, fill=label_color, font=label_font)

    panel(input_x, input_w, "INPUT", "#38bdf8")
    panel(output_x, output_w, "SUMMARY", "#34d399")

    input_lines = [
        "AI is reshaping healthcare,",
        "finance, and transportation.",
        "Machine learning detects",
        "disease from medical scans...",
    ]
    summary_lines = [
        "AI reshapes healthcare,",
        "finance, and transport.",
        "ML aids disease detection",
        "from medical imaging.",
    ]

    text_y = panel_y + 64
    for i, line in enumerate(input_lines):
        draw.text((input_x + 20, text_y + i * 30), line, fill="#cbd5e1", font=mono_font)
    for i, line in enumerate(summary_lines):
        draw.text((output_x + 20, text_y + i * 30), line, fill="#f8fafc", font=mono_font)

    # Arrow between panels
    arrow_y = panel_y + panel_h // 2
    arrow_x1 = input_x + input_w + 12
    arrow_x2 = output_x - 12
    draw.line([(arrow_x1, arrow_y), (arrow_x2, arrow_y)], fill="#34d399", width=5)
    draw.polygon(
        [(arrow_x2, arrow_y), (arrow_x2 - 16, arrow_y - 10), (arrow_x2 - 16, arrow_y + 10)],
        fill="#34d399",
    )
    draw.ellipse((arrow_x1 + 40, arrow_y - 6, arrow_x1 + 52, arrow_y + 6), fill="#38bdf8")

    badge_y = panel_y + panel_h + 28
    badges = [
        (margin, 90, "extractive", "#1e293b", "#475569", "#e2e8f0"),
        (margin + 100, 110, "map_reduce", "#1e293b", "#475569", "#e2e8f0"),
        (margin + 220, 70, "refine", "#1e293b", "#475569", "#e2e8f0"),
        (output_x, 150, "POST /summarize", "#0f766e", "#14b8a6", "#ecfdf5"),
    ]
    for x, w, text, fill, outline, color in badges:
        draw.rounded_rectangle((x, badge_y, x + w, badge_y + 30), radius=15, fill=fill, outline=outline, width=1)
        bbox = draw.textbbox((0, 0), text, font=badge_font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) / 2, badge_y + 7), text, fill=color, font=badge_font)

    draw.text(
        (WIDTH - margin, HEIGHT - 24),
        "Record a GIF: bash docs/assets/record-demo.sh",
        fill="#64748b",
        font=small_font,
        anchor="ra",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
