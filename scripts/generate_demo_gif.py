#!/usr/bin/env python3
"""Generate docs/assets/demo.gif for GitHub README display."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 525
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo.gif"
FRAME_MS = 900
HOLD_FRAMES = 3


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    arial = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf"
    )
    dejavu = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    candidates = (
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        arial,
        dejavu,
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _gradient_bg() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "#0f172a")
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(15 + t * (30 - 15))
        g = int(23 + t * (58 - 23))
        b = int(42 + t * (95 - 42))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    return img


def _header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((60, 52), title, fill="#e2e8f0", font=_font(34, bold=True))
    draw.text((60, 96), subtitle, fill="#94a3b8", font=_font(18))


def _panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    label_color: str,
) -> None:
    draw.rounded_rectangle(
        (x, y, x + w, y + h),
        radius=16,
        fill="#111827",
        outline="#334155",
        width=2,
    )
    draw.text((x + 20, y + 24), label, fill=label_color, font=_font(15, bold=True))


def _badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    text: str,
    fill: str,
    outline: str,
    color: str,
) -> None:
    draw.rounded_rectangle((x, y, x + w, y + 30), radius=15, fill=fill, outline=outline, width=1)
    bbox = draw.textbbox((0, 0), text, font=_font(13))
    tw = bbox[2] - bbox[0]
    draw.text((x + (w - tw) / 2, y + 7), text, fill=color, font=_font(13))


def _arrow(draw: ImageDraw.ImageDraw, x1: int, x2: int, y: int, color: str = "#34d399") -> None:
    draw.line([(x1, y), (x2, y)], fill=color, width=5)
    draw.polygon([(x2, y), (x2 - 16, y - 10), (x2 - 16, y + 10)], fill=color)


def frame_hero(progress: float) -> Image.Image:
    img = _gradient_bg()
    draw = ImageDraw.Draw(img)
    _header(
        draw,
        "Nexus Forge — SummarizeHub",
        "Multimodal summarization · MCP · grading loop · v1.0.0",
    )

    modalities = [
        ("Text", "#38bdf8", "T5 / BART / Pegasus"),
        ("Image", "#a78bfa", "BLIP caption → summarize"),
        ("Audio", "#f472b6", "Whisper ASR → summarize"),
        ("Video", "#fb923c", "ffmpeg + ASR + keyframes"),
    ]
    card_w, card_h, gap = 250, 170, 24
    total_w = len(modalities) * card_w + (len(modalities) - 1) * gap
    start_x = (WIDTH - total_w) // 2
    card_y = 170

    for i, (name, color, desc) in enumerate(modalities):
        x = start_x + i * (card_w + gap)
        alpha = min(1.0, max(0.0, progress * 4 - i))
        if alpha <= 0:
            continue
        _panel(draw, x, card_y, card_w, card_h, name.upper(), color)
        lines = desc.split(" → ")
        for j, line in enumerate(lines):
            draw.text((x + 20, card_y + 64 + j * 28), line, fill="#cbd5e1", font=_font(15))
        if i < len(modalities) - 1 and alpha >= 1:
            _arrow(draw, x + card_w + 6, x + card_w + gap - 6, card_y + card_h // 2, "#64748b")

    _badge(draw, WIDTH // 2 - 90, 400, 180, "6 MCP tools", "#0f766e", "#14b8a6", "#ecfdf5")
    return img


def frame_text_summarize(typed_chars: int) -> Image.Image:
    img = _gradient_bg()
    draw = ImageDraw.Draw(img)
    _header(
        draw,
        "CLI & API — Text Summarization",
        "Extractive baseline · map-reduce · refine strategies",
    )

    panel_y, panel_h = 150, 250
    input_x, input_w = 60, 500
    output_x, output_w = 640, 500
    _panel(draw, input_x, panel_y, input_w, panel_h, "INPUT", "#38bdf8")
    _panel(draw, output_x, panel_y, output_w, panel_h, "SUMMARY", "#34d399")

    source = (
        "AI is reshaping healthcare, finance, and transportation. "
        "Machine learning detects disease from medical scans."
    )
    summary = (
        "AI reshapes healthcare, finance, and transport. "
        "ML aids disease detection from imaging."
    )
    shown = source[:typed_chars]
    draw.text((input_x + 20, panel_y + 64), shown, fill="#cbd5e1", font=_font(16))
    if typed_chars >= len(source):
        draw.text((output_x + 20, panel_y + 64), summary, fill="#f8fafc", font=_font(16))
        _arrow(draw, input_x + input_w + 12, output_x - 12, panel_y + panel_h // 2)

    draw.text(
        (60, 430),
        '$ text-summarizer --text "..." --model extractive',
        fill="#64748b",
        font=_font(14),
    )
    _badge(draw, 60, 460, 110, "extractive", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, 180, 460, 120, "map_reduce", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, 310, 460, 80, "refine", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, 900, 460, 150, "POST /summarize", "#0f766e", "#14b8a6", "#ecfdf5")
    return img


def frame_mcp(highlight: int) -> Image.Image:
    img = _gradient_bg()
    draw = ImageDraw.Draw(img)
    _header(
        draw,
        "MCP Server — AI Agent Integration",
        "stdio server · Cursor skill · 6 tools for multimodal workflows",
    )

    tools = [
        "summarize_text",
        "summarize_image",
        "summarize_audio",
        "summarize_video",
        "list_models",
        "grade_summary",
    ]
    panel_y = 150
    _panel(draw, 60, panel_y, 1080, 280, "MCP TOOLS", "#a78bfa")

    for i, tool in enumerate(tools):
        col, row = i % 2, i // 2
        x = 90 + col * 520
        y = panel_y + 70 + row * 52
        active = i == highlight
        fill = "#0f766e" if active else "#1e293b"
        outline = "#14b8a6" if active else "#334155"
        color = "#ecfdf5" if active else "#e2e8f0"
        _badge(draw, x, y, 480, tool, fill, outline, color)

    draw.text(
        (60, 455),
        "$ uv run python -m textSummarizer.mcp.server",
        fill="#64748b",
        font=_font(14),
    )
    return img


def frame_grading(step: int) -> Image.Image:
    img = _gradient_bg()
    draw = ImageDraw.Draw(img)
    _header(
        draw,
        "Grading Loop — Summarize → Grade → Refine",
        "Subjective rubric: coherence · faithfulness · fluency · relevance",
    )

    steps = [
        ("1. Summarize", "Generate candidate summary", "#38bdf8"),
        ("2. Grade", "Score 1–5 on rubric dimensions", "#a78bfa"),
        ("3. Refine", "Iterate if score < threshold", "#34d399"),
    ]
    box_w, gap = 320, 40
    start_x = (WIDTH - (box_w * 3 + gap * 2)) // 2
    y = 200

    for i, (title, desc, color) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        active = i <= step
        _panel(draw, x, y, box_w, 140, title, color if active else "#64748b")
        draw.text((x + 20, y + 64), desc, fill="#cbd5e1" if active else "#64748b", font=_font(15))
        if i < 2:
            arrow_color = color if i < step else "#334155"
            _arrow(draw, x + box_w + 8, x + box_w + gap - 8, y + 70, arrow_color)

    scores = ["Coherence 4.2", "Faithfulness 4.0", "Fluency 4.5", "Relevance 4.1"]
    for i, score in enumerate(scores):
        _badge(draw, 60 + i * 170, 390, 150, score, "#1e293b", "#475569", "#e2e8f0")
    return img


def build_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []

    for i in range(5):
        frames.append(frame_hero((i + 1) / 5))

    source = (
        "AI is reshaping healthcare, finance, and transportation. "
        "Machine learning detects disease from medical scans."
    )
    for i in range(0, len(source) + 20, 12):
        frames.append(frame_text_summarize(min(i, len(source))))
    for _ in range(HOLD_FRAMES):
        frames.append(frame_text_summarize(len(source)))

    for i in range(6):
        frames.append(frame_mcp(i % 6))
    for _ in range(HOLD_FRAMES):
        frames.append(frame_mcp(5))

    for step in range(3):
        for _ in range(2):
            frames.append(frame_grading(step))
    for _ in range(HOLD_FRAMES):
        frames.append(frame_grading(2))

    return frames


def main() -> None:
    frames = build_frames()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT} ({len(frames)} frames, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
