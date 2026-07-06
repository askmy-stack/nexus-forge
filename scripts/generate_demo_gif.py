#!/usr/bin/env python3
"""Generate docs/assets/demo.gif for GitHub README display."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 525
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo.gif"
FRAME_MS = 900
HOLD_FRAMES = 3

# Layout tokens
MARGIN = 48
PANEL_GAP = 72
PANEL_RADIUS = 14
LABEL_OFFSET_Y = 22
TEXT_PAD = 22
TEXT_TOP = 58
LINE_SPACING = 7


def _font(
    size: int,
    bold: bool = False,
    mono: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if mono:
        candidates = (
            "/System/Library/Fonts/Supplemental/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        )
    else:
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
        candidates = (arial, dejavu)
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _line_height(font: ImageFont.ImageFont) -> int:
    if hasattr(font, "size"):
        return int(font.size * 1.35)
    return 20


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word]) if current else word
        if _text_width(draw, trial, font) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _truncate_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    if max_lines <= 0:
        return []
    if len(lines) <= max_lines:
        return lines
    clipped = lines[:max_lines]
    last = clipped[-1]
    ellipsis = "..."
    while last and _text_width(draw, last + ellipsis, font) > max_width:
        last = last[:-1]
    clipped[-1] = (last.rstrip() + ellipsis) if last else ellipsis
    return clipped


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    font: ImageFont.ImageFont,
    color: str,
) -> None:
    content_x = x + TEXT_PAD
    content_y = y + TEXT_TOP
    content_w = w - 2 * TEXT_PAD
    content_h = h - TEXT_TOP - TEXT_PAD
    line_h = _line_height(font)
    max_lines = max(1, content_h // line_h)
    lines = _truncate_lines(
        draw,
        _wrap_text(draw, text, font, content_w),
        font,
        content_w,
        max_lines,
    )
    for i, line in enumerate(lines):
        draw.text((content_x, content_y + i * line_h), line, fill=color, font=font)


def _gradient_bg() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "#0b1220")
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(11 + t * (22 - 11))
        g = int(18 + t * (45 - 18))
        b = int(32 + t * (78 - 32))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    return img


def _header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((MARGIN, 44), title, fill="#f1f5f9", font=_font(32, bold=True))
    draw.text((MARGIN, 88), subtitle, fill="#94a3b8", font=_font(17))


def _panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    label_color: str,
    fill: str = "#0f172a",
    outline: str = "#334155",
) -> None:
    draw.rounded_rectangle(
        (x, y, x + w, y + h),
        radius=PANEL_RADIUS,
        fill=fill,
        outline=outline,
        width=2,
    )
    label_font = _font(14, bold=True)
    draw.text((x + TEXT_PAD, y + LABEL_OFFSET_Y), label, fill=label_color, font=label_font)


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
    tw = _text_width(draw, text, _font(12))
    draw.text((x + (w - tw) / 2, y + 8), text, fill=color, font=_font(12))


def _arrow(draw: ImageDraw.ImageDraw, x1: int, x2: int, y: int, color: str = "#34d399") -> None:
    draw.line([(x1, y), (x2, y)], fill=color, width=4)
    draw.polygon([(x2, y), (x2 - 14, y - 9), (x2 - 14, y + 9)], fill=color)


def _terminal_bar(draw: ImageDraw.ImageDraw, y: int, command: str) -> None:
    bar_h = 42
    draw.rounded_rectangle(
        (MARGIN, y, WIDTH - MARGIN, y + bar_h),
        radius=10,
        fill="#020617",
        outline="#1e293b",
        width=1,
    )
    draw.ellipse((MARGIN + 14, y + 14, MARGIN + 22, y + 22), fill="#ef4444")
    draw.ellipse((MARGIN + 28, y + 14, MARGIN + 36, y + 22), fill="#f59e0b")
    draw.ellipse((MARGIN + 42, y + 14, MARGIN + 50, y + 22), fill="#22c55e")
    mono = _font(13, mono=True)
    draw.text((MARGIN + 64, y + 12), command, fill="#94a3b8", font=mono)


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
    card_w, card_h, gap = 248, 168, 22
    total_w = len(modalities) * card_w + (len(modalities) - 1) * gap
    start_x = (WIDTH - total_w) // 2
    card_y = 162

    body_font = _font(14)
    for i, (name, color, desc) in enumerate(modalities):
        x = start_x + i * (card_w + gap)
        alpha = min(1.0, max(0.0, progress * 4 - i))
        if alpha <= 0:
            continue
        _panel(draw, x, card_y, card_w, card_h, name.upper(), color)
        lines = desc.split(" → ")
        for j, line in enumerate(lines):
            draw.text((x + TEXT_PAD, card_y + 58 + j * 26), line, fill="#cbd5e1", font=body_font)
        if i < len(modalities) - 1 and alpha >= 1:
            _arrow(draw, x + card_w + 4, x + card_w + gap - 4, card_y + card_h // 2, "#64748b")

    _badge(draw, WIDTH // 2 - 88, 392, 176, "6 MCP tools", "#0f766e", "#14b8a6", "#ecfdf5")
    return img


def frame_text_summarize(typed_chars: int) -> Image.Image:
    img = _gradient_bg()
    draw = ImageDraw.Draw(img)
    _header(
        draw,
        "CLI & API — Text Summarization",
        "Extractive baseline · map-reduce · refine strategies",
    )

    panel_y, panel_h = 142, 248
    panel_w = (WIDTH - 2 * MARGIN - PANEL_GAP) // 2
    input_x = MARGIN
    output_x = input_x + panel_w + PANEL_GAP
    body_font = _font(15)

    _panel(draw, input_x, panel_y, panel_w, panel_h, "INPUT", "#38bdf8")
    _panel(draw, output_x, panel_y, panel_w, panel_h, "SUMMARY", "#34d399")

    source = (
        "AI is reshaping healthcare, finance, and transportation. "
        "Machine learning detects disease from medical scans."
    )
    summary = (
        "AI reshapes healthcare, finance, and transport. "
        "ML aids disease detection from imaging."
    )
    shown = source[:typed_chars]
    _draw_text_in_box(draw, input_x, panel_y, panel_w, panel_h, shown, body_font, "#cbd5e1")

    if typed_chars >= len(source):
        _draw_text_in_box(draw, output_x, panel_y, panel_w, panel_h, summary, body_font, "#f8fafc")
        arrow_y = panel_y + panel_h // 2
        _arrow(draw, input_x + panel_w + 10, output_x - 10, arrow_y)

    _terminal_bar(
        draw,
        410,
        '$ text-summarizer summarize "..." model=extractive strategy=map_reduce',
    )
    _badge(draw, MARGIN, 462, 108, "extractive", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, MARGIN + 118, 462, 118, "map_reduce", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, MARGIN + 246, 462, 76, "refine", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, WIDTH - MARGIN - 168, 462, 168, "POST /summarize", "#0f766e", "#14b8a6", "#ecfdf5")
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
    panel_y = 142
    panel_w = WIDTH - 2 * MARGIN
    _panel(draw, MARGIN, panel_y, panel_w, 268, "MCP TOOLS", "#a78bfa")

    badge_w = (panel_w - 80) // 2
    for i, tool in enumerate(tools):
        col, row = i % 2, i // 2
        x = MARGIN + 28 + col * (badge_w + 24)
        y = panel_y + 64 + row * 54
        active = i == highlight
        fill = "#0f766e" if active else "#1e293b"
        outline = "#14b8a6" if active else "#334155"
        color = "#ecfdf5" if active else "#e2e8f0"
        _badge(draw, x, y, badge_w, tool, fill, outline, color)

    _terminal_bar(draw, 430, "$ uv run python -m textSummarizer.mcp.server")
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
    box_w, gap = 310, 36
    start_x = (WIDTH - (box_w * 3 + gap * 2)) // 2
    y = 188
    body_font = _font(14)

    for i, (title, desc, color) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        active = i <= step
        _panel(draw, x, y, box_w, 136, title, color if active else "#64748b")
        _draw_text_in_box(
            draw,
            x,
            y + 36,
            box_w,
            96,
            desc,
            body_font,
            "#cbd5e1" if active else "#64748b",
        )
        if i < 2:
            arrow_color = color if i < step else "#334155"
            _arrow(draw, x + box_w + 6, x + box_w + gap - 6, y + 68, arrow_color)

    scores = ["Coherence 4.2", "Faithfulness 4.0", "Fluency 4.5", "Relevance 4.1"]
    for i, score in enumerate(scores):
        _badge(draw, MARGIN + i * 168, 382, 150, score, "#1e293b", "#475569", "#e2e8f0")
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
