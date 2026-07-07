#!/usr/bin/env python3
"""Generate docs/assets/demo-v2.gif for GitHub README display."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 675
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo-v2.gif"
CLI_FRAME_OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo-frame-cli.png"
FRAME_MS = 900
HOLD_FRAMES = 3

# Global layout
MARGIN = 48
PANEL_RADIUS = 14
LABEL_OFFSET_Y = 20
TEXT_PAD = 24
TEXT_TOP = 56
LINE_SPACING = 6
MAX_PANEL_LINES = 4

# Scene 2 — CLI text summarize (hard-separated panels)
SCENE2_PANEL_W = 540
SCENE2_PANEL_H = 280
SCENE2_PANEL_LEFT_X = 40
SCENE2_PANEL_RIGHT_X = 620
SCENE2_PANEL_Y = 180
SCENE2_GAP_X0 = 580
SCENE2_GAP_X1 = 620
SCENE2_ARROW_X = 600

TEXT_SOURCE = (
    "AI is reshaping healthcare, finance, and transportation. "
    "Machine learning detects disease from medical scans."
)
TEXT_SUMMARY = (
    "AI reshapes healthcare, finance, and transport. ML aids disease detection from imaging."
)


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


def _line_height(font: ImageFont.ImageFont) -> int:
    if hasattr(font, "size"):
        return int(font.size * 1.32)
    return 20


def _text_width(text: str, font: ImageFont.ImageFont) -> int:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _wrap_text_to_lines(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap text using textbbox measurement; never exceeds max_width per line."""
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word]) if current else word
        if _text_width(trial, font) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _truncate_lines(
    lines: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int = MAX_PANEL_LINES,
) -> list[str]:
    if max_lines <= 0:
        return []
    if len(lines) <= max_lines:
        return lines
    clipped = lines[:max_lines]
    last = clipped[-1]
    ellipsis = "..."
    while last and _text_width(last + ellipsis, font) > max_width:
        last = last[:-1]
    clipped[-1] = (last.rstrip() + ellipsis) if last else ellipsis
    return clipped


def _draw_lines_on_panel(
    panel: Image.Image,
    lines: list[str],
    font: ImageFont.ImageFont,
    color: str,
) -> None:
    """Draw wrapped lines inside the panel content box only."""
    draw = ImageDraw.Draw(panel)
    content_x = TEXT_PAD
    content_y = TEXT_TOP
    line_h = _line_height(font) + LINE_SPACING
    for i, line in enumerate(lines):
        draw.text((content_x, content_y + i * line_h), line, fill=color, font=font)


def _build_panel_shell(label: str, label_color: str) -> Image.Image:
    """Create an exact SCENE2_PANEL_W x SCENE2_PANEL_H panel with label only."""
    panel = Image.new("RGBA", (SCENE2_PANEL_W, SCENE2_PANEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        (0, 0, SCENE2_PANEL_W - 1, SCENE2_PANEL_H - 1),
        radius=PANEL_RADIUS,
        fill="#0f172a",
        outline="#334155",
        width=2,
    )
    draw.text((TEXT_PAD, LABEL_OFFSET_Y), label, fill=label_color, font=_font(14, bold=True))
    return panel


def _panel_text_lines(text: str, font: ImageFont.ImageFont) -> list[str]:
    content_w = SCENE2_PANEL_W - 2 * TEXT_PAD
    return _truncate_lines(_wrap_text_to_lines(text, font, content_w), font, content_w)


def _make_input_panel(typed_text: str, font: ImageFont.ImageFont) -> Image.Image:
    panel = _build_panel_shell("INPUT", "#38bdf8")
    lines = _panel_text_lines(typed_text, font)
    _draw_lines_on_panel(panel, lines, font, "#cbd5e1")
    return panel


def _make_summary_panel(summary_text: str, font: ImageFont.ImageFont, alpha: float) -> Image.Image:
    panel = _build_panel_shell("SUMMARY", "#34d399")
    if summary_text and alpha > 0:
        text_layer = Image.new("RGBA", (SCENE2_PANEL_W, SCENE2_PANEL_H), (0, 0, 0, 0))
        lines = _panel_text_lines(summary_text, font)
        _draw_lines_on_panel(text_layer, lines, font, "#f8fafc")
        if alpha < 1.0:
            r, g, b, a = text_layer.split()
            a = a.point(lambda p: int(p * alpha))
            text_layer = Image.merge("RGBA", (r, g, b, a))
        panel = Image.alpha_composite(panel, text_layer)
    return panel


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
    draw.text((MARGIN, 48), title, fill="#f1f5f9", font=_font(32, bold=True))
    draw.text((MARGIN, 96), subtitle, fill="#94a3b8", font=_font(17))


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
    draw.text(
        (x + TEXT_PAD, y + LABEL_OFFSET_Y),
        label,
        fill=label_color,
        font=_font(14, bold=True),
    )


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
    tw = _text_width(text, _font(12))
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
    draw.text((MARGIN + 64, y + 12), command, fill="#94a3b8", font=_font(13, mono=True))


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
    card_y = 200

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

    _badge(draw, WIDTH // 2 - 88, 520, 176, "6 MCP tools", "#0f766e", "#14b8a6", "#ecfdf5")
    return img


def frame_text_summarize(typed_chars: int, summary_alpha: float = 1.0) -> Image.Image:
    """Scene 2 — two isolated 540x280 panel images pasted at fixed positions."""
    img = _gradient_bg().convert("RGBA")
    body_font = _font(14)

    shown = TEXT_SOURCE[:typed_chars]
    input_panel = _make_input_panel(shown, body_font)
    img.paste(input_panel, (SCENE2_PANEL_LEFT_X, SCENE2_PANEL_Y), input_panel)

    complete = typed_chars >= len(TEXT_SOURCE)
    summary_panel = _make_summary_panel(
        TEXT_SUMMARY if complete else "",
        body_font,
        summary_alpha if complete else 0.0,
    )
    img.paste(summary_panel, (SCENE2_PANEL_RIGHT_X, SCENE2_PANEL_Y), summary_panel)

    draw = ImageDraw.Draw(img)
    if complete:
        arrow_y = SCENE2_PANEL_Y + SCENE2_PANEL_H // 2
        _arrow(draw, SCENE2_GAP_X0 + 6, SCENE2_GAP_X1 - 6, arrow_y)

    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    _header(
        draw,
        "CLI & API — Text Summarization",
        "Extractive baseline · map-reduce · refine strategies",
    )
    _terminal_bar(
        draw,
        520,
        '$ text-summarizer summarize "..." model=extractive strategy=map_reduce',
    )
    _badge(draw, MARGIN, 590, 108, "extractive", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, MARGIN + 118, 590, 118, "map_reduce", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, MARGIN + 246, 590, 76, "refine", "#1e293b", "#475569", "#e2e8f0")
    _badge(draw, WIDTH - MARGIN - 168, 590, 168, "POST /summarize", "#0f766e", "#14b8a6", "#ecfdf5")
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
    panel_y = 180
    panel_w = WIDTH - 2 * MARGIN
    _panel(draw, MARGIN, panel_y, panel_w, 300, "MCP TOOLS", "#a78bfa")

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

    _terminal_bar(draw, 530, "$ uv run python -m textSummarizer.mcp.server")
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
    y = 210
    body_font = _font(14)
    line_h = _line_height(body_font) + LINE_SPACING

    for i, (title, desc, color) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        active = i <= step
        _panel(draw, x, y, box_w, 136, title, color if active else "#64748b")
        content_w = box_w - 2 * TEXT_PAD
        lines = _truncate_lines(
            _wrap_text_to_lines(desc, body_font, content_w),
            body_font,
            content_w,
            3,
        )
        for j, line in enumerate(lines):
            draw.text(
                (x + TEXT_PAD, y + 56 + j * line_h),
                line,
                fill="#cbd5e1" if active else "#64748b",
                font=body_font,
            )
        if i < 2:
            arrow_color = color if i < step else "#334155"
            _arrow(draw, x + box_w + 6, x + box_w + gap - 6, y + 68, arrow_color)

    scores = ["Coherence 4.2", "Faithfulness 4.0", "Fluency 4.5", "Relevance 4.1"]
    for i, score in enumerate(scores):
        _badge(draw, MARGIN + i * 168, 430, 150, score, "#1e293b", "#475569", "#e2e8f0")
    return img


def build_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []

    for i in range(5):
        frames.append(frame_hero((i + 1) / 5))

    step = 5
    for i in range(0, len(TEXT_SOURCE) + step, step):
        frames.append(frame_text_summarize(min(i, len(TEXT_SOURCE)), summary_alpha=0.0))

    fade_steps = 4
    for f in range(1, fade_steps + 1):
        frames.append(frame_text_summarize(len(TEXT_SOURCE), summary_alpha=f / fade_steps))
    for _ in range(HOLD_FRAMES):
        frames.append(frame_text_summarize(len(TEXT_SOURCE), summary_alpha=1.0))

    for i in range(6):
        frames.append(frame_mcp(i % 6))
    for _ in range(HOLD_FRAMES):
        frames.append(frame_mcp(5))

    for step_idx in range(3):
        for _ in range(2):
            frames.append(frame_grading(step_idx))
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

    cli_frame = frame_text_summarize(len(TEXT_SOURCE), summary_alpha=1.0)
    CLI_FRAME_OUT.parent.mkdir(parents=True, exist_ok=True)
    cli_frame.save(CLI_FRAME_OUT)
    print(f"Wrote {CLI_FRAME_OUT}")


if __name__ == "__main__":
    main()
