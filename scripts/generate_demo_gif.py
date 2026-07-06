#!/usr/bin/env python3
"""Generate docs/assets/demo.gif for GitHub README display."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 525
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo.gif"
CLI_FRAME_OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo-frame-cli.png"
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
MAX_PANEL_LINES = 4

# CLI scene — hard panel boundaries (no text may cross these)
CLI_INPUT_X = 40
CLI_INPUT_W = 540  # x 40..580
CLI_GAP_START = 580
CLI_GAP_END = 620
CLI_SUMMARY_X = 620
CLI_SUMMARY_W = 540  # x 620..1160


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


def _render_clipped_panel_text(
    panel_w: int,
    panel_h: int,
    text: str,
    font: ImageFont.ImageFont,
    color: str,
    max_lines: int = MAX_PANEL_LINES,
) -> Image.Image:
    """Render wrapped text into an isolated layer clipped to the panel content area."""
    content_w = panel_w - 2 * TEXT_PAD
    content_h = panel_h - TEXT_TOP - TEXT_PAD
    line_h = _line_height(font)

    layer = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    probe = ImageDraw.Draw(layer)
    lines = _truncate_lines(
        probe,
        _wrap_text(probe, text, font, content_w),
        font,
        content_w,
        max_lines,
    )

    content = Image.new("RGBA", (content_w, content_h), (0, 0, 0, 0))
    content_draw = ImageDraw.Draw(content)
    for i, line in enumerate(lines):
        y = i * line_h
        if y + line_h > content_h:
            break
        content_draw.text((0, y), line, fill=color, font=font)

    layer.paste(content, (TEXT_PAD, TEXT_TOP), content)
    return layer


def _make_panel_image(
    w: int,
    h: int,
    label: str,
    label_color: str,
    text: str,
    text_font: ImageFont.ImageFont,
    text_color: str,
    fill: str = "#0f172a",
    outline: str = "#334155",
    max_lines: int = MAX_PANEL_LINES,
) -> Image.Image:
    """Build a self-contained panel with shadow, label, and clipped text."""
    shadow_offset = 4
    canvas = Image.new("RGBA", (w + shadow_offset, h + shadow_offset), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (0, 0, w - 1, h - 1),
        radius=PANEL_RADIUS,
        fill=(0, 0, 0, 72),
    )
    canvas.paste(shadow, (shadow_offset, shadow_offset), shadow)

    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle(
        (0, 0, w - 1, h - 1),
        radius=PANEL_RADIUS,
        fill=fill,
        outline=outline,
        width=2,
    )
    label_font = _font(14, bold=True)
    panel_draw.text((TEXT_PAD, LABEL_OFFSET_Y), label, fill=label_color, font=label_font)

    if text:
        text_layer = _render_clipped_panel_text(
            w, h, text, text_font, text_color, max_lines=max_lines
        )
        panel = Image.alpha_composite(panel, text_layer)

    canvas.paste(panel, (0, 0), panel)
    return canvas


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


TEXT_SOURCE = (
    "AI is reshaping healthcare, finance, and transportation. "
    "Machine learning detects disease from medical scans."
)
TEXT_SUMMARY = (
    "AI reshapes healthcare, finance, and transport. ML aids disease detection from imaging."
)


def frame_text_summarize(typed_chars: int) -> Image.Image:
    img = _gradient_bg().convert("RGBA")
    draw = ImageDraw.Draw(img)
    _header(
        draw,
        "CLI & API — Text Summarization",
        "Extractive baseline · map-reduce · refine strategies",
    )

    panel_y, panel_h = 142, 248
    body_font = _font(15)

    # INPUT panel — isolated layer, text clipped to left region only
    shown = TEXT_SOURCE[:typed_chars]
    input_panel = _make_panel_image(
        CLI_INPUT_W,
        panel_h,
        "INPUT",
        "#38bdf8",
        shown,
        body_font,
        "#cbd5e1",
    )
    img.paste(input_panel, (CLI_INPUT_X, panel_y), input_panel)

    # SUMMARY panel — only rendered after typing completes
    complete = typed_chars >= len(TEXT_SOURCE)
    if complete:
        summary_panel = _make_panel_image(
            CLI_SUMMARY_W,
            panel_h,
            "SUMMARY",
            "#34d399",
            TEXT_SUMMARY,
            body_font,
            "#f8fafc",
        )
        img.paste(summary_panel, (CLI_SUMMARY_X, panel_y), summary_panel)

        arrow_y = panel_y + panel_h // 2
        _arrow(draw, CLI_GAP_START + 4, CLI_GAP_END - 4, arrow_y)
    else:
        # Empty right panel shell (no summary text during typing)
        empty_summary = _make_panel_image(
            CLI_SUMMARY_W,
            panel_h,
            "SUMMARY",
            "#34d399",
            "",
            body_font,
            "#f8fafc",
        )
        img.paste(empty_summary, (CLI_SUMMARY_X, panel_y), empty_summary)

    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
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

    # Smoother typing: smaller steps, stays inside INPUT panel via clip mask
    step = 6
    for i in range(0, len(TEXT_SOURCE) + step, step):
        frames.append(frame_text_summarize(min(i, len(TEXT_SOURCE))))
    for _ in range(HOLD_FRAMES):
        frames.append(frame_text_summarize(len(TEXT_SOURCE)))

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


def _is_body_text_pixel(r: int, g: int, b: int) -> bool:
    """Detect INPUT/SUMMARY body text; ignore green arrow in center gap."""
    if g > 170 and g > r + 60 and g > b + 30:
        return False
    return r > 140 and g > 140 and b > 140


def verify_cli_frame(frame: Image.Image) -> None:
    """Assert INPUT text stays left of gap and SUMMARY text stays right of gap."""
    rgb = frame.convert("RGB")
    panel_y, panel_h = 142, 248
    text_y_start = panel_y + TEXT_TOP
    text_y_end = panel_y + panel_h - TEXT_PAD

    gap_text_pixels = 0
    input_in_gap = 0
    summary_in_gap = 0
    input_max_x = 0
    summary_min_x = WIDTH

    for y in range(text_y_start, text_y_end):
        for x in range(CLI_GAP_START, CLI_GAP_END):
            r, g, b = rgb.getpixel((x, y))
            if _is_body_text_pixel(r, g, b):
                gap_text_pixels += 1

        for x in range(CLI_INPUT_X, CLI_GAP_START):
            r, g, b = rgb.getpixel((x, y))
            if _is_body_text_pixel(r, g, b):
                input_max_x = max(input_max_x, x)
                if x >= CLI_GAP_START:
                    input_in_gap += 1

        for x in range(CLI_SUMMARY_X, CLI_SUMMARY_X + CLI_SUMMARY_W):
            r, g, b = rgb.getpixel((x, y))
            if _is_body_text_pixel(r, g, b):
                summary_min_x = min(summary_min_x, x)
                if x < CLI_SUMMARY_X:
                    summary_in_gap += 1

    print("CLI frame verification:")
    print(f"  gap text pixels (580-620): {gap_text_pixels} (expect 0)")
    print(f"  INPUT max text x: {input_max_x} (expect < 600)")
    print(f"  SUMMARY min text x: {summary_min_x} (expect > 620)")
    print(f"  INPUT bleed into gap: {input_in_gap}")
    print(f"  SUMMARY bleed into gap: {summary_in_gap}")

    assert gap_text_pixels == 0, f"Text found in center gap: {gap_text_pixels} pixels"
    assert input_max_x < 600, f"INPUT text extends too far right: x={input_max_x}"
    assert summary_min_x > 620, f"SUMMARY text starts too far left: x={summary_min_x}"
    print("  PASS — no overlap detected")


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

    cli_frame = frame_text_summarize(len(TEXT_SOURCE))
    CLI_FRAME_OUT.parent.mkdir(parents=True, exist_ok=True)
    cli_frame.save(CLI_FRAME_OUT)
    print(f"Wrote {CLI_FRAME_OUT}")
    verify_cli_frame(cli_frame)


if __name__ == "__main__":
    main()
