from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]


@dataclass(frozen=True)
class PanelGeometry:
    left: int
    top: int
    right: int
    bottom: int


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError("No usable Chinese font file found on this system.")


def save_canvas(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill="black", anchor: str | None = None) -> None:
    if anchor is None:
        draw.text(xy, text, font=font, fill=fill)
    else:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_multiline_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill="black") -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) // 2
    y = top + (bottom - top - text_h) // 2
    draw.text((x, y), text, font=font, fill=fill)


def panel_rect(index: int, rows: int, cols: int, *, width: int, height: int, top_margin: int = 140, bottom_margin: int = 60, left_margin: int = 60, right_margin: int = 40, h_gap: int = 40, v_gap: int = 45) -> PanelGeometry:
    plot_width = (width - left_margin - right_margin - (cols - 1) * h_gap) // cols
    plot_height = (height - top_margin - bottom_margin - (rows - 1) * v_gap) // rows
    row = index // cols
    col = index % cols
    left = left_margin + col * (plot_width + h_gap)
    top = top_margin + row * (plot_height + v_gap)
    return PanelGeometry(left, top, left + plot_width, top + plot_height)


def _draw_rotated_text(image: Image.Image, xy: tuple[int, int], text: str, font, fill="black", angle: int = 90) -> None:
    bbox = ImageDraw.Draw(image).textbbox((0, 0), text, font=font)
    temp = Image.new("RGBA", (bbox[2] - bbox[0] + 10, bbox[3] - bbox[1] + 10), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((5, 5), text, font=font, fill=fill)
    rotated = temp.rotate(angle, expand=True)
    image.alpha_composite(rotated, dest=xy)


def _scale_points(values: list[float], vmin: float, vmax: float, start: int, end: int) -> list[int]:
    if vmax <= vmin:
        mid = (start + end) // 2
        return [mid for _ in values]
    return [int(end - (value - vmin) / (vmax - vmin) * (end - start)) for value in values]


def draw_line_chart(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: PanelGeometry,
    *,
    x_values: list[float],
    series: list[dict],
    title: str,
    x_label: str,
    y_label: str,
    x_tick_labels: list[tuple[float, str]] | None = None,
    y_zero_line: bool = False,
    vertical_lines: list[tuple[float, str]] | None = None,
    shaded_ranges: list[tuple[float, float, str]] | None = None,
) -> None:
    title_font = get_font(22)
    axis_font = get_font(16)
    tick_font = get_font(13)

    left_pad = 72
    right_pad = 18
    top_pad = 36
    bottom_pad = 42
    plot_left = rect.left + left_pad
    plot_top = rect.top + top_pad
    plot_right = rect.right - right_pad
    plot_bottom = rect.bottom - bottom_pad

    draw.rounded_rectangle((rect.left, rect.top, rect.right, rect.bottom), radius=8, outline="#d1d5db", width=1, fill="white")
    draw_text(draw, (rect.left + 8, rect.top + 4), title, font=title_font, fill="black")

    all_y: list[float] = []
    for item in series:
        all_y.extend([v for v in item["y_values"] if v is not None])
    if not x_values or not all_y:
        draw_text(draw, (rect.left + 20, rect.top + 50), "无可用数据", font=axis_font, fill="#6b7280")
        return

    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = min(all_y), max(all_y)
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    y_pad = 0.08 * (ymax - ymin)
    ymin -= y_pad
    ymax += y_pad

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#9ca3af", width=1)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(plot_bottom - frac * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        value = ymin + frac * (ymax - ymin)
        draw_text(draw, (plot_left - 8, y), f"{value:.1f}", font=tick_font, fill="#374151", anchor="ra")

    if shaded_ranges:
        for x0, x1, color in shaded_ranges:
            if xmax <= xmin:
                continue
            sx0 = int(plot_left + (x0 - xmin) / (xmax - xmin) * (plot_right - plot_left))
            sx1 = int(plot_left + (x1 - xmin) / (xmax - xmin) * (plot_right - plot_left))
            draw.rectangle((sx0, plot_top, sx1, plot_bottom), fill=color)

    if vertical_lines:
        for xv, color in vertical_lines:
            if xmax <= xmin:
                continue
            sx = int(plot_left + (xv - xmin) / (xmax - xmin) * (plot_right - plot_left))
            draw.line((sx, plot_top, sx, plot_bottom), fill=color, width=1)

    if y_zero_line and ymin <= 0 <= ymax:
        sy = int(plot_bottom - (0 - ymin) / (ymax - ymin) * (plot_bottom - plot_top))
        draw.line((plot_left, sy, plot_right, sy), fill="#111827", width=2)

    for item in series:
        y_values = item["y_values"]
        color = item["color"]
        width = item.get("width", 2)
        mode = item.get("mode", "line")
        points: list[tuple[int, int]] = []
        for xv, yv in zip(x_values, y_values):
            if yv is None:
                if len(points) >= 2 and mode in {"line", "line+markers"}:
                    draw.line(points, fill=color, width=width)
                points = []
                continue
            sx = int(plot_left + (xv - xmin) / (xmax - xmin) * (plot_right - plot_left)) if xmax > xmin else plot_left
            sy = int(plot_bottom - (yv - ymin) / (ymax - ymin) * (plot_bottom - plot_top))
            points.append((sx, sy))
        if len(points) >= 2 and mode in {"line", "line+markers"}:
            draw.line(points, fill=color, width=width)
        if mode in {"markers", "line+markers"}:
            for px, py in points:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=color)

    if x_tick_labels:
        for xv, label in x_tick_labels:
            sx = int(plot_left + (xv - xmin) / (xmax - xmin) * (plot_right - plot_left)) if xmax > xmin else plot_left
            draw.line((sx, plot_bottom, sx, plot_bottom + 4), fill="#374151", width=1)
            draw_text(draw, (sx, plot_bottom + 8), label, font=tick_font, fill="#374151", anchor="ma")

    draw_text(draw, ((plot_left + plot_right) // 2, rect.bottom - 8), x_label, font=axis_font, fill="black", anchor="ms")
    _draw_rotated_text(image, (rect.left + 4, rect.top + (rect.bottom - rect.top) // 2 - 40), y_label, axis_font, fill="black", angle=90)


def draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    rect: PanelGeometry,
    *,
    categories: list[str],
    values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    image: Image.Image,
) -> None:
    title_font = get_font(22)
    axis_font = get_font(16)
    tick_font = get_font(13)

    left_pad = 72
    right_pad = 18
    top_pad = 36
    bottom_pad = 50
    plot_left = rect.left + left_pad
    plot_top = rect.top + top_pad
    plot_right = rect.right - right_pad
    plot_bottom = rect.bottom - bottom_pad

    draw.rounded_rectangle((rect.left, rect.top, rect.right, rect.bottom), radius=8, outline="#d1d5db", width=1, fill="white")
    draw_text(draw, (rect.left + 8, rect.top + 4), title, font=title_font, fill="black")
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#9ca3af", width=1)

    ymin = min(0.0, min(values))
    ymax = max(0.0, max(values))
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    y_pad = 0.08 * (ymax - ymin)
    ymin -= y_pad
    ymax += y_pad

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(plot_bottom - frac * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        value = ymin + frac * (ymax - ymin)
        draw_text(draw, (plot_left - 8, y), f"{value:.1f}", font=tick_font, fill="#374151", anchor="ra")

    zero_y = int(plot_bottom - (0 - ymin) / (ymax - ymin) * (plot_bottom - plot_top))
    draw.line((plot_left, zero_y, plot_right, zero_y), fill="#111827", width=2)

    bar_space = (plot_right - plot_left) / max(len(categories), 1)
    bar_width = int(bar_space * 0.65)
    for idx, (category, value) in enumerate(zip(categories, values)):
        center_x = int(plot_left + (idx + 0.5) * bar_space)
        top_y = int(plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top))
        x0 = center_x - bar_width // 2
        x1 = center_x + bar_width // 2
        y0, y1 = sorted([zero_y, top_y])
        draw.rectangle((x0, y0, x1, y1), fill="#2563eb", outline="#2563eb")
        draw_text(draw, (center_x, plot_bottom + 8), category, font=tick_font, fill="#374151", anchor="ma")

    draw_text(draw, ((plot_left + plot_right) // 2, rect.bottom - 10), x_label, font=axis_font, fill="black", anchor="ms")
    _draw_rotated_text(image, (rect.left + 4, rect.top + (rect.bottom - rect.top) // 2 - 40), y_label, axis_font, fill="black", angle=90)
