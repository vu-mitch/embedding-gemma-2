"""HTML timelines for visual, acoustic and spoken events."""

from __future__ import annotations

from html import escape

from .events import Event, fmt_clock

CHANNEL_LABEL = {"visual": "Video events", "acoustic": "Audio events", "spoken": "Spoken events"}
CHANNEL_ORDER = ("visual", "acoustic", "spoken")

# Close to the manufacturing-timeline figure: blue, orange, purple, green.
COLORS = {
    "normal_operation": "#5B7C99",
    "abnormal_vibration": "#C4843C",
    "acoustic_anomaly": "#A57BB5",
    "operator_report": "#6AA84F",
}


def render_timeline(events: list[Event], duration: float, title: str) -> str:
    duration = max(duration, 0.1)
    ticks = _ticks(duration)
    rows = []
    for channel in CHANNEL_ORDER:
        group = [e for e in events if e.channel == channel]
        if not group:
            continue
        bars = []
        for event in group:
            left = 100.0 * max(0.0, event.start) / duration
            width = 100.0 * max(0.4, event.end - event.start) / duration
            color = COLORS.get(event.type_id, "#888")
            label = escape(event.label)
            tip = escape(f"{event.label} {event.locator()} · {event.origin}")
            bars.append(
                f'<div title="{tip}" style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;'
                f'top:6px;height:28px;background:{color};color:white;border-radius:6px;'
                f'font:600 12px/28px sans-serif;text-align:center;overflow:hidden;'
                f'white-space:nowrap;padding:0 6px;box-sizing:border-box">{label}</div>'
            )
        rows.append(
            f'<div style="display:flex;gap:12px;align-items:center;margin:8px 0">'
            f'<div style="width:110px;flex:none;font:600 13px sans-serif">{escape(CHANNEL_LABEL[channel])}</div>'
            f'<div style="position:relative;flex:1;height:40px;background:#f4f1ea;border-radius:8px">{"".join(bars)}</div>'
            f"</div>"
        )
    tick_row = "".join(
        f'<span style="position:absolute;left:{100.0 * t / duration:.2f}%;transform:translateX(-50%);'
        f'font:12px sans-serif;color:#666">{fmt_clock(t)[:5]}</span>'
        for t in ticks
    )
    return (
        f'<div style="border:1px solid #e6e0d6;border-radius:12px;padding:12px 14px;margin:8px 0">'
        f'<div style="font:700 15px sans-serif;margin-bottom:6px">{escape(title)}</div>'
        f'{"".join(rows)}'
        f'<div style="position:relative;height:18px;margin-left:122px">{tick_row}</div>'
        f"</div>"
    )


def _ticks(duration: float) -> list[float]:
    if duration <= 60:
        step = 10
    elif duration <= 180:
        step = 30
    else:
        step = 60
    ticks = [float(t) for t in range(0, int(duration) + 1, step)]
    if not ticks or ticks[-1] != duration:
        ticks.append(float(duration))
    return ticks
