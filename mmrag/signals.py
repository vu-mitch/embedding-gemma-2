"""Short-event localization from motion and loudness.

Transcript windows and keyframe captions are tens of seconds wide. A vibration
or a squeal is often two seconds inside a long take. These detectors read the
recording's own quiet baseline and return only the bins that rise above it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .events import Event, fmt_clock


def abnormal_mask(energy: np.ndarray, z: float = 3.0, rise: float = 0.35) -> tuple[np.ndarray, float]:
    """Mark bins that sit well above the quiet baseline of this recording.

    The threshold uses the spread of the quieter half, and also a fraction of
    the gap up to the peak. A long elevated stretch (half the clip) and a short
    spike in an otherwise flat take both qualify; a flat recording does not.
    """
    energy = np.asarray(energy, dtype=np.float64)
    if energy.size == 0:
        return np.zeros(0, dtype=bool), 0.0
    baseline = float(np.quantile(energy, 0.2))
    quiet = energy[energy <= np.quantile(energy, 0.5)]
    scale = float(np.std(quiet)) if quiet.size else 0.0
    peak = float(np.max(energy))
    gap = peak - baseline
    if gap <= 1e-8:
        return np.zeros(energy.shape, dtype=bool), baseline
    thresh = baseline + max(z * scale, rise * gap)
    # A threshold glued to the peak would flicker on noise. Keep it strictly
    # below the peak so only the elevated bins pass.
    if thresh >= peak:
        thresh = baseline + 0.5 * gap
    return energy > thresh, float(thresh)


def spans_from_mask(
    times: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray,
    bin_s: float,
    min_duration: float,
) -> list[tuple[float, float, float]]:
    """Merge consecutive True bins into ``(start, end, peak)`` spans."""
    times = np.asarray(times, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    spans: list[tuple[float, float, float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        start = float(times[i])
        end = float(times[j - 1]) + bin_s
        if end - start >= min_duration:
            spans.append((start, end, float(np.max(energy[i:j]))))
        i = j
    return spans


def events_from_series(
    times: np.ndarray,
    energy: np.ndarray,
    *,
    source: str,
    bin_s: float,
    high_type: str,
    high_channel: str,
    high_origin: str,
    modality: str = "video",
    low_type: str | None = None,
    low_channel: str = "visual",
    min_high: float = 0.4,
    min_low: float = 5.0,
    z: float = 3.0,
) -> list[Event]:
    """Turn an energy series into abnormal (and optional calm) events."""
    mask, thresh = abnormal_mask(energy, z=z)
    events: list[Event] = []
    for start, end, peak in spans_from_mask(times, energy, mask, bin_s, min_high):
        events.append(Event(
            type_id=high_type,
            source=source,
            channel=high_channel,
            start=start,
            end=end,
            score=min(1.0, peak / thresh) if thresh > 0 else 1.0,
            evidence=(
                f"{high_origin} peak {peak:.2f} above threshold {thresh:.2f} "
                f"over {fmt_clock(start)}–{fmt_clock(end)}"
            ),
            modality=modality,
            origin=high_origin,
        ))
    if low_type is not None and len(mask):
        for start, end, _peak in spans_from_mask(times, energy, ~mask, bin_s, min_low):
            events.append(Event(
                type_id=low_type,
                source=source,
                channel=low_channel,
                start=start,
                end=end,
                score=0.5,
                evidence=f"quiet baseline {fmt_clock(start)}–{fmt_clock(end)}",
                modality=modality,
                origin=high_origin,
            ))
    return events


def frame_energy(samples: np.ndarray, sample_rate: int, win_s: float = 0.5, hop_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """RMS energy at the start of each hop. Returns ``(times, rms)``."""
    samples = np.asarray(samples, dtype=np.float64)
    win = max(1, int(sample_rate * win_s))
    hop = max(1, int(sample_rate * hop_s))
    if samples.size < win:
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        return np.array([0.0]), np.array([rms])
    starts = range(0, samples.size - win + 1, hop)
    times = np.array([s / sample_rate for s in starts], dtype=np.float64)
    rms = np.array([
        np.sqrt(np.mean(samples[s:s + win] ** 2) + 1e-12) for s in starts
    ], dtype=np.float64)
    return times, rms


def read_pcm(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode the audio track to mono float32 in ``[-1, 1]``."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    if not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def read_motion(path: str | Path, fps: float = 2.0, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute frame difference at ``fps``, on a tiny grayscale decode.

    One ffmpeg pass. A ten-minute file at 2 fps and 32×32 is a few megabytes,
    so a short shake inside a long recording is still visible to the detector.
    """
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={fps},scale={size}:{size},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    frame_bytes = size * size
    n = len(proc.stdout) // frame_bytes
    if n < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    frames = np.frombuffer(proc.stdout[:n * frame_bytes], dtype=np.uint8).reshape(n, size, size).astype(np.float32)
    diff = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2))
    times = np.arange(1, n, dtype=np.float64) / fps
    return times, diff.astype(np.float64)


def acoustic_series(path: str | Path, sample_rate: int = 16000, hop_s: float = 0.5) -> tuple[np.ndarray, np.ndarray, float]:
    samples = read_pcm(path, sample_rate)
    times, energy = frame_energy(samples, sample_rate, win_s=hop_s, hop_s=hop_s)
    return times, energy, hop_s


def motion_series(path: str | Path, fps: float = 2.0) -> tuple[np.ndarray, np.ndarray, float]:
    times, energy = read_motion(path, fps=fps)
    return times, energy, 1.0 / fps
