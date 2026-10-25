"""ffmpeg round-trip: a one-second shake and a one-second tone inside five seconds."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from mmrag.metrics import temporal_iou
from mmrag.signals import acoustic_series, events_from_series, motion_series


def test_synthesised_clip_localizes_motion_and_tone(tmp_path: Path):
    fps = 8
    size = 64
    duration = 5
    sr = 16000
    shake = (2.0, 3.0)

    audio = np.zeros(sr * duration, dtype=np.float32)
    tone = np.arange(sr, dtype=np.float32) / sr
    audio[2 * sr:3 * sr] = 0.9 * np.sin(2 * np.pi * 880 * tone)
    wav_path = tmp_path / "tone.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((audio * 32767).astype(np.int16).tobytes())

    rng = np.random.default_rng(0)
    still = np.full((size, size), 40, dtype=np.uint8)
    frames = []
    for index in range(fps * duration):
        second = index / fps
        if shake[0] <= second < shake[1]:
            frames.append(rng.integers(0, 255, (size, size), dtype=np.uint8))
        else:
            frames.append(still)
    raw_path = tmp_path / "frames.raw"
    raw_path.write_bytes(b"".join(frame.tobytes() for frame in frames))
    video_path = tmp_path / "shake.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{size}x{size}", "-r", str(fps), "-i", str(raw_path),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(video_path)],
        check=True,
    )

    times, energy, bin_s = acoustic_series(wav_path, sample_rate=sr, hop_s=0.25)
    tones = events_from_series(
        times, energy, source="tone", bin_s=bin_s,
        high_type="acoustic_anomaly", high_channel="acoustic", high_origin="acoustic",
        modality="audio",
    )
    assert tones, "expected a loudness spike"
    best_tone = max(temporal_iou((e.start, e.end), shake) for e in tones)
    assert best_tone >= 0.5

    times, energy, bin_s = motion_series(video_path, fps=fps)
    shakes = events_from_series(
        times, energy, source="shake", bin_s=bin_s,
        high_type="abnormal_vibration", high_channel="visual", high_origin="motion",
        low_type="normal_operation",
    )
    vibrations = [e for e in shakes if e.type_id == "abnormal_vibration"]
    assert vibrations, "expected a motion spike"
    best_motion = max(temporal_iou((e.start, e.end), shake) for e in vibrations)
    assert best_motion >= 0.5
