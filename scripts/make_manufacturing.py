"""Write a 50 second manufacturing clip laid out like the research figure.

0–10s calm picture, 10–30s shaking picture, 15–30s a loud tone, then silence.
The operator report at 40s is spoken text in the benchmark; this file only
carries the visual and acoustic events Whisper is not required to build.

    python scripts/make_manufacturing.py
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "samples" / "manufacturing_line_a.mp4"


def main() -> None:
    fps, size, duration, sr = 8, 64, 50, 16000
    rng = np.random.default_rng(1)
    still = np.full((size, size, 3), (70, 110, 150), dtype=np.uint8)
    frames = []
    for index in range(fps * duration):
        second = index / fps
        if 10 <= second < 30:
            noise = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            frames.append(noise)
        else:
            frames.append(still.copy())
    raw = OUT.with_suffix(".raw")
    wav = OUT.with_suffix(".wav")
    OUT.parent.mkdir(exist_ok=True)
    raw.write_bytes(b"".join(frame.tobytes() for frame in frames))

    audio = np.zeros(sr * duration, dtype=np.float32)
    tone = np.arange(15 * sr, dtype=np.float32) / sr
    audio[15 * sr:30 * sr] = 0.6 * np.sin(2 * np.pi * 740 * tone)
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((audio * 32767).astype(np.int16).tobytes())

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps), "-i", str(raw),
         "-i", str(wav),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(OUT)],
        check=True,
    )
    raw.unlink()
    wav.unlink()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
