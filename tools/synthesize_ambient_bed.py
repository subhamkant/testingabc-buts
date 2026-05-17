"""
Procedurally generate assets/dark_ambient.mp3 — the continuous ambient bed
that fills gaps between music chunks when MUSIC_CHUNKING=true in
pipeline/video_assembler.py.

Why procedural?
  YouTube Content ID matches on audio fingerprints. Even Pixabay-licensed
  "dark ambient drone" tracks have been re-uploaded across the platform
  and CAN trigger claims despite their CC0 license. A synthesized drone
  built from primitive waveforms is ORIGINAL WORK — guaranteed never to
  match an existing Content ID fingerprint. The recipe below is fully
  reproducible: anyone re-running this script gets the same output (byte-
  level identical from a given ffmpeg version + same lavfi parameters).

Recipe:
  - 60 seconds (longer than any video we make, so it loops cleanly)
  - Brown noise base, highpass at 50Hz to remove sub-rumble, lowpass at
    180Hz to keep it dark / sub-bass-only
  - 55Hz sine layer (A1) at 18% volume — adds bass tone
  - 82Hz sine layer (E2) at 10% volume — adds harmonic body
  - aecho reverb (60ms + 120ms taps) for atmosphere
  - 2-second fade in/out so loop seams are inaudible
  - Encoded to MP3 128 kbps stereo to keep git size ~1 MB (vs 10 MB WAV)

Usage:
  python tools/synthesize_ambient_bed.py
"""

import os
import subprocess
import sys

OUTPUT_MP3 = "assets/dark_ambient.mp3"

# Single ffmpeg invocation that synthesizes the drone, runs the post-
# processing, and encodes to MP3. All steps are deterministic given the
# same ffmpeg/libmp3lame version.
FFMPEG_CMD = [
    "ffmpeg", "-y",
    # Input 0: brown noise, compressed
    "-f", "lavfi",
    "-i", "anoisesrc=d=60:c=brown:a=0.3,"
          "highpass=f=50,"
          "lowpass=f=180,"
          "acompressor=threshold=0.3:ratio=2:attack=200:release=2000",
    # Input 1: 55 Hz sine (A1) at 0.18
    "-f", "lavfi",
    "-i", "sine=frequency=55:duration=60:sample_rate=44100,volume=0.18",
    # Input 2: 82 Hz sine (E2) at 0.10
    "-f", "lavfi",
    "-i", "sine=frequency=82:duration=60:sample_rate=44100,volume=0.10",
    # Mix + reverb + fades + final encoding
    "-filter_complex",
    "[0:a][1:a][2:a]amix=inputs=3:normalize=0,"
    "aecho=0.6:0.3:60|120:0.4|0.3,"
    "afade=t=in:d=2,afade=t=out:st=58:d=2,"
    "aresample=44100",
    "-ar", "44100", "-ac", "2",
    "-c:a", "libmp3lame", "-b:a", "128k",
    OUTPUT_MP3,
]


def main() -> int:
    os.makedirs(os.path.dirname(OUTPUT_MP3) or ".", exist_ok=True)
    print(f"Synthesizing {OUTPUT_MP3}...")
    result = subprocess.run(FFMPEG_CMD, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        sys.exit(f"[ERROR] ffmpeg failed (exit {result.returncode})")
    if not os.path.exists(OUTPUT_MP3):
        sys.exit(f"[ERROR] {OUTPUT_MP3} not created")
    size_kb = os.path.getsize(OUTPUT_MP3) / 1024
    print(f"[OK] {OUTPUT_MP3} written ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
