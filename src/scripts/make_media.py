#!/usr/bin/env python3
"""
Make GIF and/or MP4 animations from a folder of rendered PNG frames.

Examples:
  python scripts/make_media.py --input images/images_fourier --gif outputs/fourier.gif
  python scripts/make_media.py --input images/images_fourier --mp4 outputs/fourier.mp4 --fps 5 --size 1920 1080
  python scripts/make_media.py --input images/images_no_fourier --gif outputs/no_fourier.gif --duration-ms 200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# MP4 export depends on imageio + ffmpeg
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


def list_frames(input_dir: Path, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    """List image frames in sorted order from a directory.

    Args:
        input_dir: Directory containing frame images.
        exts: Allowed file extensions.

    Returns:
        Sorted list of paths to frame files.

    Raises:
        FileNotFoundError: If input_dir does not exist.
        ValueError: If no frames are found.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not files:
        raise ValueError(f"No frames found in {input_dir} with extensions {exts}")
    return files


def load_frames_rgb(paths: List[Path], size: Tuple[int, int] | None = None) -> List[Image.Image]:
    """Load frames as RGB PIL images, optionally resizing.

    Args:
        paths: List of frame paths in display order.
        size: Optional (width, height) to resize frames to.

    Returns:
        List of RGB PIL Images.
    """
    frames: List[Image.Image] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)
        frames.append(img)
    return frames


def save_gif(frames: List[Image.Image], out_path: Path, duration_ms: int = 200, loop: int = 0) -> None:
    """Save frames to an animated GIF.

    Args:
        frames: List of RGB PIL images.
        out_path: Output .gif path.
        duration_ms: Duration per frame in milliseconds.
        loop: Number of loops (0 = infinite).

    Raises:
        ValueError: If frames is empty.
    """
    if not frames:
        raise ValueError("No frames provided for GIF export")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )


def save_mp4(
    frame_paths: List[Path],
    out_path: Path,
    size: Tuple[int, int] = (1920, 1080),
    fps: int = 5,
    crf: str = "18",
    preset: str = "slow",
) -> None:
    """Save frames to an H.264 MP4 using imageio/ffmpeg.

    Notes:
        - Requires `imageio` and an ffmpeg backend available to imageio.
        - Output is yuv420p for broad compatibility (browsers, PowerPoint, QuickTime).

    Args:
        frame_paths: Frame paths in order.
        out_path: Output .mp4 path.
        size: (width, height) to resize frames to.
        fps: Frames per second.
        crf: H.264 constant rate factor (lower = higher quality).
        preset: ffmpeg preset (slower = better compression).

    Raises:
        RuntimeError: If imageio is not installed.
    """
    if imageio is None:
        raise RuntimeError("MP4 export requires imageio. Install with: pip install imageio imageio-ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", str(preset),
        ],
    )

    try:
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB").resize(size, Image.LANCZOS)
            writer.append_data(np.array(img))
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GIF/MP4 from a folder of frame images.")
    parser.add_argument("--input", required=True, type=str, help="Directory containing frames (png/jpg).")
    parser.add_argument("--gif", type=str, default=None, help="Output GIF path (optional).")
    parser.add_argument("--mp4", type=str, default=None, help="Output MP4 path (optional).")

    parser.add_argument("--duration-ms", type=int, default=200, help="GIF: ms per frame.")
    parser.add_argument("--loop", type=int, default=0, help="GIF: loop count (0=infinite).")

    parser.add_argument("--fps", type=int, default=5, help="MP4: frames per second.")
    parser.add_argument("--crf", type=str, default="18", help="MP4: quality (lower=better).")
    parser.add_argument("--preset", type=str, default="slow", help="MP4: ffmpeg preset.")

    parser.add_argument("--size", nargs=2, type=int, default=None, metavar=("W", "H"),
                        help="Optional resize (width height). Used for MP4 and optionally GIF.")

    args = parser.parse_args()

    input_dir = Path(args.input)
    frames_paths = list_frames(input_dir)

    size = tuple(args.size) if args.size is not None else None

    if args.gif is None and args.mp4 is None:
        raise SystemExit("Nothing to do: provide at least --gif or --mp4")

    # GIF (loads all frames into memory)
    if args.gif is not None:
        frames = load_frames_rgb(frames_paths, size=size)
        out_gif = Path(args.gif)
        save_gif(frames, out_gif, duration_ms=args.duration_ms, loop=args.loop)
        print(f"Saved GIF: {out_gif} | frames={len(frames)} | duration_ms={args.duration_ms}")

    # MP4 (streams frame-by-frame)
    if args.mp4 is not None:
        out_mp4 = Path(args.mp4)
        mp4_size = size if size is not None else (1920, 1080)
        save_mp4(frames_paths, out_mp4, size=mp4_size, fps=args.fps, crf=args.crf, preset=args.preset)
        print(f"Saved MP4: {out_mp4} | frames={len(frames_paths)} | fps={args.fps} | size={mp4_size}")


if __name__ == "__main__":
    main()
