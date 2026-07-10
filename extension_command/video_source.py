# RunSpec:
#   plugins:
#     - kiari_plugins/**/*.py
#
# Usage:
#   camera:
#     kiari ext -v video-source --output-dir ./.tmp/video_source_camera --output-video
#   file:
#     kiari ext -v video-source --input-file ./assets/video/3.mp4 --output-dir ./.tmp/video_source_file --output-video
import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias, cast

import imageio.v3 as iio
import numpy as np
from jaxtyping import UInt8
from kiarina.agi.image_types import ImagePixels
from kiarina.agi.video_source import (
    VideoFrame,
    VideoSource,
    video_source_registry,
)
from PIL import Image

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

ImagePixelBatch: TypeAlias = UInt8[np.ndarray, "frames height width rgb"]

# --------------------------------------------------
# Utilities
# --------------------------------------------------


def _print_frame(frame: VideoFrame, *, previous_frame: VideoFrame | None) -> None:
    delta_ms = (
        round((frame.timestamp - previous_frame.timestamp) * 1000, 3)
        if previous_frame is not None
        else None
    )
    print(
        json.dumps(
            {
                "frame_index": frame.frame_index,
                "shape": list(frame.pixels.shape),
                "dtype": str(frame.pixels.dtype),
                "timestamp": frame.timestamp,
                "delta_ms": delta_ms,
                "mean": round(float(np.asarray(frame.pixels).mean()), 3)
                if frame.pixels.size
                else 0.0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _to_uint8_rgb(pixels: ImagePixels) -> ImagePixels:
    pixels = np.asarray(pixels)

    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    if pixels.ndim == 2:
        pixels = np.repeat(pixels[:, :, np.newaxis], 3, axis=2)

    if pixels.ndim != 3:
        raise ValueError(f"Unsupported frame pixels shape: {pixels.shape}")

    if pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    elif pixels.shape[2] == 4:
        pixels = pixels[:, :, :3]
    elif pixels.shape[2] != 3:
        raise ValueError(f"Unsupported frame pixels shape: {pixels.shape}")

    return cast(ImagePixels, pixels)


def _create_synthetic_video(*, frames: int, width: int, height: int) -> ImagePixelBatch:
    pixels: ImagePixelBatch = np.zeros((frames, height, width, 3), dtype=np.uint8)

    for frame_index in range(frames):
        pixels[frame_index, :, :, 0] = (frame_index * 31) % 256
        pixels[frame_index, :, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)
        pixels[frame_index, :, :, 2] = np.linspace(0, 255, height, dtype=np.uint8)[
            :, np.newaxis
        ]

    return pixels


def _format_target(target: object | None) -> str:
    if isinstance(target, np.ndarray):
        return f"<synthetic {list(target.shape)} {target.dtype}>"

    return str(target) if target is not None else "<default>"


def _estimate_fps(frames: list[VideoFrame]) -> float | None:
    if len(frames) < 2:
        return None

    duration = frames[-1].timestamp - frames[0].timestamp

    if duration <= 0:
        return None

    return (len(frames) - 1) / duration


def _summary(frames: list[VideoFrame]) -> dict:
    if not frames:
        return {
            "frames": 0,
            "duration_ms": 0,
            "estimated_fps": None,
        }

    duration = frames[-1].timestamp - frames[0].timestamp
    estimated_fps = _estimate_fps(frames)

    return {
        "frames": len(frames),
        "first_timestamp": frames[0].timestamp,
        "last_timestamp": frames[-1].timestamp,
        "duration_ms": round(duration * 1000, 3),
        "estimated_fps": round(estimated_fps, 3) if estimated_fps is not None else None,
        "first_shape": list(frames[0].pixels.shape),
        "last_shape": list(frames[-1].pixels.shape),
    }


# --------------------------------------------------
# Services
# --------------------------------------------------


class FrameFileWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(self, frame: VideoFrame) -> Path:
        pixels = np.asarray(frame.pixels)

        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)

        if pixels.ndim == 2:
            image = Image.fromarray(pixels)
        elif pixels.ndim == 3 and pixels.shape[2] in (1, 3, 4):
            if pixels.shape[2] == 1:
                pixels = pixels[:, :, 0]

            image = Image.fromarray(pixels)
        else:
            raise ValueError(f"Unsupported frame pixels shape: {pixels.shape}")

        frame_file_path = self.output_dir / f"frame-{frame.frame_index:06d}.png"
        image.save(frame_file_path)
        return frame_file_path

    def save_video(self, frames: list[VideoFrame], *, fps: float) -> Path:
        if not frames:
            raise ValueError("Cannot save video with no frames.")

        pixels = np.stack([_to_uint8_rgb(frame.pixels) for frame in frames], axis=0)
        video_file_path = self.output_dir / "frames.mp4"
        iio.imwrite(video_file_path, pixels, fps=fps)
        return video_file_path

    def save_metadata(self, frames: list[VideoFrame], summary: dict) -> Path:
        metadata_file_path = self.output_dir / "metadata.json"
        metadata_file_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "frames": [
                        {
                            "frame_index": frame.frame_index,
                            "timestamp": frame.timestamp,
                            "shape": list(frame.pixels.shape),
                            "dtype": str(frame.pixels.dtype),
                        }
                        for frame in frames
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return metadata_file_path


# --------------------------------------------------
# Main
# --------------------------------------------------


class VideoSourceCommand(BaseExtensionCommand):
    def __init__(self) -> None:
        self._options: argparse.Namespace | None = None
        self.frame_writer: FrameFileWriter | None = None

    @property
    def options(self) -> argparse.Namespace:
        assert self._options is not None
        return self._options

    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        target, video_source = self._setup(args)

        frames: list[VideoFrame] = []

        async with video_source.open(target):
            async for frame in video_source.read():
                frames.append(frame)
                _print_frame(
                    frame, previous_frame=frames[-2] if len(frames) > 1 else None
                )

                if self.frame_writer is not None:
                    self.frame_writer.save_frame(frame)

                if len(frames) >= self.options.frames:
                    break

        print()
        print("Summary:")
        summary = _summary(frames)

        if self.frame_writer is not None:
            self.frame_writer.save_metadata(frames, summary)

            if self.options.output_video:
                if fps := _estimate_fps(frames):
                    video_file_path = self.frame_writer.save_video(frames, fps=fps)
                    summary["video_file"] = str(video_file_path)
                    summary["video_fps"] = fps
                else:
                    print(
                        "Skipping video file: FPS could not be estimated from timestamps.",
                        flush=True,
                    )
                    summary["video_file"] = None
                    summary["video_fps"] = None

        print(json.dumps(summary, ensure_ascii=False, indent=2))

    def _setup(
        self,
        args: Sequence[str],
    ) -> tuple[Path | ImagePixelBatch | None, VideoSource]:
        options = _parse_args(args)
        self._options = options

        target: Path | ImagePixelBatch | None = (
            Path(options.input_file).expanduser() if options.input_file else None
        )
        video_source_specifier = options.video_source or (
            "file" if target else "camera"
        )
        video_source = video_source_registry.resolve(video_source_specifier)

        if target is None and video_source.name == "numpy":
            target = _create_synthetic_video(
                frames=options.frames,
                width=options.synthetic_width,
                height=options.synthetic_height,
            )

        output_dir = (
            Path(options.output_dir).expanduser() if options.output_dir else None
        )

        self.frame_writer = (
            FrameFileWriter(output_dir) if output_dir is not None else None
        )

        print(f"Video source: {video_source.name}")
        print(f"Target: {_format_target(target)}")
        print(f"Frames: {options.frames}")

        if self.frame_writer is not None:
            print(f"Output directory: {self.frame_writer.output_dir}")

        return target, video_source


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kiari ext video-source",
        description="Inspect frames emitted by kiarina_agi VideoSource.",
    )
    # fmt: off
    parser.add_argument("--input-file", help="Input video file path. If omitted, the selected source receives None.")
    parser.add_argument("--output-dir", help="Directory to save emitted frames as PNG files.")
    parser.add_argument("--output-video", dest="output_video", action="store_true", help="Save emitted frames as frames.mp4 when --output-dir is set.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to read.")
    parser.add_argument("--video-source", help="Video source specifier.")
    parser.add_argument("--synthetic-width", type=int, default=64, help="Synthetic frame width used when --source resolves to numpy without --input-file.")
    parser.add_argument("--synthetic-height", type=int, default=48, help="Synthetic frame height used when --source resolves to numpy without --input-file.")
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("video-source", VideoSourceCommand)
