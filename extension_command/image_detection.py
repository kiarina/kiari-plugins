# RunSpec:
#   plugins:
#     - kiari_plugins/**/*.py
#
# Usage:
#   object detection (alias `object` -> D-FINE):
#     kiari ext -v image-detection ./assets/jpg/3.jpg
#   face detection (alias `face` -> YuNet, draws 5-point keypoints):
#     kiari ext -v image-detection --faces ./assets/image_detection/1.png
#   choose model explicitly and save an annotated image:
#     kiari ext -v image-detection --image-detection-model yunet --output-image .tmp/image_detection/out.png ./assets/face.jpg
#   filter by score and write JSON to file:
#     kiari ext -v image-detection --score-threshold 0.5 --json --output-file .tmp/image_detection/out.json ./assets/jpg/3.jpg
#   save cropped objects and ArcFace-aligned faces:
#     kiari ext -v image-detection --crop-dir .tmp/image_detection_objects/crops --align-dir .tmp/image_detection_objects/aligned ./assets/image_detection/1.png
#     kiari ext -v image-detection --faces --crop-dir .tmp/image_detection_faces/crops --align-dir .tmp/image_detection_faces/aligned ./assets/image_detection/1.png
import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from kiarina.agi.cost_recorder import cost_recorder_registry
from kiarina.agi.image_detection_model import (
    ImageDetectionOptions,
    crop_align_faces,
    crop_objects,
    detect_faces,
    detect_objects,
)
from kiarina.agi.image_detection_provider import DetectedObject
from kiarina.agi.image_types import ImagePixels
from kiarina.agi.run_context import RunContext
from PIL import Image, ImageColor, ImageDraw

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
]
_KEYPOINT_COLOR = "#00ffff"


class ImageDetectionCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args)

        input_file_path = Path(options.input_file).expanduser()
        image = Image.open(input_file_path).convert("RGB")
        pixels: ImagePixels = np.array(image, dtype=np.uint8)
        height, width = pixels.shape[:2]

        run_context = RunContext(agent_id="image-detection-command")
        cost_recorder = cost_recorder_registry.resolve(
            context.run_options.cost_recorder
        )

        image_detection_options: ImageDetectionOptions = {}

        if options.image_detection_model:
            image_detection_options["image_detection_model"] = (
                options.image_detection_model
            )

        detect = detect_faces if options.faces else detect_objects

        detections = await detect(
            pixels,
            image_detection_options=image_detection_options,
            cost_recorder=cost_recorder,
            run_context=run_context,
        )

        if options.score_threshold is not None:
            detections = [d for d in detections if d.score >= options.score_threshold]

        detections = sorted(detections, key=lambda d: d.score, reverse=True)

        result_payload = {
            "input_file": str(input_file_path),
            "image_detection_model": options.image_detection_model,
            "faces": options.faces,
            "width": width,
            "height": height,
            "score_threshold": options.score_threshold,
            "detections": [
                {
                    "label": d.label,
                    "score": d.score,
                    "bbox": d.bbox,
                    "keypoint_type": d.keypoint_type,
                    "keypoints": d.keypoints,
                }
                for d in detections
            ],
        }

        if options.json:
            print(json.dumps(result_payload, ensure_ascii=False, indent=2))
        else:
            print(f"Input file: {input_file_path}")
            print(
                f"Image detection model: "
                f"{options.image_detection_model or ('face' if options.faces else 'object')}"
            )
            print(f"Image size: {width}x{height}")

            if options.score_threshold is not None:
                print(f"Score threshold: {options.score_threshold}")

            print(f"Detections: {len(detections)}")
            print("-" * 20)

            for index, detection in enumerate(detections, 1):
                bbox = ", ".join(f"{v:.3f}" for v in detection.bbox)
                line = (
                    f"{index:03d} score={detection.score:.4f} "
                    f"label={detection.label} bbox=[{bbox}]"
                )

                if detection.keypoints:
                    line += (
                        f" keypoints={len(detection.keypoints)}"
                        f"({detection.keypoint_type})"
                    )

                print(line)

            print("-" * 20)

        if options.output_file:
            output_path = Path(options.output_file).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Result saved to: {output_path}")

        if options.output_image:
            output_image_path = Path(options.output_image).expanduser()
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = _draw_detections(image, detections)
            annotated.save(output_image_path)
            print(f"Annotated image saved to: {output_image_path}")

        if options.crop_dir:
            crop_dir = Path(options.crop_dir).expanduser()
            crop_dir.mkdir(parents=True, exist_ok=True)

            for index, cropped in enumerate(crop_objects(pixels, detections)):
                if cropped.pixels.size == 0:
                    continue

                crop_path = crop_dir / f"crop-{index:03d}-{_slug(cropped.label)}.png"
                Image.fromarray(cropped.pixels).save(crop_path)

            print(f"Cropped objects saved to: {crop_dir}")

        if options.align_dir:
            align_dir = Path(options.align_dir).expanduser()
            align_dir.mkdir(parents=True, exist_ok=True)

            aligned_faces = crop_align_faces(pixels, detections, options.align_size)

            for index, aligned in enumerate(aligned_faces):
                Image.fromarray(aligned).save(align_dir / f"aligned-{index:03d}.png")

            print(f"Aligned faces saved to: {align_dir} ({len(aligned_faces)} faces)")

        if cost_recorder.records:
            print("Cost records:")
            for record in cost_recorder.records:
                print(f" - {record}")

        await cost_recorder.flush(run_context)


def _draw_detections(
    image: Image.Image, detections: Sequence[DetectedObject]
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    for index, detection in enumerate(detections):
        color = _PALETTE[index % len(_PALETTE)]

        x1 = detection.bbox[0] * width
        y1 = detection.bbox[1] * height
        x2 = detection.bbox[2] * width
        y2 = detection.bbox[3] * height
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        caption = f"{detection.label} {detection.score:.2f}"
        _draw_caption(draw, x1, y1, caption, color)

        radius = max(2, round(min(width, height) * 0.004))

        for point in detection.keypoints:
            px = point[0] * width
            py = point[1] * height
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=_KEYPOINT_COLOR,
                outline=_KEYPOINT_COLOR,
            )

    return annotated


def _draw_caption(
    draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: str
) -> None:
    text_box = draw.textbbox((0, 0), text)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    top = max(0.0, y - text_height - 4)
    draw.rectangle((x, top, x + text_width + 4, top + text_height + 4), fill=color)
    draw.text((x + 2, top + 2), text, fill=_text_color(color))


def _text_color(background: str) -> str:
    r, g, b = ImageColor.getrgb(background)[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 140 else "#ffffff"


def _slug(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_") or "object"


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kiari ext image-detection",
        description="Detect faces/objects in an image with kiarina_agi ImageDetectionModel.",
    )
    # fmt: off
    parser.add_argument("input_file")
    parser.add_argument("--image-detection-model", help="Image detection model specifier. Defaults to the `object` (or `face` with --faces) alias.")
    parser.add_argument("--faces", action="store_true", help="Use detect_faces (defaults to the `face` alias) instead of detect_objects.")
    parser.add_argument("--score-threshold", type=float, help="Drop detections whose score is below this value.")
    parser.add_argument("--output-image", help="Path to save the annotated image (bbox + label + keypoints).")
    parser.add_argument("--output-file", help="Output JSON file path.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--crop-dir", help="Directory to save each detection's bbox crop as a PNG.")
    parser.add_argument("--align-dir", help="Directory to save ArcFace-aligned crops of face_5pt detections as PNGs.")
    parser.add_argument("--align-size", type=int, default=112, help="Output square size for --align-dir. Defaults to 112.")
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("image-detection", ImageDetectionCommand)
