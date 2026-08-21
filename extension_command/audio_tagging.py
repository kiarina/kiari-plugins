# Usage:
#   kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/audio_tagging.py" audio-tagging ./sample.wav    # default model (alias `local` -> yamnet)
#
#   The examples below omit the `kiari ext -v --plugin ... audio-tagging` prefix:
#     --audio-tagging-model yamnet --top-k 10 ./sample.wav    # choose model and top-k
#     --threshold 0.1 --json ./sample.wav    # threshold filter, JSON to stdout
#     --top-k 20 --output-file .tmp/audio_tagging/sample.json ./sample.wav
import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from kiarina.agi.audio_tagging_model import (
    AudioTaggingOptions,
    tag_audio,
)
from kiarina.agi.cost_recorder import cost_recorder_registry
from kiarina.agi.run_context import RunContext

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)
from kiari.lib.audio_utils import load_audio_samples


class AudioTaggingCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args)

        input_file_path = Path(options.input_file).expanduser()
        samples, sample_rate = load_audio_samples(input_file_path)

        run_context = RunContext(agent_id="audio-tagging-command")
        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)

        audio_tagging_options: AudioTaggingOptions = {}

        if options.audio_tagging_model:
            audio_tagging_options["audio_tagging_model"] = options.audio_tagging_model

        if options.top_k is not None:
            audio_tagging_options["top_k"] = options.top_k

        if options.threshold is not None:
            audio_tagging_options["threshold"] = options.threshold

        predictions = await tag_audio(
            samples,
            sample_rate,
            audio_tagging_options=audio_tagging_options,
            cost_recorder=cost_recorder,
            run_context=run_context,
        )

        result_payload = {
            "input_file": str(input_file_path),
            "audio_tagging_model": options.audio_tagging_model,
            "sample_rate": sample_rate,
            "samples": len(samples),
            "duration_ms": round(len(samples) / sample_rate * 1000),
            "top_k": options.top_k,
            "threshold": options.threshold,
            "predictions": [{"label": p.label, "score": p.score} for p in predictions],
        }

        if options.json:
            print(json.dumps(result_payload, ensure_ascii=False, indent=2))
        else:
            print(f"Input file: {input_file_path}")
            print(f"Audio tagging model: {options.audio_tagging_model or '(default)'}")
            print(f"Sample rate: {sample_rate} Hz")
            print(
                f"Duration: {result_payload['duration_ms']} ms "
                f"({result_payload['samples']} samples)"
            )

            if options.threshold is not None:
                print(f"Threshold: {options.threshold}")

            print(f"Predictions: {len(predictions)}")
            print("-" * 20)

            for index, prediction in enumerate(predictions, 1):
                print(f"{index:03d} score={prediction.score:.6f} label={prediction.label}")

            print("-" * 20)

        if options.output_file:
            output_path = Path(options.output_file).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Result saved to: {output_path}")

        if cost_recorder.records:
            print("Cost records:")
            for record in cost_recorder.records:
                print(f" - {record}")

        await cost_recorder.flush(run_context)


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kiari ext audio-tagging",
        description="Tag an audio file with kiarina_agi AudioTaggingModel.",
    )
    # fmt: off
    parser.add_argument("input_file")
    parser.add_argument("--audio-tagging-model", help="Audio tagging model specifier.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top predictions to keep. Defaults to 10. Pass 0 to disable.")
    parser.add_argument("--threshold", type=float, help="Minimum score to keep a prediction.")
    parser.add_argument("--output-file", help="Output JSON file path.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    # fmt: on

    options = parser.parse_args(list(args))

    if options.top_k is not None and options.top_k <= 0:
        options.top_k = None

    return options


extension_command_registry.register("audio-tagging", AudioTaggingCommand)
