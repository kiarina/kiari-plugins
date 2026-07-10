# RunSpec:
#   plugins:
#     - kiari_plugins/**/*.py
#
# Usage:
#   local
#     kiari ext -v asr --asr-model local --output-file .tmp/asr_local/asr.txt ./assets/asr/multi_speaker_audio.mp3
#     kiari ext -v asr --asr-model local --segments --output-file .tmp/asr_local/asr.srt ./assets/asr/multi_speaker_audio.mp3
#   openai
#     kiari ext -v asr --asr-model openai --output-file .tmp/asr_openai/asr.txt ./assets/asr/multi_speaker_audio.mp3
#     kiari ext -v asr --asr-model openai --segments --output-file .tmp/asr_openai/asr.srt ./assets/asr/multi_speaker_audio.mp3
#   google
#     kiari ext -v asr --asr-model google --output-file .tmp/asr_google/asr.txt ./assets/asr/multi_speaker_audio.mp3
#     kiari ext -v asr --asr-model google --segments --output-file .tmp/asr_google/asr.srt ./assets/asr/multi_speaker_audio.mp3
import argparse
from collections.abc import Sequence
from pathlib import Path

from kiarina.agi import asr_model
from kiarina.agi.cost_recorder import cost_recorder_registry
from kiarina.agi.run_context import RunContext

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)
from kiari.lib.audio_utils import load_audio_samples


class ASRCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args)

        run_context = RunContext(agent_id="asr-command")
        cost_recorder = cost_recorder_registry.resolve(
            context.run_options.cost_recorder
        )
        samples, sample_rate = load_audio_samples(options.input_file)

        if options.segments:
            segments = await asr_model.speech_to_segments(
                samples,
                sample_rate,
                asr_options={
                    "asr_model": options.asr_model,
                },
                cost_recorder=cost_recorder,
                run_context=run_context,
            )
            text = "\n\n".join(
                f"{i}\n{segment.to_srt()}" for i, segment in enumerate(segments, 1)
            )

        else:
            text = await asr_model.speech_to_text(
                samples,
                sample_rate,
                asr_options={
                    "asr_model": options.asr_model,
                },
                cost_recorder=cost_recorder,
                run_context=run_context,
            )

        print("-" * 20)
        print(text)
        print("-" * 20)

        if options.output_file:
            output_path = Path(options.output_file).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            print(f"Transcript saved to: {output_path}")

        if cost_recorder.records:
            print("Cost records:")
            for record in cost_recorder.records:
                print(f" - {record}")

        await cost_recorder.flush(run_context)


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kiari ext asr",
        description="Transcribe audio with kiarina_agi ASRModel.",
    )
    # fmt: off
    parser.add_argument("input_file")
    parser.add_argument("--asr-model", help="ASR model specifier.")
    parser.add_argument("--output-file", help="Output transcript text file path.")
    parser.add_argument("--segments", action="store_true", help="Print segment details when the ASR provider returns them.")
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("asr", ASRCommand)
