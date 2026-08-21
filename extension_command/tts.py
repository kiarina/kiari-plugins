# Usage:
#   kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/tts.py" tts --tts-model local "Hello, world"
#
#   The examples below omit the `kiari ext -v --plugin ... tts` prefix:
#     --tts-model openai "Hello, world"
#     --tts-model google "Hello, world"
#     --ignore-cache "Excellent"
#     --ignore-cache --tts-model openai --instructions "激しく、力強く" "No programming, No life"
#     --ignore-cache --output-format m4a "こんにちは"
#     --ignore-cache --no-play "こんにちは"
import argparse
from collections.abc import Sequence
from pathlib import Path

from kiarina.agi import tts_model
from kiarina.agi.cost_recorder import cost_recorder_registry
from kiarina.agi.run_context import RunContext

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)
from kiari.lib.audio_utils import play_audio


class TTSCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args)
        text = " ".join(options.text).strip()

        if not text:
            raise ValueError("No text provided.")

        run_context = RunContext(agent_id="tts-command")
        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)

        audio_file_path = await tts_model.text_to_speech(
            text,
            tts_options={
                "tts_model": options.tts_model or context.run_options.tts_model,
                "instructions": options.instructions,
                "output_format": options.output_format,
                "ignore_cache": options.ignore_cache,
            },
            cost_recorder=cost_recorder,
            run_context=run_context,
        )

        print(f"Audio saved to: {audio_file_path}")
        print(f"File size: {Path(audio_file_path).stat().st_size} bytes")

        if cost_recorder.records:
            print("Cost records:")
            for record in cost_recorder.records:
                print(f" - {record}")

        await cost_recorder.flush(run_context)

        if options.play:
            await play_audio(audio_file_path)


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kiari ext tts",
        description="Generate speech with kiarina_agi TTSModel.",
    )
    # fmt: off
    parser.add_argument("text", nargs="+")
    parser.add_argument("--tts-model", help="TTS model specifier.")
    parser.add_argument("--instructions", help="Voice/style instructions.")
    parser.add_argument("--output-format", help="Output audio format.")
    parser.add_argument("--ignore-cache", action="store_true", help="Generate speech without reading an existing cache entry.")
    parser.add_argument("--play", dest="play", action="store_true", default=True, help="Play the generated audio after saving it.")
    parser.add_argument("--no-play", dest="play", action="store_false", help="Do not play the generated audio.")
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("tts", TTSCommand)
