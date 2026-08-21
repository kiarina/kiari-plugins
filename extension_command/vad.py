# Usage:
#   kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/vad.py" vad --asr-model local --output-dir ./.tmp/vad_mic    # mic
#
#   The examples below omit the `kiari ext -v --plugin ... vad` prefix:
#     --input-file ./assets/asr/multi_speaker_audio.mp3 --audio-source file?sample_rate=16000 --asr-model local --output-dir ./.tmp/vad_file    # file
import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from kiarina.agi import asr_model
from kiarina.agi.asr_provider import write_wav
from kiarina.agi.audio_source import AudioSource, audio_source_registry
from kiarina.agi.audio_types import AudioSamples
from kiarina.agi.cost_recorder import CostRecorder, cost_recorder_registry
from kiarina.agi.run_context import RunContext
from kiarina.agi.vad_model import vad_model_registry
from kiarina.agi.voice_detector import (
    Voice,
    VoiceDetector,
    create_voice_detector,
)

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

# --------------------------------------------------
# Utilities
# --------------------------------------------------


def _print_debug(chunk_count: int, samples: AudioSamples, probability: float) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    print(
        f"chunk={chunk_count} prob={probability:.3f} rms={rms:.5f} peak={peak:.5f}",
        flush=True,
    )


def _voice_metadata(voice: Voice) -> dict[str, Any]:
    return {
        "sample_rate": voice.sample_rate,
        "start_timestamp": voice.start_timestamp,
        "end_timestamp": voice.end_timestamp,
        "start_datetime": datetime.fromtimestamp(voice.start_timestamp).isoformat(),
        "end_datetime": datetime.fromtimestamp(voice.end_timestamp).isoformat(),
        "duration_ms": round((voice.end_timestamp - voice.start_timestamp) * 1000),
        "samples": len(voice.samples),
        "metadata": voice.metadata,
    }


# --------------------------------------------------
# Schemas
# --------------------------------------------------


@dataclass(frozen=True)
class SavedVoiceFiles:
    audio_file_path: Path
    metadata_file_path: Path
    transcript_file_path: Path | None = None


# --------------------------------------------------
# Services
# --------------------------------------------------


class VoiceFileWriter:
    def __init__(self, output_dir: Path, *, metadata: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.metadata = metadata
        self._index = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, voice: Voice) -> SavedVoiceFiles:
        self._index += 1
        stem = f"voice-{self._index:06d}"
        audio_file_path = self.output_dir / f"{stem}.wav"
        metadata_file_path = self.output_dir / f"{stem}.json"

        write_wav(audio_file_path, voice.samples, voice.sample_rate)

        metadata_file_path.write_text(
            json.dumps(
                {
                    **_voice_metadata(voice),
                    "command": self.metadata,
                    "audio_file": str(audio_file_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return SavedVoiceFiles(
            audio_file_path=audio_file_path,
            metadata_file_path=metadata_file_path,
        )

    def save_transcript(self, saved_files: SavedVoiceFiles, text: str) -> Path:
        transcript_file_path = saved_files.audio_file_path.with_suffix(".txt")
        transcript_file_path.write_text(text, encoding="utf-8")
        self._update_metadata(
            saved_files.metadata_file_path,
            {
                "transcript_file": str(transcript_file_path),
                "transcript_text": text,
            },
        )
        return transcript_file_path

    def _update_metadata(self, metadata_file_path: Path, values: dict[str, Any]) -> None:
        metadata = json.loads(metadata_file_path.read_text(encoding="utf-8"))
        metadata.update(values)
        metadata_file_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# --------------------------------------------------
# Main
# --------------------------------------------------


class VADCommand(BaseExtensionCommand):
    def __init__(self) -> None:
        self._options: argparse.Namespace | None = None
        self._run_context: RunContext | None = None
        self._cost_recorder: CostRecorder | None = None
        self.voice_writer: VoiceFileWriter | None = None

    @property
    def options(self) -> argparse.Namespace:
        assert self._options is not None
        return self._options

    @property
    def run_context(self) -> RunContext:
        assert self._run_context is not None
        return self._run_context

    @property
    def cost_recorder(self) -> CostRecorder:
        assert self._cost_recorder is not None
        return self._cost_recorder

    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        target, audio_source, voice_detector = self._setup(context, args)

        chunk_count = 0
        in_voice = False

        try:
            async with audio_source.open(target):
                async for chunk in audio_source.read():
                    result = await voice_detector.detect(
                        chunk.samples, chunk.sample_rate, chunk.timestamp
                    )

                    chunk_count += 1

                    if self.options.debug:
                        _print_debug(chunk_count, chunk.samples, result.probability)

                    if result.is_voice and not in_voice:
                        print(
                            f"Voice started: prob={result.probability:.3f}",
                            flush=True,
                        )
                        in_voice = True

                    if result.voice is not None:
                        in_voice = False
                        await self._handle_voice(result.voice)

            if voice := voice_detector.flush():
                await self._handle_voice(voice)

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Stopped.")

        finally:
            await self.cost_recorder.flush(self.run_context)

    def _setup(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> tuple[Path | None, AudioSource, VoiceDetector]:
        options = _parse_args(args)
        self._options = options
        self._run_context = RunContext(agent_id="vad-command")
        self._cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)

        target = Path(options.input_file).expanduser() if options.input_file else None

        output_dir = Path(options.output_dir).expanduser() if options.output_dir else None

        audio_source_specifier = (
            options.audio_source
            if options.audio_source
            else ("file" if options.input_file else "mic")
        )

        vad_model_specifier = options.vad_model if options.vad_model else "silero"

        self.voice_writer = (
            VoiceFileWriter(
                output_dir,
                metadata={
                    "audio_source": audio_source_specifier,
                    "input_file": str(target) if target is not None else None,
                    "vad_model": vad_model_specifier,
                    "threshold": options.threshold,
                    "min_silence_ms": options.min_silence_ms,
                    "voice_pad_ms": options.voice_pad_ms,
                },
            )
            if output_dir is not None
            else None
        )

        print("Listening. Press Ctrl+C to stop.")
        print(f"Audio source: {audio_source_specifier}")
        print(f"VAD model: {vad_model_specifier}")
        print(f"Threshold: {options.threshold}")

        if self.voice_writer:
            print(f"Output directory: {self.voice_writer.output_dir}")

        audio_source = audio_source_registry.resolve(audio_source_specifier)
        vad_model = vad_model_registry.resolve(vad_model_specifier)
        voice_detector = create_voice_detector(
            vad_model,
            threshold=options.threshold,
            min_silence_ms=options.min_silence_ms,
            voice_pad_ms=options.voice_pad_ms,
        )

        return target, audio_source, voice_detector

    async def _handle_voice(self, voice: Voice) -> None:
        print()
        print("Voice detected:")
        print(json.dumps(_voice_metadata(voice), ensure_ascii=False, indent=2))

        saved_files: SavedVoiceFiles | None = None

        if self.voice_writer:
            saved_files = self.voice_writer.save(voice)
            print(f"Saved audio: {saved_files.audio_file_path}")
            print(f"Saved metadata: {saved_files.metadata_file_path}")

        if not self.options.asr_model:
            return

        text = await asr_model.speech_to_text(
            voice.samples,
            voice.sample_rate,
            asr_options={
                "asr_model": self.options.asr_model,
            },
            cost_recorder=self.cost_recorder,
            run_context=self.run_context,
        )

        print("-" * 20)
        print(text)
        print("-" * 20)

        if self.voice_writer and saved_files:
            transcript_file_path = self.voice_writer.save_transcript(saved_files, text)
            print(f"Saved transcript: {transcript_file_path}")

        if self.cost_recorder.records:
            print("Cost records:")

            for record in self.cost_recorder.records:
                print(f" - {record}")


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kiari ext vad",
        description="Detect voice segments with kiarina_agi VADModel.",
    )
    # fmt: off
    parser.add_argument("--input-file", help="Input audio file path. If omitted, microphone input is used.")
    parser.add_argument("--output-dir", help="Directory to save detected voice WAV files and metadata JSON files.")
    parser.add_argument("--audio-source", help="Audio source specifier.")
    parser.add_argument("--vad-model", help="VAD model specifier.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Voice probability threshold.")
    parser.add_argument("--min-silence-ms", type=int, default=500, help="Milliseconds of silence required to end a voice segment.")
    parser.add_argument("--voice-pad-ms", type=int, default=300, help="Padding milliseconds around voice segments.")
    parser.add_argument("--asr-model", help="ASR model specifier. If omitted, only voice metadata is printed.")
    parser.add_argument("--debug", action="store_true", help="Print VAD probability and input audio level for each chunk.")
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("vad", VADCommand)
