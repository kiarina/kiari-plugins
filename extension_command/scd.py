# RunSpec:
#   plugins:
#     - kiari_plugins/**/*.py
#
# Usage:
#   mic:
#     kiari ext -v scd --asr-model local --output-dir ./.tmp/scd_mic
#   file:
#     kiari ext -v scd --input-file ./assets/asr/multi_speaker_audio.mp3 --audio-source file?sample_rate=16000 --asr-model local --output-dir ./.tmp/scd_file
import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from kiarina.agi import asr_model
from kiarina.agi.asr_provider import write_wav
from kiarina.agi.audio_source import AudioSource, audio_source_registry
from kiarina.agi.audio_types import AudioSamples
from kiarina.agi.cost_recorder import CostRecorder, cost_recorder_registry
from kiarina.agi.run_context import RunContext
from kiarina.agi.scd_model import scd_model_registry
from kiarina.agi.speaker_change_detector import (
    SpeakerChangeDetector,
    Speech,
    create_speaker_change_detector,
)
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


def _voice_metadata(voice: Voice) -> dict:
    return {
        "sample_rate": voice.sample_rate,
        "start_timestamp": voice.start_timestamp,
        "end_timestamp": voice.end_timestamp,
        "start_datetime": datetime.fromtimestamp(voice.start_timestamp).isoformat(),
        "end_datetime": datetime.fromtimestamp(voice.end_timestamp).isoformat(),
        "duration_ms": round((voice.end_timestamp - voice.start_timestamp) * 1000),
        "samples": int(len(voice.samples)),
        "metadata": voice.metadata,
    }


def _speech_metadata(speech: Speech) -> dict:
    return {
        "kind": speech.kind,
        "speaker_index": speech.speaker_index,
        "sample_rate": speech.sample_rate,
        "start_timestamp": speech.start_timestamp,
        "end_timestamp": speech.end_timestamp,
        "start_datetime": datetime.fromtimestamp(speech.start_timestamp).isoformat(),
        "end_datetime": datetime.fromtimestamp(speech.end_timestamp).isoformat(),
        "duration_ms": round((speech.end_timestamp - speech.start_timestamp) * 1000),
        "samples": int(len(speech.samples)),
        "metadata": speech.metadata,
    }


# --------------------------------------------------
# Schemas
# --------------------------------------------------


@dataclass(frozen=True)
class SavedSpeechFiles:
    voice_audio_file_path: Path
    metadata_file_path: Path
    speech_audio_file_paths: list[Path]
    speech_transcript_file_paths: list[Path] | None = None


# --------------------------------------------------
# Services
# --------------------------------------------------


class SpeechFileWriter:
    def __init__(self, output_dir: Path, *, metadata: dict) -> None:
        self.output_dir = output_dir
        self.metadata = metadata
        self._index = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, voice: Voice, speeches: list[Speech]) -> SavedSpeechFiles:
        self._index += 1
        stem = f"voice-{self._index:06d}"
        voice_audio_file_path = self.output_dir / f"{stem}.wav"
        metadata_file_path = self.output_dir / f"{stem}.json"
        speech_audio_file_paths: list[Path] = []

        write_wav(voice_audio_file_path, voice.samples, voice.sample_rate)

        for index, speech in enumerate(speeches, 1):
            speech_audio_file_path = self.output_dir / f"{stem}-speech-{index:03d}.wav"
            write_wav(speech_audio_file_path, speech.samples, speech.sample_rate)
            speech_audio_file_paths.append(speech_audio_file_path)

        metadata_file_path.write_text(
            json.dumps(
                {
                    "voice": _voice_metadata(voice),
                    "speeches": [
                        {
                            **_speech_metadata(speech),
                            "audio_file": str(speech_audio_file_paths[index]),
                        }
                        for index, speech in enumerate(speeches)
                    ],
                    "command": self.metadata,
                    "voice_audio_file": str(voice_audio_file_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return SavedSpeechFiles(
            voice_audio_file_path=voice_audio_file_path,
            metadata_file_path=metadata_file_path,
            speech_audio_file_paths=speech_audio_file_paths,
        )

    def save_transcripts(
        self, saved_files: SavedSpeechFiles, texts: list[str]
    ) -> list[Path]:
        transcript_file_paths: list[Path] = []

        for speech_audio_file_path, text in zip(
            saved_files.speech_audio_file_paths, texts
        ):
            transcript_file_path = speech_audio_file_path.with_suffix(".txt")
            transcript_file_path.write_text(text, encoding="utf-8")
            transcript_file_paths.append(transcript_file_path)

        self._update_metadata(
            saved_files.metadata_file_path,
            {
                "speeches": [
                    {
                        "transcript_file": str(transcript_file_path),
                        "transcript_text": text,
                    }
                    for transcript_file_path, text in zip(transcript_file_paths, texts)
                ],
            },
        )

        return transcript_file_paths

    def _update_metadata(self, metadata_file_path: Path, values: dict) -> None:
        metadata = json.loads(metadata_file_path.read_text(encoding="utf-8"))

        if "speeches" in values and "speeches" in metadata:
            for speech_metadata, speech_values in zip(
                metadata["speeches"], values["speeches"]
            ):
                speech_metadata.update(speech_values)

            values = {k: v for k, v in values.items() if k != "speeches"}

        metadata.update(values)
        metadata_file_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# --------------------------------------------------
# Main
# --------------------------------------------------


class SCDCommand(BaseExtensionCommand):
    def __init__(self) -> None:
        self._options: argparse.Namespace | None = None
        self._run_context: RunContext | None = None
        self._cost_recorder: CostRecorder | None = None
        self._speaker_change_detector: SpeakerChangeDetector | None = None
        self.speech_writer: SpeechFileWriter | None = None

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

    @property
    def speaker_change_detector(self) -> SpeakerChangeDetector:
        assert self._speaker_change_detector is not None
        return self._speaker_change_detector

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
        self._run_context = RunContext(agent_id="scd-command")
        self._cost_recorder = cost_recorder_registry.resolve(
            context.run_options.cost_recorder
        )

        target = Path(options.input_file).expanduser() if options.input_file else None

        output_dir = (
            Path(options.output_dir).expanduser() if options.output_dir else None
        )

        audio_source_specifier = options.audio_source_specifier or (
            "file" if options.input_file else "mic"
        )

        self.speech_writer = (
            SpeechFileWriter(
                output_dir,
                metadata={
                    "audio_source": audio_source_specifier,
                    "input_file": str(target) if target is not None else None,
                    "vad_model": options.vad_model,
                    "scd_model": options.scd_model,
                    "vad_threshold": options.vad_threshold,
                    "min_silence_ms": options.min_silence_ms,
                    "voice_pad_ms": options.voice_pad_ms,
                    "scd_threshold": options.scd_threshold,
                    "overlap_margin": options.overlap_margin,
                    "min_change_ms": options.min_change_ms,
                    "min_speech_ms": options.min_speech_ms,
                    "asr_model": options.asr_model,
                },
            )
            if output_dir is not None
            else None
        )

        print("Listening. Press Ctrl+C to stop.")
        print(f"Audio source: {audio_source_specifier}")
        print(f"VAD model: {options.vad_model}")
        print(f"SCD model: {options.scd_model}")
        print(f"VAD threshold: {options.vad_threshold}")
        print(f"SCD threshold: {options.scd_threshold}")

        if options.asr_model:
            print(f"ASR model: {options.asr_model}")

        if self.speech_writer:
            print(f"Output directory: {self.speech_writer.output_dir}")

        audio_source = audio_source_registry.resolve(audio_source_specifier)
        vad_model = vad_model_registry.resolve(options.vad_model)
        voice_detector = create_voice_detector(
            vad_model,
            threshold=options.vad_threshold,
            min_silence_ms=options.min_silence_ms,
            voice_pad_ms=options.voice_pad_ms,
        )

        scd_model = scd_model_registry.resolve(options.scd_model)
        self._speaker_change_detector = create_speaker_change_detector(
            scd_model,
            threshold=options.scd_threshold,
            overlap_margin=options.overlap_margin,
            min_change_ms=options.min_change_ms,
            min_speech_ms=options.min_speech_ms,
        )

        return target, audio_source, voice_detector

    async def _handle_voice(self, voice: Voice) -> None:
        print()
        print("Voice detected:")
        print(json.dumps(_voice_metadata(voice), ensure_ascii=False, indent=2))

        speeches = await self.speaker_change_detector.detect(
            voice.samples,
            voice.sample_rate,
            voice.start_timestamp,
        )

        print("Speeches:")

        for index, speech in enumerate(speeches, 1):
            print(
                f" - #{index}: "
                f"{json.dumps(_speech_metadata(speech), ensure_ascii=False)}"
            )

        saved_files: SavedSpeechFiles | None = None

        if self.speech_writer:
            saved_files = self.speech_writer.save(voice, speeches)
            print(f"Saved voice audio: {saved_files.voice_audio_file_path}")
            print(f"Saved metadata: {saved_files.metadata_file_path}")

            for speech_audio_file_path in saved_files.speech_audio_file_paths:
                print(f"Saved speech audio: {speech_audio_file_path}")

        if not self.options.asr_model:
            return

        texts: list[str] = []

        for index, speech in enumerate(speeches, 1):
            text = await asr_model.speech_to_text(
                speech.samples,
                speech.sample_rate,
                asr_options={
                    "asr_model": self.options.asr_model,
                },
                cost_recorder=self.cost_recorder,
                run_context=self.run_context,
            )
            texts.append(text)

            print("-" * 20)
            print(f"#{index}: {text}")
            print("-" * 20)

        if self.speech_writer and saved_files:
            transcript_file_paths = self.speech_writer.save_transcripts(
                saved_files, texts
            )

            for transcript_file_path in transcript_file_paths:
                print(f"Saved transcript: {transcript_file_path}")

        if self.cost_recorder.records:
            print("Cost records:")

            for record in self.cost_recorder.records:
                print(f" - {record}")


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kiari ext scd",
        description="Detect speaker changes with kiarina_agi SCDModel.",
    )
    # fmt: off
    parser.add_argument("--input-file", help="Input audio file path. If omitted, microphone input is used.")
    parser.add_argument("--output-dir", help="Directory to save voice/speech WAV files and metadata JSON files.")
    parser.add_argument("--audio-source", dest="audio_source_specifier", help="Audio source specifier.")
    parser.add_argument("--vad-model", default="silero", help="VAD model specifier.")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Voice probability threshold.")
    parser.add_argument("--min-silence-ms", type=int, default=500, help="Milliseconds of silence required to end a voice segment.")
    parser.add_argument("--voice-pad-ms", type=int, default=300, help="Padding milliseconds around voice segments.")
    parser.add_argument("--scd-model", default="pyannote", help="SCD model specifier.")
    parser.add_argument("--scd-threshold", type=float, default=0.5, help="Speaker probability threshold.")
    parser.add_argument("--overlap-margin", type=float, default=0.1, help="Top-two speaker probability margin for unknown overlap.")
    parser.add_argument("--min-change-ms", type=int, default=100, help="Minimum duration for a speaker state change.")
    parser.add_argument("--min-speech-ms", type=int, default=100, help="Minimum duration for an emitted speech segment.")
    parser.add_argument("--asr-model", help="ASR model specifier. If omitted, speech segments are not transcribed.")
    parser.add_argument("--debug", action="store_true", help="Print VAD probability and input audio level for each chunk.")
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("scd", SCDCommand)
