# Usage:
#   kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/audio_embedding.py" audio-embedding add --store-dir ./.tmp/audio_embeddings/speaker --audio-embedding-model speaker ./sample.wav
#
#   The examples below omit the `kiari ext -v --plugin ... audio-embedding` prefix:
#     list --store-dir ./.tmp/audio_embeddings/speaker
#     search --store-dir ./.tmp/audio_embeddings/speaker --audio-embedding-model speaker --top-n 10 ./query.wav
import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from kiarina.agi.audio_embedding_model import embed_audio
from kiarina.agi.cost_recorder import cost_recorder_registry
from kiarina.agi.embedding import Embedding, search_embeddings
from kiarina.agi.run_context import RunContext

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)
from kiari.lib.audio_utils import load_audio_samples

type SearchResult = tuple[Embedding, float]


# --------------------------------------------------
# Utils
# --------------------------------------------------


def _entry_summary(entry: Embedding) -> dict[str, Any]:
    return {
        "id": entry.id,
        "label": entry.metadata.get("label") or entry.id,
        "created_at": entry.created_at.isoformat(),
        "audio_embedding_model": entry.metadata.get("audio_embedding_model"),
        "kind": entry.kind,
        "space_id": entry.space_id,
        "embedding_dim": len(entry.vector),
        "sample_rate": entry.metadata.get("sample_rate"),
        "duration_ms": entry.metadata.get("duration_ms"),
        "source_file": entry.metadata.get("source_file"),
    }


# --------------------------------------------------
# Services
# --------------------------------------------------


class AudioEmbeddingStore:
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir.expanduser()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_file_path = self.store_dir / "embeddings.json"

    def add(self, embedding: Embedding) -> Embedding:
        embeddings = self.list_entries()
        embeddings.append(embedding)
        self._save(embeddings)
        return embedding

    def list_entries(self) -> list[Embedding]:
        if not self.embeddings_file_path.exists():
            return []

        data = json.loads(self.embeddings_file_path.read_text(encoding="utf-8"))
        return [Embedding.model_validate(item) for item in data]

    def search(
        self,
        query: Embedding,
        *,
        top_n: int,
        include_all_spaces: bool,
        min_score: float | None,
    ) -> list[SearchResult]:
        embeddings = self.list_entries()
        search_query = query.to_numpy() if include_all_spaces else query
        results = search_embeddings(
            search_query,
            embeddings,
            top_k=top_n,
            min_score=min_score,
        )

        return [(result.embedding, result.score) for result in results]

    def _save(self, embeddings: list[Embedding]) -> None:
        self.embeddings_file_path.write_text(
            json.dumps(
                [embedding.model_dump(mode="json") for embedding in embeddings],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


# --------------------------------------------------
# Main
# --------------------------------------------------


class AudioEmbeddingCommand(BaseExtensionCommand):
    def __init__(
        self,
        *,
        default_model: str | None = None,
        default_min_score: float | None = None,
    ) -> None:
        super().__init__()
        self.default_model = default_model
        self.default_min_score = default_min_score

    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(
            args,
            default_model=self.default_model,
            default_min_score=self.default_min_score,
            prog=self.name,
        )
        store = AudioEmbeddingStore(Path(options.store_dir))

        if options.command == "add":
            await self._add(context, store, options)
        elif options.command == "list":
            self._list(store, options)
        elif options.command == "search":
            await self._search(context, store, options)
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {options.command}")

    async def _add(
        self,
        context: ExtensionCommandContext,
        store: AudioEmbeddingStore,
        options: argparse.Namespace,
    ) -> None:
        input_file_path = Path(options.input_file).expanduser()
        samples, sample_rate = load_audio_samples(input_file_path)
        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)
        run_context = RunContext(agent_id="audio-embedding-command")

        embedding = await embed_audio(
            samples,
            sample_rate,
            audio_embedding_options={
                "audio_embedding_model": options.audio_embedding_model,
            },
            cost_recorder=cost_recorder,
            run_context=run_context,
        )

        sample_count = len(samples)
        label = options.label or (
            f"{input_file_path.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        embedding.metadata = {
            **embedding.metadata,
            "label": label,
            "source_file": str(input_file_path),
            "audio_embedding_model": options.audio_embedding_model,
            "sample_rate": sample_rate,
            "samples": sample_count,
            "duration_ms": round(sample_count / sample_rate * 1000),
        }
        store.add(embedding)

        print(f"Saved embedding: {embedding.id}")
        print(f"Label: {label}")
        print(f"Embedding kind: {embedding.kind}")
        print(f"Embedding space: {embedding.space_id}")
        print(f"Embedding dim: {len(embedding.vector)}")
        print(f"Source file: {input_file_path}")
        print(f"Store file: {store.embeddings_file_path}")

        await cost_recorder.flush(run_context)

    def _list(self, store: AudioEmbeddingStore, options: argparse.Namespace) -> None:
        entries = store.list_entries()

        if options.json:
            print(
                json.dumps(
                    [_entry_summary(entry) for entry in entries],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        print(f"Store: {store.embeddings_file_path}")
        print(f"Embeddings: {len(entries)}")

        for index, entry in enumerate(entries, 1):
            summary = _entry_summary(entry)
            print(
                f"{index:03d} "
                f"label={summary['label']} "
                f"dim={summary['embedding_dim']} "
                f"kind={summary['kind']} "
                f"space={summary['space_id']} "
                f"source={summary['source_file']}"
            )

    async def _search(
        self,
        context: ExtensionCommandContext,
        store: AudioEmbeddingStore,
        options: argparse.Namespace,
    ) -> None:
        input_file_path = Path(options.input_file).expanduser()
        samples, sample_rate = load_audio_samples(input_file_path)
        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)
        run_context = RunContext(agent_id="audio-embedding-search-command")

        query = await embed_audio(
            samples,
            sample_rate,
            audio_embedding_options={
                "audio_embedding_model": options.audio_embedding_model,
            },
            cost_recorder=cost_recorder,
            run_context=run_context,
        )
        results = store.search(
            query,
            top_n=options.top_n,
            include_all_spaces=options.include_all_spaces,
            min_score=options.min_score,
        )

        if options.json:
            print(
                json.dumps(
                    [
                        {
                            "score": score,
                            **_entry_summary(entry),
                        }
                        for entry, score in results
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Query file: {input_file_path}")
            print(f"Query embedding kind: {query.kind}")
            print(f"Query embedding space: {query.space_id}")
            if options.min_score is not None:
                print(f"Minimum score: {options.min_score:.6f}")
            print(f"Matched embeddings: {len(results)}")

            for index, (entry, score) in enumerate(results, 1):
                print(
                    f"{index:03d} "
                    f"score={score:.6f} "
                    f"label={entry.metadata.get('label') or entry.id} "
                    f"source={entry.metadata.get('source_file')} "
                    f"id={entry.id}"
                )

        await cost_recorder.flush(run_context)


def _parse_args(
    args: Sequence[str],
    *,
    default_model: str | None,
    default_min_score: float | None,
    prog: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Create, list, and search audio embeddings.",
    )
    common_parser = argparse.ArgumentParser(add_help=False)

    # fmt: off
    common_parser.add_argument("--store-dir", required=True, help="Directory where embedding entries are stored.")
    common_parser.add_argument("--audio-embedding-model", default=default_model, help="Audio embedding model specifier.")
    common_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    # fmt: on

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(
        "add", parents=[common_parser], help="Create and store an embedding."
    )
    # fmt: off
    add_parser.add_argument("input_file")
    add_parser.add_argument("--label", help="Entry label. Defaults to filename + timestamp.")
    # fmt: on

    list_parser = subparsers.add_parser(
        "list", parents=[common_parser], help="List stored embeddings."
    )
    list_parser.set_defaults(command="list")

    search_parser = subparsers.add_parser(
        "search",
        parents=[common_parser],
        help="Search stored embeddings by cosine similarity.",
    )
    # fmt: off
    search_parser.add_argument("input_file")
    search_parser.add_argument("--top-n", type=int, default=10, help="Number of results to print.")
    search_parser.add_argument("--include-all-spaces", action="store_true", help="Search across embedding spaces. By default, only exact space_id matches are searched.")
    search_parser.add_argument("--min-score", type=float, default=default_min_score, help="Minimum cosine similarity score. Defaults to 0.45 for audio-embedding-speaker and no filter otherwise.")
    # fmt: on

    return parser.parse_args(list(args))


extension_command_registry.register("audio-embedding", AudioEmbeddingCommand)
