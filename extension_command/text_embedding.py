# Usage:
#   kiari ext -v text-embedding add --store-dir ./.tmp/text_embeddings --text-embedding-model local "Apple SiliconでローカルLLMを動かす"
#
#   The examples below omit the `kiari ext -v text-embedding` prefix:
#     add --store-dir ./.tmp/text_embeddings --input-file ./note.txt --label note-1
#     list --store-dir ./.tmp/text_embeddings
#     search --store-dir ./.tmp/text_embeddings --text-embedding-model local --top-n 10 "日本語検索に強い埋め込み"
#     validate --text-embedding-model local --samples-per-class 4    # built-in labeled text set, measure kNN retrieval accuracy
import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from kiarina.agi.cost_recorder import cost_recorder_registry
from kiarina.agi.embedding import Embedding, search_embeddings
from kiarina.agi.run_context import RunContext
from kiarina.agi.text_embedding_model import embed_text

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

type SearchResult = tuple[Embedding, float]
type ValidationBundle = tuple[list[str], list[int], dict[int, str]]

_VALIDATION_DATASET: dict[str, list[str]] = {
    "animals": [
        "Dogs are loyal animals that enjoy walking with people.",
        "Cats often sleep in sunny places and groom their fur.",
        "Birds build nests and communicate with songs.",
        "Horses can run quickly across open fields.",
        "犬は人と散歩するのが好きな賢い動物です。",
        "猫は日なたで眠り、毛づくろいをよくします。",
    ],
    "technology": [
        "Apple Silicon makes local machine learning workloads efficient.",
        "Vector databases store embeddings for semantic search.",
        "A GPU can accelerate neural network inference.",
        "Open source models can run on a laptop with enough memory.",
        "ローカルLLMは開発者の試行錯誤を速くします。",
        "ベクトル検索は意味の近い文章を探すために使われます。",
    ],
    "food": [
        "Fresh apples and oranges are common breakfast fruit.",
        "A bowl of ramen has noodles, broth, and toppings.",
        "Coffee is often brewed in the morning.",
        "Bakers use flour, yeast, and water to make bread.",
        "寿司は酢飯と魚を組み合わせた料理です。",
        "カレーには香辛料と野菜や肉がよく使われます。",
    ],
    "travel": [
        "Travelers book hotels before visiting a new city.",
        "A train station can connect several regional lines.",
        "Airports handle luggage, security checks, and boarding.",
        "A map helps hikers choose a safe route.",
        "京都旅行では寺社や庭園を巡る人が多いです。",
        "飛行機の搭乗前には手荷物検査があります。",
    ],
}


# --------------------------------------------------
# Utils
# --------------------------------------------------


def _load_text(options: argparse.Namespace) -> tuple[str, str | None]:
    if options.input_file:
        input_file_path = Path(options.input_file).expanduser()
        return input_file_path.read_text(encoding=options.encoding), str(input_file_path)

    if options.text is not None:
        return options.text, None

    raise ValueError("Either TEXT or --input-file must be provided.")


def _text_preview(text: str, *, max_length: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized

    return normalized[: max_length - 3] + "..."


def _entry_summary(entry: Embedding) -> dict[str, Any]:
    return {
        "id": entry.id,
        "label": entry.metadata.get("label") or entry.id,
        "created_at": entry.created_at.isoformat(),
        "text_embedding_model": entry.metadata.get("text_embedding_model"),
        "kind": entry.kind,
        "space_id": entry.space_id,
        "embedding_dim": len(entry.vector),
        "text_length": entry.metadata.get("text_length"),
        "text_preview": entry.metadata.get("text_preview"),
        "source_file": entry.metadata.get("source_file"),
    }


def _create_run_context(
    context: ExtensionCommandContext,
    *,
    agent_id: str,
) -> RunContext:
    values = {
        "organization_id": context.run_options.organization_id,
        "user_id": context.run_options.user_id,
        "agent_id": agent_id,
        "node_id": context.run_options.node_id or agent_id,
        "language": context.run_options.language or "en",
        "timezone": context.run_options.timezone or "UTC",
        "currency": context.run_options.currency or "USD",
    }
    run_context_kwargs: dict[str, Any] = {key: value for key, value in values.items() if value}
    return RunContext(**run_context_kwargs)


def _load_validation_dataset(
    *,
    samples_per_class: int,
    num_classes: int,
    seed: int,
) -> ValidationBundle:
    rng = np.random.default_rng(seed)
    class_names = sorted(_VALIDATION_DATASET)

    if num_classes > 0 and len(class_names) > num_classes:
        chosen = rng.choice(np.asarray(class_names), size=num_classes, replace=False)
        class_names = sorted(str(name) for name in chosen)

    texts: list[str] = []
    labels: list[int] = []
    names: dict[int, str] = {}

    for label, class_name in enumerate(class_names):
        candidates = _VALIDATION_DATASET[class_name]
        take = min(samples_per_class, len(candidates))
        indices = rng.choice(len(candidates), size=take, replace=False)

        for index in sorted(int(i) for i in indices):
            texts.append(candidates[index])
            labels.append(label)

        names[label] = class_name

    return texts, labels, names


# --------------------------------------------------
# Services
# --------------------------------------------------


class TextEmbeddingStore:
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


class TextEmbeddingCommand(BaseExtensionCommand):
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

        if options.command == "add":
            await self._add(context, TextEmbeddingStore(Path(options.store_dir)), options)
        elif options.command == "list":
            self._list(TextEmbeddingStore(Path(options.store_dir)), options)
        elif options.command == "search":
            await self._search(context, TextEmbeddingStore(Path(options.store_dir)), options)
        elif options.command == "validate":
            await self._validate(context, options)
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {options.command}")

    async def _add(
        self,
        context: ExtensionCommandContext,
        store: TextEmbeddingStore,
        options: argparse.Namespace,
    ) -> None:
        text, source_file = _load_text(options)
        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)
        run_context = _create_run_context(
            context,
            agent_id="text-embedding-command",
        )

        embedding = await embed_text(
            text,
            text_embedding_options={
                "text_embedding_model": options.text_embedding_model,
            },
            cost_recorder=cost_recorder,
            run_context=run_context,
        )

        label_base = Path(source_file).stem if source_file else "text"
        label = options.label or (f"{label_base}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        embedding.metadata = {
            **embedding.metadata,
            "label": label,
            "source_file": source_file,
            "text_embedding_model": options.text_embedding_model,
            "text_length": len(text),
            "text_preview": _text_preview(text),
        }
        store.add(embedding)

        print(f"Saved embedding: {embedding.id}")
        print(f"Label: {label}")
        print(f"Embedding kind: {embedding.kind}")
        print(f"Embedding space: {embedding.space_id}")
        print(f"Embedding dim: {len(embedding.vector)}")
        print(f"Text preview: {_text_preview(text)}")
        if source_file:
            print(f"Source file: {source_file}")
        print(f"Store file: {store.embeddings_file_path}")

        await cost_recorder.flush(run_context)

    def _list(self, store: TextEmbeddingStore, options: argparse.Namespace) -> None:
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
                f"text={summary['text_preview']!r}"
            )

    async def _search(
        self,
        context: ExtensionCommandContext,
        store: TextEmbeddingStore,
        options: argparse.Namespace,
    ) -> None:
        text, source_file = _load_text(options)
        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)
        run_context = _create_run_context(
            context,
            agent_id="text-embedding-search-command",
        )

        query = await embed_text(
            text,
            text_embedding_options={
                "text_embedding_model": options.text_embedding_model,
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
            if source_file:
                print(f"Query file: {source_file}")
            print(f"Query text: {_text_preview(text)}")
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
                    f"text={entry.metadata.get('text_preview')!r} "
                    f"id={entry.id}"
                )

        await cost_recorder.flush(run_context)

    async def _validate(
        self,
        context: ExtensionCommandContext,
        options: argparse.Namespace,
    ) -> None:
        texts, labels, names = _load_validation_dataset(
            samples_per_class=options.samples_per_class,
            num_classes=options.num_classes,
            seed=options.seed,
        )

        if len(texts) < 2:
            raise ValueError("Need at least 2 samples to evaluate.")

        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)
        run_context = _create_run_context(
            context,
            agent_id="text-embedding-validate-command",
        )

        print(
            f"Embedding {len(texts)} texts from dataset={options.dataset} "
            f"with text-embedding-model={options.text_embedding_model or 'default'} ..."
        )

        vectors: list[np.ndarray] = []

        for count, text in enumerate(texts, 1):
            embedding = await embed_text(
                text,
                text_embedding_options={
                    "text_embedding_model": options.text_embedding_model,
                },
                cost_recorder=cost_recorder,
                run_context=run_context,
            )
            vectors.append(embedding.to_numpy())

            if count % 20 == 0 or count == len(texts):
                print(f"  embedded {count}/{len(texts)}")

        report = _evaluate_knn(np.stack(vectors), np.asarray(labels), k=options.k)

        await cost_recorder.flush(run_context)

        if options.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return

        print()
        print(f"Dataset: {options.dataset}")
        print(f"Samples: {report['num_samples']} (k={report['k']})")
        print(f"Top-1 kNN accuracy: {report['top1_accuracy'] * 100:.2f}%")
        print(f"Mean same-class similarity:      {report['mean_same_class_similarity']:.4f}")
        print(f"Mean different-class similarity: {report['mean_different_class_similarity']:.4f}")
        print(f"Per-class accuracy ({len(report['per_class_accuracy'])} classes):")
        for class_id, accuracy in sorted(report["per_class_accuracy"].items()):
            name = names.get(int(class_id), str(class_id))
            print(f"  {name:<24} {accuracy * 100:6.2f}%")


def _evaluate_knn(vectors: np.ndarray, labels: np.ndarray, *, k: int) -> dict[str, Any]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = vectors / norms
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)

    neighbors = np.argsort(-similarity, axis=1)[:, :k]
    correct = 0
    per_class_total: dict[int, int] = {}
    per_class_correct: dict[int, int] = {}

    for index, label in enumerate(labels):
        neighbor_labels = labels[neighbors[index]]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        predicted = int(values[int(np.argmax(counts))])
        label_int = int(label)
        per_class_total[label_int] = per_class_total.get(label_int, 0) + 1

        if predicted == label_int:
            correct += 1
            per_class_correct[label_int] = per_class_correct.get(label_int, 0) + 1

    same_mask = labels[:, None] == labels[None, :]
    np.fill_diagonal(same_mask, False)
    finite = np.isfinite(similarity)
    same = similarity[same_mask & finite]
    diff = similarity[(~same_mask) & finite]

    return {
        "num_samples": len(labels),
        "k": k,
        "top1_accuracy": correct / len(labels),
        "mean_same_class_similarity": float(same.mean()) if same.size else 0.0,
        "mean_different_class_similarity": float(diff.mean()) if diff.size else 0.0,
        "per_class_accuracy": {
            str(class_id): per_class_correct.get(class_id, 0) / total
            for class_id, total in per_class_total.items()
        },
    }


def _add_text_input_args(parser: argparse.ArgumentParser) -> None:
    # fmt: off
    parser.add_argument("text", nargs="?", help="Text to embed. Use --input-file for file input.")
    parser.add_argument("--input-file", help="Read text from a UTF-8 file instead of the positional TEXT.")
    parser.add_argument("--encoding", default="utf-8", help="Input file encoding.")
    # fmt: on


def _parse_args(
    args: Sequence[str],
    *,
    default_model: str | None,
    default_min_score: float | None,
    prog: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Create, list, search, and validate text embeddings.",
    )
    common_parser = argparse.ArgumentParser(add_help=False)

    # fmt: off
    common_parser.add_argument("--text-embedding-model", default=default_model, help="Text embedding model specifier (e.g. local, openai, google).")
    common_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    # fmt: on

    store_parser = argparse.ArgumentParser(add_help=False)
    store_parser.add_argument(
        "--store-dir",
        required=True,
        help="Directory where embedding entries are stored.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(
        "add",
        parents=[common_parser, store_parser],
        help="Create and store an embedding.",
    )
    _add_text_input_args(add_parser)
    add_parser.add_argument("--label", help="Entry label. Defaults to timestamp.")

    subparsers.add_parser(
        "list",
        parents=[common_parser, store_parser],
        help="List stored embeddings.",
    )

    search_parser = subparsers.add_parser(
        "search",
        parents=[common_parser, store_parser],
        help="Search stored embeddings by cosine similarity.",
    )
    _add_text_input_args(search_parser)
    # fmt: off
    search_parser.add_argument("--top-n", type=int, default=10, help="Number of results to print.")
    search_parser.add_argument("--include-all-spaces", action="store_true", help="Search across embedding spaces. By default, only exact space_id matches are searched.")
    search_parser.add_argument("--min-score", type=float, default=default_min_score, help="Minimum cosine similarity score.")
    # fmt: on

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[common_parser],
        help="Measure kNN retrieval accuracy on a built-in labeled text set.",
    )
    # fmt: off
    validate_parser.add_argument("--dataset", choices=["semantic-basic"], default="semantic-basic", help="Dataset to validate against.")
    validate_parser.add_argument("--samples-per-class", type=int, default=6, help="Max texts to sample per class.")
    validate_parser.add_argument("--num-classes", type=int, default=0, help="Number of classes to sample (0 = all).")
    validate_parser.add_argument("--k", type=int, default=1, help="Number of neighbors for kNN classification.")
    validate_parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    # fmt: on

    options = parser.parse_args(list(args))

    if options.command in ("add", "search"):
        has_text = options.text is not None
        has_input_file = options.input_file is not None

        if has_text == has_input_file:
            parser.error("Exactly one of TEXT or --input-file must be provided.")

    if options.command == "validate" and options.k < 1:
        parser.error("--k must be >= 1.")

    return options


extension_command_registry.register("text-embedding", TextEmbeddingCommand)
