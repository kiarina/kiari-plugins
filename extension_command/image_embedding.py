# Usage:
#   kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/image_embedding.py" image-embedding add --store-dir ./.tmp/image_embeddings/object --image-embedding-model object ./sample.jpg
#
#   The examples below omit the `kiari ext -v --plugin ... image-embedding` prefix:
#     list --store-dir ./.tmp/image_embeddings/object
#     search --store-dir ./.tmp/image_embeddings/object --image-embedding-model object --top-n 10 ./query.jpg
#     validate --image-embedding-model object --samples-per-class 20    # download CIFAR-10, measure kNN retrieval accuracy
import argparse
import json
import pickle
import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from jaxtyping import UInt8
from kiarina.agi.cost_recorder import cost_recorder_registry
from kiarina.agi.embedding import Embedding, search_embeddings
from kiarina.agi.image_detection_model import (
    crop_align_faces,
    detect_faces,
)
from kiarina.agi.image_embedding_model import embed_image
from kiarina.agi.image_types import ImagePixels
from kiarina.agi.run_context import RunContext
from PIL import Image

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

type SearchResult = tuple[Embedding, float]
ImagePixelBatch: TypeAlias = UInt8[np.ndarray, "images height width rgb"]  # noqa: UP040, F722
type DatasetBundle = tuple[list[ImagePixels], list[int], dict[int, str]]

_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# scikit-learn's figshare mirror of LFW (funneled). Roughly aligned 250x250 JPEGs
# under <root>/<Person_Name>/<Person_Name>_NNNN.jpg.
_LFW_URL = "https://ndownloader.figshare.com/files/5976015"
_CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# --------------------------------------------------
# Utils
# --------------------------------------------------


def _load_pixels(input_file_path: Path) -> ImagePixels:
    image = Image.open(input_file_path).convert("RGB")
    pixels: ImagePixels = np.array(image, dtype=np.uint8)
    return pixels


def _entry_summary(entry: Embedding) -> dict[str, Any]:
    return {
        "id": entry.id,
        "label": entry.metadata.get("label") or entry.id,
        "created_at": entry.created_at.isoformat(),
        "image_embedding_model": entry.metadata.get("image_embedding_model"),
        "kind": entry.kind,
        "space_id": entry.space_id,
        "embedding_dim": len(entry.vector),
        "width": entry.metadata.get("width"),
        "height": entry.metadata.get("height"),
        "source_file": entry.metadata.get("source_file"),
    }


# --------------------------------------------------
# Services
# --------------------------------------------------


class ImageEmbeddingStore:
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


def _ensure_cifar10(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = data_dir / "cifar-10-batches-py"

    if batches_dir.exists():
        return batches_dir

    archive_path = data_dir / "cifar-10-python.tar.gz"

    if not archive_path.exists():
        print(f"Downloading CIFAR-10 to {archive_path} ...")
        urllib.request.urlretrieve(_CIFAR10_URL, archive_path)

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)

    return batches_dir


def _load_cifar10_test(batches_dir: Path) -> tuple[ImagePixelBatch, list[int]]:
    with open(batches_dir / "test_batch", "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    raw = np.asarray(batch[b"data"], dtype=np.uint8)
    labels = [int(label) for label in batch[b"labels"]]
    images = raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NCHW -> NHWC (RGB)
    pixels: ImagePixelBatch = np.ascontiguousarray(images)
    return pixels, labels


def _select_class_ids(
    class_ids: list[int],
    num_classes: int,
    rng: np.random.Generator,
) -> list[int]:
    if num_classes > 0 and len(class_ids) > num_classes:
        chosen = rng.choice(np.asarray(class_ids), size=num_classes, replace=False)
        return sorted(int(c) for c in chosen)

    return sorted(class_ids)


def _dataset_cifar10(
    data_dir: Path,
    *,
    samples_per_class: int,
    num_classes: int,
    seed: int,
    min_images: int,
) -> DatasetBundle:
    batches_dir = _ensure_cifar10(data_dir)
    all_images, all_labels = _load_cifar10_test(batches_dir)

    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(all_labels):
        by_class.setdefault(label, []).append(index)

    class_ids = _select_class_ids(list(by_class), num_classes, rng)

    images: list[ImagePixels] = []
    labels: list[int] = []

    for label in class_ids:
        indices = np.asarray(by_class[label])
        take = min(samples_per_class, len(indices))
        for chosen in rng.choice(indices, size=take, replace=False):
            images.append(all_images[int(chosen)])
            labels.append(label)

    names = dict(enumerate(_CIFAR10_CLASSES))
    return images, labels, names


def _find_lfw_root(data_dir: Path) -> Path | None:
    for candidate in ("lfw_funneled", "lfw-funneled", "lfw-deepfunneled", "lfw"):
        root = data_dir / candidate
        if root.is_dir() and any(child.is_dir() for child in root.iterdir()):
            return root

    return None


def _ensure_lfw(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    if root := _find_lfw_root(data_dir):
        return root

    archive_path = data_dir / "lfw.tgz"

    if not archive_path.exists():
        print(f"Downloading LFW to {archive_path} ...")
        urllib.request.urlretrieve(_LFW_URL, archive_path)

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)

    if root := _find_lfw_root(data_dir):
        return root

    raise ValueError(f"Could not locate the extracted LFW directory under {data_dir}.")


def _dataset_lfw(
    data_dir: Path,
    *,
    samples_per_class: int,
    num_classes: int,
    seed: int,
    min_images: int,
) -> DatasetBundle:
    root_dir = _ensure_lfw(data_dir)
    rng = np.random.default_rng(seed)

    people = sorted(person for person in root_dir.iterdir() if person.is_dir())
    eligible = [
        person for person in people if len(list(person.glob("*.jpg"))) >= max(min_images, 2)
    ]

    if not eligible:
        raise ValueError(f"No LFW identities with >= {max(min_images, 2)} images found.")

    if num_classes > 0 and len(eligible) > num_classes:
        indices = rng.choice(len(eligible), size=num_classes, replace=False)
        eligible = [eligible[i] for i in sorted(indices)]

    images: list[ImagePixels] = []
    labels: list[int] = []
    names: dict[int, str] = {}

    for label, person in enumerate(eligible):
        files = sorted(person.glob("*.jpg"))
        take = min(samples_per_class, len(files))
        for chosen in sorted(rng.choice(len(files), size=take, replace=False)):
            image = Image.open(files[int(chosen)]).convert("RGB")
            pixels: ImagePixels = np.array(image, dtype=np.uint8)
            images.append(pixels)
            labels.append(label)
        names[label] = person.name

    return images, labels, names


_DATASETS: dict[str, tuple[Callable[..., DatasetBundle], bool]] = {
    # name: (loader, align_faces_by_default)
    "cifar10": (_dataset_cifar10, False),
    "lfw": (_dataset_lfw, True),
}


# --------------------------------------------------
# Main
# --------------------------------------------------


class ImageEmbeddingCommand(BaseExtensionCommand):
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
            await self._add(context, ImageEmbeddingStore(Path(options.store_dir)), options)
        elif options.command == "list":
            self._list(ImageEmbeddingStore(Path(options.store_dir)), options)
        elif options.command == "search":
            await self._search(context, ImageEmbeddingStore(Path(options.store_dir)), options)
        elif options.command == "validate":
            await self._validate(context, options)
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {options.command}")

    async def _add(
        self,
        context: ExtensionCommandContext,
        store: ImageEmbeddingStore,
        options: argparse.Namespace,
    ) -> None:
        input_file_path = Path(options.input_file).expanduser()
        pixels = _load_pixels(input_file_path)
        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)
        run_context = RunContext(agent_id="image-embedding-command")

        embedding = await embed_image(
            pixels,
            image_embedding_options={
                "image_embedding_model": options.image_embedding_model,
            },
            cost_recorder=cost_recorder,
            run_context=run_context,
        )

        height, width = pixels.shape[:2]
        label = options.label or (
            f"{input_file_path.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        embedding.metadata = {
            **embedding.metadata,
            "label": label,
            "source_file": str(input_file_path),
            "image_embedding_model": options.image_embedding_model,
            "width": width,
            "height": height,
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

    def _list(self, store: ImageEmbeddingStore, options: argparse.Namespace) -> None:
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
        store: ImageEmbeddingStore,
        options: argparse.Namespace,
    ) -> None:
        input_file_path = Path(options.input_file).expanduser()
        pixels = _load_pixels(input_file_path)
        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)
        run_context = RunContext(agent_id="image-embedding-search-command")

        query = await embed_image(
            pixels,
            image_embedding_options={
                "image_embedding_model": options.image_embedding_model,
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

    async def _validate(
        self,
        context: ExtensionCommandContext,
        options: argparse.Namespace,
    ) -> None:
        loader, align_default = _DATASETS[options.dataset]
        align_faces = align_default if options.align_faces is None else options.align_faces

        images, labels, names = loader(
            Path(options.data_dir),
            samples_per_class=options.samples_per_class,
            num_classes=options.num_classes,
            seed=options.seed,
            min_images=options.min_images,
        )

        cost_recorder = cost_recorder_registry.resolve(context.run_options.cost_recorder)
        run_context = RunContext(agent_id="image-embedding-validate-command")

        if align_faces:
            images, labels = await self._align_faces(images, labels, run_context)

        if len(images) < 2:
            raise ValueError("Need at least 2 samples to evaluate.")

        print(
            f"Embedding {len(images)} images from dataset={options.dataset} "
            f"with image-embedding-model={options.image_embedding_model or 'default'} "
            f"(align_faces={align_faces}) ..."
        )

        vectors: list[np.ndarray] = []

        for count, pixels in enumerate(images, 1):
            embedding = await embed_image(
                pixels,
                image_embedding_options={
                    "image_embedding_model": options.image_embedding_model,
                },
                cost_recorder=cost_recorder,
                run_context=run_context,
            )
            vectors.append(embedding.to_numpy())

            if count % 50 == 0 or count == len(images):
                print(f"  embedded {count}/{len(images)}")

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

    async def _align_faces(
        self,
        images: list[ImagePixels],
        labels: list[int],
        run_context: RunContext,
    ) -> tuple[list[ImagePixels], list[int]]:
        aligned_images: list[ImagePixels] = []
        aligned_labels: list[int] = []
        skipped = 0

        for pixels, label in zip(images, labels, strict=True):
            faces = await detect_faces(pixels, run_context=run_context)

            if not faces:
                skipped += 1
                continue

            best = max(faces, key=lambda face: face.score)
            crops = crop_align_faces(pixels, [best], output_size=112)

            if not crops:
                skipped += 1
                continue

            aligned_images.append(crops[0])
            aligned_labels.append(label)

        print(f"Aligned {len(aligned_images)} faces (skipped {skipped} with no face).")
        return aligned_images, aligned_labels


def _evaluate_knn(vectors: np.ndarray, labels: np.ndarray, *, k: int) -> dict[str, Any]:
    # Normalize so cosine similarity reduces to a dot product.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = vectors / norms
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)

    neighbors = np.argsort(-similarity, axis=1)[:, :k]
    correct = 0
    per_class_total: dict[int, int] = {}
    per_class_correct: dict[int, int] = {}

    for i, label in enumerate(labels):
        neighbor_labels = labels[neighbors[i]]
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


def _parse_args(
    args: Sequence[str],
    *,
    default_model: str | None,
    default_min_score: float | None,
    prog: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Create, list, search, and validate image embeddings.",
    )
    common_parser = argparse.ArgumentParser(add_help=False)

    # fmt: off
    common_parser.add_argument("--image-embedding-model", default=default_model, help="Image embedding model specifier (e.g. seen, object, face).")
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
    # fmt: off
    add_parser.add_argument("input_file")
    add_parser.add_argument("--label", help="Entry label. Defaults to filename + timestamp.")
    # fmt: on

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
    # fmt: off
    search_parser.add_argument("input_file")
    search_parser.add_argument("--top-n", type=int, default=10, help="Number of results to print.")
    search_parser.add_argument("--include-all-spaces", action="store_true", help="Search across embedding spaces. By default, only exact space_id matches are searched.")
    search_parser.add_argument("--min-score", type=float, default=default_min_score, help="Minimum cosine similarity score.")
    # fmt: on

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[common_parser],
        help="Download a labeled dataset and measure kNN retrieval accuracy.",
    )
    # fmt: off
    validate_parser.add_argument("--dataset", choices=sorted(_DATASETS), default="cifar10", help="Dataset to validate against (cifar10: objects, lfw: faces).")
    validate_parser.add_argument("--data-dir", default=None, help="Directory to download/extract the dataset into. Defaults to .tmp/datasets/<dataset>.")
    validate_parser.add_argument("--samples-per-class", type=int, default=20, help="Max images to sample per class.")
    validate_parser.add_argument("--num-classes", type=int, default=0, help="Number of classes to sample (0 = all). Recommended for lfw.")
    validate_parser.add_argument("--min-images", type=int, default=2, help="Skip classes with fewer than this many images (datasets with variable counts, e.g. lfw).")
    validate_parser.add_argument("--k", type=int, default=1, help="Number of neighbors for kNN classification.")
    validate_parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    validate_parser.add_argument("--align-faces", dest="align_faces", action=argparse.BooleanOptionalAction, default=None, help="Detect and ArcFace-align a face per image before embedding. Defaults on for lfw, off for cifar10.")
    # fmt: on

    options = parser.parse_args(list(args))

    if options.command == "validate" and options.data_dir is None:
        options.data_dir = f".tmp/datasets/{options.dataset}"

    return options


extension_command_registry.register("image-embedding", ImageEmbeddingCommand)
