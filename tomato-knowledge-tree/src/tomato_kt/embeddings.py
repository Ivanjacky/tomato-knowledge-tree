from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tomato_kt.schema import EmbeddingBundle, load_tables, save_embeddings


def _hash_vector(value: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _encode_batches(
    values: list[dict], encode: Callable[[list[dict]], np.ndarray], batch_size: int, label: str
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for start in tqdm(range(0, len(values), batch_size), desc=label):
        batch = values[start : start + batch_size]
        if batch:
            outputs.extend(encode(batch))
    return outputs


def _cache_key(value: dict) -> str:
    parts: list[str] = []
    for key in sorted(value):
        item = str(value[key])
        if key == "image":
            image = Path(item)
            if image.is_file():
                stat = image.stat()
                item = f"{image.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        parts.append(f"{key}={item}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _encode_batches_cached(
    values: list[dict],
    encode: Callable[[list[dict]], np.ndarray],
    batch_size: int,
    label: str,
    cache_dir: Path,
) -> list[np.ndarray]:
    """Encode jobs with one vector per file so interrupted runs can resume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[np.ndarray | None] = [None] * len(values)
    missing: list[tuple[int, Path]] = []
    for index, value in enumerate(values):
        cache_path = cache_dir / f"{_cache_key(value)}.npy"
        if cache_path.is_file():
            try:
                vector = np.load(cache_path, allow_pickle=False).astype(np.float32)
                if vector.ndim == 1 and np.isfinite(vector).all():
                    outputs[index] = vector
                    continue
            except (OSError, ValueError):
                pass
        missing.append((index, cache_path))

    if len(missing) != len(values):
        print(f"{label}: reused {len(values) - len(missing)}/{len(values)} cached vectors")
    progress = tqdm(total=len(values), initial=len(values) - len(missing), desc=label)
    for start in range(0, len(missing), batch_size):
        selected = missing[start : start + batch_size]
        batch = [values[index] for index, _ in selected]
        vectors = np.asarray(encode(batch), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(selected):
            raise ValueError(
                f"Embedding backend returned shape {vectors.shape} for a batch of {len(selected)}"
            )
        if not np.isfinite(vectors).all():
            raise FloatingPointError("Embedding backend returned NaN or infinity")
        for (index, cache_path), vector in zip(selected, vectors, strict=True):
            temporary = cache_path.with_suffix(".tmp.npy")
            with temporary.open("wb") as stream:
                np.save(stream, vector.astype(np.float32), allow_pickle=False)
            os.replace(temporary, cache_path)
            outputs[index] = vector
        progress.update(len(selected))
    progress.close()
    if any(vector is None for vector in outputs):
        raise RuntimeError("Embedding cache is incomplete")
    return [vector for vector in outputs if vector is not None]


def build_hash_embeddings(data_dir: Path, output: Path, dim: int) -> None:
    tables = load_tables(data_dir)
    text_by_node: dict[str, list[str]] = defaultdict(list)
    image_by_node: dict[str, list[str]] = defaultdict(list)

    for row in tables.nodes.itertuples(index=False):
        description = str(row.description).strip()
        if description and description.lower() != "nan":
            text_by_node[str(row.node_id)].append(description)

    for row in tables.media.itertuples(index=False):
        node_id = str(row.node_id)
        modality = str(row.modality)
        content = str(row.content).strip()
        path = str(row.path).strip()
        if modality in {"text", "text_image"} and content and content.lower() != "nan":
            text_by_node[node_id].append(content)
        if modality in {"image", "text_image"} and path and path.lower() != "nan":
            image_by_node[node_id].append(path)

    node_ids = tables.nodes["node_id"].astype(str).tolist()
    text = np.zeros((len(node_ids), dim), dtype=np.float32)
    image = np.zeros_like(text)
    text_mask = np.zeros(len(node_ids), dtype=np.float32)
    image_mask = np.zeros(len(node_ids), dtype=np.float32)
    for index, node_id in enumerate(node_ids):
        if text_by_node[node_id]:
            text[index] = np.mean(
                [_hash_vector(value, dim) for value in text_by_node[node_id]], axis=0
            )
            text_mask[index] = 1.0
        if image_by_node[node_id]:
            image[index] = np.mean(
                [_hash_vector(str(Path(value).resolve()), dim) for value in image_by_node[node_id]],
                axis=0,
            )
            image_mask[index] = 1.0

    save_embeddings(
        output,
        EmbeddingBundle(
            node_ids=node_ids,
            text=text,
            image=image,
            text_mask=text_mask,
            image_mask=image_mask,
        ),
    )


def build_qwen_embeddings(
    data_dir: Path,
    output: Path,
    model_path: str,
    vendor_root: Path,
    batch_size: int,
    max_pixels: int,
    cache_dir: Path | None = None,
) -> None:
    qwen_repo = vendor_root.resolve()
    if not (qwen_repo / "src" / "models" / "qwen3_vl_embedding.py").exists():
        raise FileNotFoundError(
            f"Official Qwen embedding source was not found at {qwen_repo}. "
            "Restore the Qwen source from the release package."
        )
    instruction = "为番茄农业知识树构建生成可比较的语义向量。"

    tables = load_tables(data_dir)
    text_by_node: dict[str, list[str]] = defaultdict(list)
    image_by_node: dict[str, list[str]] = defaultdict(list)
    for row in tables.nodes.itertuples(index=False):
        value = str(row.description).strip()
        if value and value.lower() != "nan":
            text_by_node[str(row.node_id)].append(value)
    for row in tables.media.itertuples(index=False):
        node_id = str(row.node_id)
        modality = str(row.modality)
        content = str(row.content).strip()
        path = str(row.path).strip()
        if modality in {"text", "text_image"} and content and content.lower() != "nan":
            text_by_node[node_id].append(content)
        if modality in {"image", "text_image"} and path and path.lower() != "nan":
            if not Path(path).is_file():
                raise FileNotFoundError(f"Image does not exist: {path}")
            image_by_node[node_id].append(path)

    text_jobs: list[dict] = []
    text_owners: list[list[str]] = []
    image_jobs: list[dict] = []
    image_owners: list[list[str]] = []
    text_job_by_value: dict[str, int] = {}
    image_job_by_value: dict[str, int] = {}
    for node_id, values in text_by_node.items():
        for value in dict.fromkeys(values):
            job_index = text_job_by_value.get(value)
            if job_index is None:
                job_index = len(text_jobs)
                text_job_by_value[value] = job_index
                text_jobs.append({"text": value, "instruction": instruction})
                text_owners.append([])
            text_owners[job_index].append(node_id)
    for node_id, values in image_by_node.items():
        for value in dict.fromkeys(values):
            normalized = str(Path(value).resolve())
            job_index = image_job_by_value.get(normalized)
            if job_index is None:
                job_index = len(image_jobs)
                image_job_by_value[normalized] = job_index
                image_jobs.append({"image": normalized, "instruction": instruction})
                image_owners.append([])
            image_owners[job_index].append(node_id)

    import torch

    sys.path.insert(0, str(qwen_repo))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        max_length=512,
        min_pixels=64 * 64,
        max_pixels=max_pixels,
        torch_dtype=dtype,
    )

    def encode(batch: list[dict]) -> np.ndarray:
        with torch.inference_mode():
            result = embedder.process(batch)
        return result.detach().float().cpu().numpy()

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_pixels < 64 * 64:
        raise ValueError("max_pixels must be at least 4096")
    cache_root = cache_dir or output.parent / f"{output.stem}_cache"
    model_config = Path(model_path).resolve() / "config.json"
    if model_config.is_file():
        stat = model_config.stat()
        model_version = f"{model_config}|{stat.st_size}|{stat.st_mtime_ns}"
    else:
        model_version = str(Path(model_path).resolve())
    cache_namespace = hashlib.sha256(
        f"{model_version}|max_pixels={max_pixels}".encode()
    ).hexdigest()[:16]
    cache_root = cache_root / cache_namespace
    print(f"Resumable embedding cache: {cache_root.resolve()}")
    text_values = _encode_batches_cached(
        text_jobs, encode, batch_size, "Encoding text", cache_root / "text"
    )
    image_values = _encode_batches_cached(
        image_jobs, encode, batch_size, "Encoding images", cache_root / "image"
    )
    if not text_values and not image_values:
        raise RuntimeError("No encodable text or images were found")
    dim = int((text_values or image_values)[0].shape[-1])

    text_accumulator: dict[str, list[np.ndarray]] = defaultdict(list)
    image_accumulator: dict[str, list[np.ndarray]] = defaultdict(list)
    for owners, vector in zip(text_owners, text_values, strict=True):
        for node_id in owners:
            text_accumulator[node_id].append(vector)
    for owners, vector in zip(image_owners, image_values, strict=True):
        for node_id in owners:
            image_accumulator[node_id].append(vector)

    node_ids = tables.nodes["node_id"].astype(str).tolist()
    text = np.zeros((len(node_ids), dim), dtype=np.float32)
    image = np.zeros_like(text)
    text_mask = np.zeros(len(node_ids), dtype=np.float32)
    image_mask = np.zeros(len(node_ids), dtype=np.float32)
    for index, node_id in enumerate(node_ids):
        if text_accumulator[node_id]:
            text[index] = np.mean(text_accumulator[node_id], axis=0)
            text_mask[index] = 1.0
        if image_accumulator[node_id]:
            image[index] = np.mean(image_accumulator[node_id], axis=0)
            image_mask[index] = 1.0

    save_embeddings(
        output,
        EmbeddingBundle(
            node_ids=node_ids,
            text=text,
            image=image,
            text_mask=text_mask,
            image_mask=image_mask,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text/image embeddings for knowledge nodes"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["hash", "qwen"], default="hash")
    parser.add_argument("--dim", type=int, default=64, help="Only used by the hash smoke backend")
    parser.add_argument("--model-path", default="models/Qwen3-VL-Embedding-2B")
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/qwen_embed"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Per-item vector cache used to resume interrupted Qwen extraction",
    )
    args = parser.parse_args()
    if args.backend == "hash":
        build_hash_embeddings(args.data_dir, args.output, args.dim)
    else:
        build_qwen_embeddings(
            data_dir=args.data_dir,
            output=args.output,
            model_path=args.model_path,
            vendor_root=args.vendor_root,
            batch_size=args.batch_size,
            max_pixels=args.max_pixels,
            cache_dir=args.cache_dir,
        )
    print(f"Embeddings written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
