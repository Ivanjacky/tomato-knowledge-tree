from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from .evaluate import evaluate
from .schema import load_embeddings, load_tables
from .utils import write_json

PROMPT_TEMPLATES = {
    "description": "{description}",
    "bbch_description": "BBCH {stage}: {description}",
    "photo": "a photo of a tomato plant at BBCH stage {stage}: {description}.",
}
SPLITS = ("train", "val", "test")


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Feature matrix contains a zero, NaN, or infinite norm")
    return (features / norms).astype(np.float32)


def _load_descriptions(path: Path, parent_ids: list[str]) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    descriptions = {f"bbch_{int(code):02d}": str(value) for code, value in raw.items()}
    missing = sorted(set(parent_ids) - set(descriptions))
    extra = sorted(set(descriptions) - set(parent_ids))
    if missing or extra:
        raise ValueError(
            f"Description codes do not match candidate parents; missing={missing}, extra={extra}"
        )
    return descriptions


def _prompt_texts(
    parent_ids: list[str], descriptions: dict[str, str], template_name: str
) -> list[str]:
    template = PROMPT_TEMPLATES[template_name]
    return [
        template.format(stage=parent_id.removeprefix("bbch_"), description=descriptions[parent_id])
        for parent_id in parent_ids
    ]


def _rank_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    target_scores = scores[np.arange(len(scores)), targets]
    ranks = 1 + np.sum(scores > target_scores[:, None], axis=1)
    logits = torch.from_numpy(scores.astype(np.float32))
    target_tensor = torch.from_numpy(targets.astype(np.int64))
    return {
        "loss": float(F.cross_entropy(logits, target_tensor)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "count": len(ranks),
    }


def _save_image_chunks(
    *,
    item_ids: list[str],
    image_paths: list[Path],
    cache_dir: Path,
    chunk_size: int,
    encode_images: Callable[[list[Image.Image]], np.ndarray],
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    starts = range(0, len(item_ids), chunk_size)
    for chunk_index, start in enumerate(tqdm(starts, desc="image feature chunks")):
        stop = min(start + chunk_size, len(item_ids))
        expected_ids = item_ids[start:stop]
        cache_path = cache_dir / f"image_chunk_{chunk_index:05d}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                stored_ids = [str(value) for value in cached["node_ids"].tolist()]
                features = cached["features"].astype(np.float32)
            if stored_ids != expected_ids or features.shape[0] != len(expected_ids):
                raise ValueError(f"Cache chunk does not match this dataset: {cache_path}")
        else:
            images: list[Image.Image] = []
            try:
                for path in image_paths[start:stop]:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                features = _normalize(encode_images(images))
            finally:
                for image in images:
                    image.close()
            np.savez_compressed(
                cache_path,
                node_ids=np.asarray(expected_ids, dtype=np.str_),
                features=features,
            )
        chunks.append(features)
    return np.concatenate(chunks, axis=0)


def _load_hf_encoder(
    model_path: Path, device: torch.device
) -> tuple[Callable[[list[Image.Image]], np.ndarray], Callable[[list[str]], np.ndarray], int]:
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        model = model.to(dtype=dtype)

    def pooled(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        raise TypeError(f"Unsupported feature output type: {type(output).__name__}")

    def encode_images(images: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(device=device, dtype=dtype)
            if value.is_floating_point()
            else value.to(device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output = model.get_image_features(**inputs)
        return pooled(output).float().cpu().numpy()

    def encode_texts(texts: list[str]) -> np.ndarray:
        inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.get_text_features(**inputs)
        return pooled(output).float().cpu().numpy()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return encode_images, encode_texts, parameter_count


def _qwen_features(
    data_dir: Path, node_ids: list[str], parent_ids: list[str], item_ids: list[str]
) -> tuple[np.ndarray, Callable[[list[str]], np.ndarray], int]:
    bundle = load_embeddings(data_dir / "embeddings.npz", node_ids)
    row_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    image_features = _normalize(bundle.image[[row_by_id[node_id] for node_id in item_ids]])
    parent_features = _normalize(bundle.text[[row_by_id[node_id] for node_id in parent_ids]])

    def encode_texts(_: list[str]) -> np.ndarray:
        return parent_features

    return image_features, encode_texts, 2_000_000_000


def _truth_fingerprint(predictions: pd.DataFrame) -> str:
    truth = (
        predictions[["child_id", "split", "true_parent_id"]]
        .sort_values("child_id")
        .reset_index(drop=True)
    )
    payload = truth.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def run_cosine_baseline(
    *,
    data_dir: Path,
    output_dir: Path,
    backend: str,
    descriptions_path: Path,
    model_path: Path | None = None,
    model_id: str | None = None,
    device_name: str | None = None,
    image_batch_size: int = 64,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    tables = load_tables(data_dir)
    nodes = tables.nodes.reset_index(drop=True)
    node_ids = nodes["node_id"].astype(str).tolist()
    parent_ids = sorted(
        nodes.loc[
            (nodes["level"].astype(int) == 3)
            & (nodes["is_candidate_parent"].astype(int) == 1),
            "node_id",
        ].astype(str)
    )
    edge_frame = tables.edges[tables.edges["is_fixed"].astype(int) == 0].copy()
    edge_frame = edge_frame.sort_values(["split", "child_id"]).reset_index(drop=True)
    item_ids = edge_frame["child_id"].astype(str).tolist()
    parent_position = {node_id: index for index, node_id in enumerate(parent_ids)}
    targets = np.asarray(
        [parent_position[str(parent_id)] for parent_id in edge_frame["parent_id"]],
        dtype=np.int64,
    )
    descriptions = _load_descriptions(descriptions_path, parent_ids)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if backend == "qwen":
        image_features, encode_texts, parameter_count = _qwen_features(
            data_dir, node_ids, parent_ids, item_ids
        )
        resolved_model = "Qwen3-VL-Embedding-2B (existing frozen embeddings)"
    elif backend == "hf":
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Local Hugging Face model directory not found: {model_path}")
        encode_images, encode_texts, parameter_count = _load_hf_encoder(model_path, device)
        media_images = tables.media[tables.media["modality"].astype(str) == "image"]
        path_by_id = dict(
            zip(media_images["node_id"].astype(str), media_images["path"].astype(str))
        )
        missing_paths = [node_id for node_id in item_ids if node_id not in path_by_id]
        if missing_paths:
            raise ValueError(f"Image paths missing for item nodes: {missing_paths[:5]}")
        image_paths = [Path(path_by_id[node_id]) for node_id in item_ids]
        absent_files = [str(path) for path in image_paths if not path.is_file()]
        if absent_files:
            raise FileNotFoundError(f"Image media files are missing: {absent_files[:5]}")
        image_features = _save_image_chunks(
            item_ids=item_ids,
            image_paths=image_paths,
            cache_dir=cache_dir or (output_dir / "feature_cache"),
            chunk_size=image_batch_size,
            encode_images=encode_images,
        )
        resolved_model = model_id or str(model_path)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    image_features = _normalize(image_features)
    validation_mask = edge_frame["split"].astype(str).to_numpy() == "val"
    prompt_reports: dict[str, dict[str, float | int]] = {}
    text_features_by_prompt: dict[str, np.ndarray] = {}
    prompt_names = ["cached_node_description"] if backend == "qwen" else list(PROMPT_TEMPLATES)
    for prompt_name in prompt_names:
        texts = (
            parent_ids
            if prompt_name == "cached_node_description"
            else _prompt_texts(parent_ids, descriptions, prompt_name)
        )
        text_features = _normalize(encode_texts(texts))
        if text_features.shape[0] != len(parent_ids):
            raise ValueError("Text encoder did not return one feature per parent prompt")
        text_features_by_prompt[prompt_name] = text_features
        scores = image_features[validation_mask] @ text_features.T
        prompt_reports[prompt_name] = _rank_metrics(scores, targets[validation_mask])
    selected_prompt = max(
        prompt_names,
        key=lambda name: (float(prompt_reports[name]["mrr"]), -prompt_names.index(name)),
    )
    selected_text = text_features_by_prompt[selected_prompt]
    all_scores = image_features @ selected_text.T

    fixed_parent = {
        str(row.child_id): str(row.parent_id)
        for row in tables.edges[tables.edges["is_fixed"].astype(int) == 1].itertuples(index=False)
    }
    predictions: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float | int]] = {}
    for split in SPLITS:
        split_mask = edge_frame["split"].astype(str).to_numpy() == split
        split_scores = all_scores[split_mask]
        split_targets = targets[split_mask]
        split_rows = edge_frame.loc[split_mask].reset_index(drop=True)
        rank_metrics = _rank_metrics(split_scores, split_targets)
        target_scores = split_scores[np.arange(len(split_scores)), split_targets]
        ranks = 1 + np.sum(split_scores > target_scores[:, None], axis=1)
        predicted_positions = np.argmax(split_scores, axis=1)
        probabilities = torch.softmax(torch.from_numpy(split_scores), dim=1).numpy()
        upper_correct: list[int] = []
        for row_index, row in enumerate(split_rows.itertuples(index=False)):
            predicted_parent = parent_ids[int(predicted_positions[row_index])]
            true_parent = str(row.parent_id)
            same_upper = int(fixed_parent[predicted_parent] == fixed_parent[true_parent])
            upper_correct.append(same_upper)
            predictions.append(
                {
                    "split": split,
                    "child_id": str(row.child_id),
                    "true_parent_id": true_parent,
                    "predicted_parent_id": predicted_parent,
                    "rank": int(ranks[row_index]),
                    "confidence": float(probabilities[row_index, predicted_positions[row_index]]),
                    "correct": int(predicted_parent == true_parent),
                    "same_upper_parent": same_upper,
                }
            )
        rank_metrics["upper_parent_accuracy"] = float(np.mean(upper_correct))
        metrics[split] = rank_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(predictions).sort_values(["split", "child_id"])
    prediction_frame.to_csv(
        output_dir / "predicted_edges.csv", index=False, encoding="utf-8-sig"
    )
    fixed_edges = tables.edges[tables.edges["is_fixed"].astype(int) == 1][
        ["parent_id", "child_id", "relation"]
    ].copy()
    predicted_edges = prediction_frame[["predicted_parent_id", "child_id"]].rename(
        columns={"predicted_parent_id": "parent_id"}
    )
    predicted_edges["relation"] = "predicted_parent"
    pd.concat([fixed_edges, predicted_edges], ignore_index=True).to_csv(
        output_dir / "constructed_tree_edges.csv", index=False, encoding="utf-8-sig"
    )
    write_json(output_dir / "metrics.json", metrics)
    supplementary = evaluate(data_dir, output_dir)
    run_info = {
        "method": f"{backend}_frozen_cosine_parent_retrieval",
        "model": resolved_model,
        "model_path": str(model_path.resolve()) if model_path is not None else None,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": 0,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "image_count": len(item_ids),
        "candidate_parent_count": len(parent_ids),
        "image_batch_size": int(image_batch_size),
        "selected_prompt": selected_prompt,
        "prompt_templates": PROMPT_TEMPLATES if backend == "hf" else {},
        "validation_prompt_metrics": prompt_reports,
        "description_source": (
            "cached nodes.csv descriptions embedded by Qwen3-VL"
            if backend == "qwen"
            else "TomatoMAP FiftyOne field bbch_description"
        ),
        "query_modalities": ["image"],
        "target_text_or_metadata_in_query": False,
        "selection_split": "val",
        "ground_truth_sha256": _truth_fingerprint(prediction_frame),
        "elapsed_seconds": float(time.time() - started),
        "structure_valid": bool(supplementary["structure"]["is_valid_tree"]),
    }
    write_json(output_dir / "run_info.json", run_info)
    print(json.dumps({"run_info": run_info, "test": metrics["test"]}, indent=2))
    return {"run_info": run_info, "metrics": metrics, "supplementary": supplementary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a frozen cosine parent-retrieval baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/tomatomap_kt5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["qwen", "hf"], required=True)
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=Path("configs/bbch_descriptions_en.json"),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.image_batch_size <= 0:
        raise ValueError("--image-batch-size must be positive")
    run_cosine_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        descriptions_path=args.descriptions,
        model_path=args.model_path,
        model_id=args.model_id,
        device_name=args.device,
        image_batch_size=args.image_batch_size,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
