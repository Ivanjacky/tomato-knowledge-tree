from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cosine_baseline import _load_hf_encoder
from .schema import EmbeddingBundle, load_embeddings, load_tables, save_embeddings

MAJOR_TEXT = {
    1: "BBCH principal stage 1: leaf development in tomato.",
    2: "BBCH principal stage 2: formation of side shoots in tomato.",
    5: "BBCH principal stage 5: emergence of tomato inflorescences.",
    6: "BBCH principal stage 6: flowering in tomato.",
    7: "BBCH principal stage 7: fruit development in tomato.",
    8: "BBCH principal stage 8: fruit ripening in tomato.",
}


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Embedding matrix contains an invalid norm")
    return values / norms


def _load_descriptions(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): str(value).strip() for key, value in raw.items()}


def _fine_prompt(stage: int, description: str, variant: str) -> str:
    if variant == "official":
        return f"BBCH {stage:02d}: {description}"
    if variant == "contextual":
        return (
            f"A tomato plant at BBCH stage {stage:02d}, characterized by "
            f"{description.rstrip('.')} ."
        ).replace("  ", " ")
    if variant == "concise":
        return f"Tomato BBCH stage {stage:02d}."
    raise ValueError(f"Unsupported prompt variant: {variant}")


def _concept_prompts(
    nodes: pd.DataFrame,
    descriptions: dict[int, str],
    variant: str,
) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for row in nodes.itertuples(index=False):
        node_id = str(row.node_id)
        level = int(row.level)
        if level == 0:
            prompts[node_id] = "Tomato phenological knowledge."
        elif level == 1:
            prompts[node_id] = "Tomato growth and phenological development knowledge."
        elif level == 2:
            major = int(node_id.removeprefix("bbch_major_"))
            prompts[node_id] = MAJOR_TEXT[major]
        elif level == 3:
            stage = int(node_id.removeprefix("bbch_"))
            prompts[node_id] = _fine_prompt(stage, descriptions[stage], variant)
    return prompts


def _encode_qwen_texts(
    texts: list[str],
    *,
    model_path: Path,
    vendor_root: Path,
    cache_dir: Path,
) -> np.ndarray:
    qwen_repo = vendor_root.resolve()
    sys.path.insert(0, str(qwen_repo))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    cache_dir.mkdir(parents=True, exist_ok=True)
    instruction = "Represent tomato phenological knowledge for hierarchical parent assignment."
    vectors: list[np.ndarray | None] = [None] * len(texts)
    missing: list[int] = []
    for index, text in enumerate(texts):
        key = hashlib.sha256(f"{instruction}\n{text}".encode()).hexdigest()
        path = cache_dir / f"{key}.npy"
        if path.exists():
            vector = np.load(path, allow_pickle=False).astype(np.float32)
            if vector.ndim == 1 and np.isfinite(vector).all():
                vectors[index] = vector
                continue
        missing.append(index)
    if missing:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        embedder = Qwen3VLEmbedder(
            model_name_or_path=str(model_path),
            max_length=512,
            min_pixels=64 * 64,
            max_pixels=512 * 512,
            torch_dtype=dtype,
        )
        for start in range(0, len(missing), 8):
            selected = missing[start : start + 8]
            jobs = [{"text": texts[index], "instruction": instruction} for index in selected]
            with torch.inference_mode():
                encoded = embedder.process(jobs).detach().float().cpu().numpy()
            for index, vector in zip(selected, encoded, strict=True):
                key = hashlib.sha256(
                    f"{instruction}\n{texts[index]}".encode()
                ).hexdigest()
                np.save(cache_dir / f"{key}.npy", vector.astype(np.float32))
                vectors[index] = vector.astype(np.float32)
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if any(vector is None for vector in vectors):
        raise RuntimeError("Qwen text embedding cache is incomplete")
    return _normalize(np.stack([vector for vector in vectors if vector is not None]))


def _cached_image_features(cache_dir: Path) -> tuple[list[str], np.ndarray]:
    node_ids: list[str] = []
    features: list[np.ndarray] = []
    paths = sorted(cache_dir.glob("image_chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No cached image feature chunks found in {cache_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            node_ids.extend(str(value) for value in data["node_ids"].tolist())
            features.append(data["features"].astype(np.float32))
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Cached image feature chunks contain duplicate node IDs")
    return node_ids, _normalize(np.concatenate(features, axis=0))


def build_bundle(
    *,
    encoder: str,
    data_dir: Path,
    output_path: Path,
    descriptions_path: Path,
    prompt_variant: str,
    original_qwen_embeddings: Path,
    model_path: Path | None,
    image_cache_dir: Path | None,
    qwen_vendor_root: Path,
    text_cache_dir: Path,
) -> dict[str, object]:
    tables = load_tables(data_dir)
    nodes = tables.nodes.reset_index(drop=True)
    node_ids = nodes["node_id"].astype(str).tolist()
    descriptions = _load_descriptions(descriptions_path)
    prompts = _concept_prompts(nodes, descriptions, prompt_variant)
    prompt_ids = list(prompts)
    prompt_texts = [prompts[node_id] for node_id in prompt_ids]

    if encoder == "qwen":
        original = load_embeddings(original_qwen_embeddings, node_ids)
        text = original.text.copy()
        image = original.image.copy()
        text_mask = original.text_mask.copy()
        image_mask = original.image_mask.copy()
        encoded = _encode_qwen_texts(
            prompt_texts,
            model_path=model_path or Path("models/Qwen3-VL-Embedding-2B"),
            vendor_root=qwen_vendor_root,
            cache_dir=text_cache_dir / f"qwen_{prompt_variant}",
        )
        row_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
        for prompt_id, vector in zip(prompt_ids, encoded, strict=True):
            row = row_by_id[prompt_id]
            text[row] = vector
            text_mask[row] = 1.0
        model_name = "Qwen3-VL-Embedding-2B"
    elif encoder in {"clip", "siglip2"}:
        if model_path is None or image_cache_dir is None:
            raise ValueError("HF encoder bundles require model_path and image_cache_dir")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, encode_texts, _ = _load_hf_encoder(model_path, device)
        encoded = _normalize(encode_texts(prompt_texts))
        cached_ids, cached_features = _cached_image_features(image_cache_dir)
        dim = int(encoded.shape[1])
        text = np.zeros((len(node_ids), dim), dtype=np.float32)
        image = np.zeros_like(text)
        text_mask = np.zeros(len(node_ids), dtype=np.float32)
        image_mask = np.zeros(len(node_ids), dtype=np.float32)
        row_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
        for prompt_id, vector in zip(prompt_ids, encoded, strict=True):
            row = row_by_id[prompt_id]
            text[row] = vector
            text_mask[row] = 1.0
        for item_id, vector in zip(cached_ids, cached_features, strict=True):
            row = row_by_id[item_id]
            image[row] = vector
            image_mask[row] = 1.0
        model_name = model_path.name
        del encode_texts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")

    if np.any((text_mask + image_mask) == 0):
        missing = [
            node_ids[index]
            for index in np.flatnonzero((text_mask + image_mask) == 0)[:10]
        ]
        raise ValueError(f"Generated bundle contains nodes without a modality: {missing}")
    bundle = EmbeddingBundle(
        node_ids=node_ids,
        text=text,
        image=image,
        text_mask=text_mask,
        image_mask=image_mask,
    )
    save_embeddings(output_path, bundle)
    prompt_frame = pd.DataFrame(
        [{"node_id": node_id, "text": prompts[node_id]} for node_id in prompt_ids]
    )
    prompt_path = output_path.with_name(f"{output_path.stem}_prompts.csv")
    prompt_frame.to_csv(prompt_path, index=False, encoding="utf-8-sig")
    report = {
        "encoder": encoder,
        "model": model_name,
        "prompt_variant": prompt_variant,
        "embedding_dim": bundle.dim,
        "node_count": len(node_ids),
        "text_node_count": int(text_mask.sum()),
        "image_node_count": int(image_mask.sum()),
        "output": str(output_path.resolve()),
        "prompts": str(prompt_path.resolve()),
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reviewer-controlled embedding bundles")
    parser.add_argument("--encoder", choices=["qwen", "clip", "siglip2"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument(
        "--prompt-variant",
        choices=["official", "contextual", "concise"],
        default="official",
    )
    parser.add_argument("--original-qwen-embeddings", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--image-cache-dir", type=Path)
    parser.add_argument("--qwen-vendor-root", type=Path, default=Path("vendor/qwen_embed"))
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=Path("data/processed/reviewer_embeddings/text_cache"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_bundle(
        encoder=args.encoder,
        data_dir=args.data_dir,
        output_path=args.output,
        descriptions_path=args.descriptions,
        prompt_variant=args.prompt_variant,
        original_qwen_embeddings=args.original_qwen_embeddings,
        model_path=args.model_path,
        image_cache_dir=args.image_cache_dir,
        qwen_vendor_root=args.qwen_vendor_root,
        text_cache_dir=args.text_cache_dir,
    )


if __name__ == "__main__":
    main()
