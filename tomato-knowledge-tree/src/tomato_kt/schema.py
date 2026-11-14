from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NODE_COLUMNS = {
    "node_id",
    "name",
    "description",
    "node_kind",
    "level",
    "split",
    "is_candidate_parent",
}
EDGE_COLUMNS = {"parent_id", "child_id", "relation", "split", "is_fixed"}
MEDIA_COLUMNS = {"media_id", "node_id", "modality", "content", "path", "split"}


@dataclass(frozen=True)
class KnowledgeTables:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    media: pd.DataFrame


@dataclass(frozen=True)
class EmbeddingBundle:
    node_ids: list[str]
    text: np.ndarray
    image: np.ndarray
    text_mask: np.ndarray
    image_mask: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.text.shape[1])


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def load_tables(data_dir: str | Path) -> KnowledgeTables:
    root = Path(data_dir)
    nodes_path = root / "nodes.csv"
    edges_path = root / "edges.csv"
    media_path = root / "media.csv"

    nodes = pd.read_csv(nodes_path, dtype={"node_id": str})
    edges = pd.read_csv(edges_path, dtype={"parent_id": str, "child_id": str})
    media = pd.read_csv(media_path, dtype={"node_id": str, "media_id": str})
    _require_columns(nodes, NODE_COLUMNS, nodes_path)
    _require_columns(edges, EDGE_COLUMNS, edges_path)
    _require_columns(media, MEDIA_COLUMNS, media_path)

    if nodes["node_id"].duplicated().any():
        duplicates = nodes.loc[nodes["node_id"].duplicated(), "node_id"].tolist()
        raise ValueError(f"Duplicate node_id values: {duplicates[:5]}")

    known = set(nodes["node_id"])
    dangling = (set(edges["parent_id"]) | set(edges["child_id"])) - known
    if dangling:
        raise ValueError(f"Edges contain unknown nodes: {sorted(dangling)[:5]}")

    unknown_media_nodes = set(media["node_id"]) - known
    if unknown_media_nodes:
        raise ValueError(f"Media contain unknown nodes: {sorted(unknown_media_nodes)[:5]}")

    nodes["level"] = nodes["level"].astype(int)
    nodes["is_candidate_parent"] = nodes["is_candidate_parent"].astype(int)
    edges["is_fixed"] = edges["is_fixed"].astype(int)
    return KnowledgeTables(nodes=nodes, edges=edges, media=media)


def load_embeddings(path: str | Path, expected_node_ids: list[str]) -> EmbeddingBundle:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {"node_ids", "text_embeddings", "image_embeddings", "text_mask", "image_mask"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{source} is missing arrays: {', '.join(missing)}")

        stored_ids = [str(value) for value in data["node_ids"].tolist()]
        text = data["text_embeddings"].astype(np.float32)
        image = data["image_embeddings"].astype(np.float32)
        text_mask = data["text_mask"].astype(np.float32)
        image_mask = data["image_mask"].astype(np.float32)

    if text.ndim != 2 or image.ndim != 2 or text.shape != image.shape:
        raise ValueError("Text and image embeddings must be matching 2-D arrays")
    if len(stored_ids) != text.shape[0]:
        raise ValueError("node_ids length does not match embedding rows")
    if len(set(stored_ids)) != len(stored_ids):
        raise ValueError("Embedding file contains duplicate node IDs")

    row_by_id = {node_id: row for row, node_id in enumerate(stored_ids)}
    absent = [node_id for node_id in expected_node_ids if node_id not in row_by_id]
    if absent:
        raise ValueError(f"Embeddings are missing nodes: {absent[:5]}")
    order = np.asarray([row_by_id[node_id] for node_id in expected_node_ids], dtype=np.int64)

    return EmbeddingBundle(
        node_ids=expected_node_ids,
        text=text[order],
        image=image[order],
        text_mask=text_mask[order],
        image_mask=image_mask[order],
    )


def save_embeddings(path: str | Path, bundle: EmbeddingBundle) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        node_ids=np.asarray(bundle.node_ids, dtype=np.str_),
        text_embeddings=bundle.text.astype(np.float32),
        image_embeddings=bundle.image.astype(np.float32),
        text_mask=bundle.text_mask.astype(np.float32),
        image_mask=bundle.image_mask.astype(np.float32),
    )
