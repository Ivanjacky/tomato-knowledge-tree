from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from tomato_kt.model import HierarchicalTreeModel
from tomato_kt.schema import EmbeddingBundle, KnowledgeTables, load_embeddings, load_tables
from tomato_kt.utils import load_yaml, set_seed, write_json


def _tensor_bundle(
    bundle: EmbeddingBundle,
    device: torch.device,
    item_text_rows: np.ndarray | None = None,
    item_text_mode: str = "enabled",
    item_text_shuffle_seed: int = 2026,
    group_image: np.ndarray | None = None,
    image_delta: np.ndarray | None = None,
    levels: np.ndarray | None = None,
    group_index: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    text = bundle.text.copy()
    text_mask = bundle.text_mask.copy()
    if item_text_rows is not None and len(item_text_rows) > 0:
        if item_text_mode == "disabled":
            text_mask[item_text_rows] = 0.0
        elif item_text_mode == "shuffled":
            rng = np.random.default_rng(item_text_shuffle_seed)
            permuted = rng.permutation(item_text_rows)
            text[item_text_rows] = text[permuted]
            text_mask[item_text_rows] = text_mask[permuted]
        elif item_text_mode == "constant":
            available = item_text_rows[text_mask[item_text_rows] > 0]
            if len(available) == 0:
                raise ValueError("Constant item text requires an available text embedding")
            source = int(available[0])
            text[item_text_rows] = text[source]
            text_mask[item_text_rows] = text_mask[source]
        elif item_text_mode != "enabled":
            raise ValueError(f"Unsupported item_text_mode: {item_text_mode}")
    tensors = {
        "text": torch.from_numpy(text).to(device),
        "image": torch.from_numpy(bundle.image).to(device),
        "text_mask": torch.from_numpy(text_mask).to(device),
        "image_mask": torch.from_numpy(bundle.image_mask).to(device),
    }
    if group_image is not None:
        tensors["group_image"] = torch.from_numpy(group_image).to(device)
    if image_delta is not None:
        tensors["image_delta"] = torch.from_numpy(image_delta).to(device)
    if levels is not None:
        tensors["levels"] = torch.from_numpy(levels).to(device)
    if group_index is not None:
        tensors["group_index"] = torch.from_numpy(group_index).to(device)
    return tensors


def _group_context_arrays(
    bundle: EmbeddingBundle,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Group context requires {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"image_id", "plant_id", "capture_date", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest.csv is missing group columns: {sorted(missing)}")
    row_by_id = {node_id: row for row, node_id in enumerate(bundle.node_ids)}
    absent = sorted(set(manifest["image_id"].astype(str)) - set(row_by_id))
    if absent:
        raise ValueError(f"Manifest group nodes are absent from embeddings: {absent[:5]}")
    group_image = bundle.image.copy()
    image_delta = np.zeros_like(bundle.image)
    group_index = np.full(len(bundle.node_ids), -1, dtype=np.int64)
    keys = (
        manifest["split"].astype(str)
        + "|"
        + manifest["plant_id"].astype(str)
        + "|"
        + manifest["capture_date"].astype(str)
    )
    sizes: list[int] = []
    for group_number, positions in enumerate(manifest.groupby(keys, sort=False).groups.values()):
        manifest_rows = np.asarray(list(positions), dtype=np.int64)
        node_rows = np.asarray(
            [row_by_id[str(value)] for value in manifest.iloc[manifest_rows]["image_id"]],
            dtype=np.int64,
        )
        mean = bundle.image[node_rows].mean(axis=0, keepdims=True)
        group_image[node_rows] = mean
        image_delta[node_rows] = np.abs(bundle.image[node_rows] - mean)
        group_index[node_rows] = group_number
        sizes.append(len(node_rows))
    return (
        group_image,
        image_delta,
        group_index,
        {
            "group_count": len(sizes),
            "minimum_group_size": min(sizes),
            "maximum_group_size": max(sizes),
        },
    )


def _apply_group_consensus(
    logits: torch.Tensor,
    group_indices: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("group_consensus_weight must be between 0 and 1")
    if weight == 0.0:
        return logits
    adjusted = logits.clone()
    for group_id in torch.unique(group_indices):
        if int(group_id) < 0:
            continue
        mask = group_indices == group_id
        group_mean = logits[mask].mean(dim=0, keepdim=True)
        adjusted[mask] = (1.0 - weight) * logits[mask] + weight * group_mean
    return adjusted


def _counterfactual_text_rows(group_index: np.ndarray) -> np.ndarray:
    """Map every grouped image to a different pose text from the same capture."""
    source_rows = np.arange(len(group_index), dtype=np.int64)
    for group_id in np.unique(group_index[group_index >= 0]):
        rows = np.flatnonzero(group_index == group_id)
        if len(rows) >= 2:
            source_rows[rows] = np.roll(rows, 1)
    return source_rows


def _classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
) -> torch.Tensor:
    if mode in {"cross_entropy", "class_weighted"}:
        weights = class_weights if mode == "class_weighted" else None
        return F.cross_entropy(logits, targets, weight=weights)
    if mode == "focal":
        per_example = F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            reduction="none",
        )
        target_probability = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        return (((1.0 - target_probability) ** focal_gamma) * per_example).mean()
    raise ValueError(f"Unsupported classification_loss: {mode}")


def _class_weights(
    task: ParentTask,
    train_frame: pd.DataFrame,
    power: float,
    device: torch.device,
) -> torch.Tensor:
    _, targets, _ = task.batch_arrays(train_frame, "previous_level")
    counts = np.bincount(targets, minlength=len(task.parent_pool)).astype(np.float64)
    weights = np.zeros_like(counts)
    seen = counts > 0
    if not np.any(seen):
        raise ValueError("Cannot compute class weights without training targets")
    weights[seen] = np.power(counts[seen].sum() / counts[seen], power)
    weights[seen] /= weights[seen].mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _class_log_prior(
    task: ParentTask,
    train_frame: pd.DataFrame,
    device: torch.device,
) -> torch.Tensor:
    _, targets, _ = task.batch_arrays(train_frame, "previous_level")
    counts = np.bincount(targets, minlength=len(task.parent_pool)).astype(np.float64)
    log_prior = np.zeros_like(counts)
    seen = counts > 0
    log_prior[seen] = np.log(counts[seen] / counts[seen].sum())
    return torch.as_tensor(log_prior, dtype=torch.float32, device=device)


class ParentTask:
    def __init__(self, tables: KnowledgeTables) -> None:
        self.nodes = tables.nodes.reset_index(drop=True)
        self.edges = tables.edges[tables.edges["is_fixed"] == 0].reset_index(drop=True)
        self.node_index = {
            node_id: index for index, node_id in enumerate(self.nodes["node_id"].astype(str))
        }
        self.parent_pool = self.nodes.index[self.nodes["is_candidate_parent"] == 1].to_numpy()
        self.parent_pool_position = {
            int(node_index): position for position, node_index in enumerate(self.parent_pool)
        }
        self.fixed_parent = {
            str(row.child_id): str(row.parent_id)
            for row in tables.edges[tables.edges["is_fixed"] == 1].itertuples(index=False)
        }
        parent_levels = self.nodes.loc[self.parent_pool, "level"].to_numpy(dtype=np.int64)
        self.fine_parent_positions = np.flatnonzero(parent_levels == 3)
        self.macro_parent_positions = np.flatnonzero(parent_levels == 2)
        macro_local = {
            int(parent_position): local
            for local, parent_position in enumerate(self.macro_parent_positions)
        }
        self.candidate_macro_parent_positions = np.full(len(self.parent_pool), -1, dtype=np.int64)
        self.candidate_macro_local_positions = np.full(len(self.parent_pool), -1, dtype=np.int64)
        for parent_position in self.fine_parent_positions:
            node_row = int(self.parent_pool[parent_position])
            node_id = str(self.nodes.iloc[node_row]["node_id"])
            macro_id = self.fixed_parent.get(node_id)
            if macro_id is None:
                raise ValueError(f"Fine parent has no macro ancestor: {node_id}")
            macro_row = self.node_index[macro_id]
            macro_parent_position = self.parent_pool_position[macro_row]
            self.candidate_macro_parent_positions[parent_position] = macro_parent_position
            self.candidate_macro_local_positions[parent_position] = macro_local[
                macro_parent_position
            ]

        for row in self.edges.itertuples(index=False):
            true_parent_index = self.node_index[str(row.parent_id)]
            if true_parent_index not in self.parent_pool_position:
                raise ValueError(f"True parent is not a candidate parent: {row.parent_id}")

    def edge_frame(self, split: str) -> pd.DataFrame:
        return self.edges[self.edges["split"] == split].reset_index(drop=True)

    def batch_arrays(
        self, frame: pd.DataFrame, candidate_mode: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        child_indices = np.asarray(
            [self.node_index[str(value)] for value in frame["child_id"]], dtype=np.int64
        )
        parent_indices = np.asarray(
            [self.node_index[str(value)] for value in frame["parent_id"]], dtype=np.int64
        )
        target_positions = np.asarray(
            [self.parent_pool_position[int(value)] for value in parent_indices], dtype=np.int64
        )
        child_levels = self.nodes.loc[child_indices, "level"].to_numpy(dtype=np.int64)
        parent_levels = self.nodes.loc[self.parent_pool, "level"].to_numpy(dtype=np.int64)
        if candidate_mode == "previous_level":
            allowed = parent_levels[None, :] == child_levels[:, None] - 1
        elif candidate_mode == "any_ancestor_level":
            allowed = parent_levels[None, :] < child_levels[:, None]
        else:
            raise ValueError(f"Unknown candidate mode: {candidate_mode}")
        if not np.all(allowed[np.arange(len(frame)), target_positions]):
            raise ValueError("At least one gold parent is excluded by the candidate-level rule")
        return child_indices, target_positions, allowed


def _slices(
    length: int, batch_size: int, shuffle: bool, rng: np.random.Generator
) -> Iterable[np.ndarray]:
    indices = np.arange(length)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, length, batch_size):
        yield indices[start : start + batch_size]


def _encode_rows(
    model: HierarchicalTreeModel,
    tensors: dict[str, torch.Tensor],
    rows: torch.Tensor,
    text_source_rows: torch.Tensor | None = None,
    apply_modality_dropout: bool = True,
) -> dict[str, torch.Tensor]:
    text_rows = rows if text_source_rows is None else text_source_rows
    return model.encode(
        tensors["text"][text_rows],
        tensors["image"][rows],
        tensors["text_mask"][text_rows],
        tensors["image_mask"][rows],
        group_image=tensors.get("group_image", tensors["image"])[rows],
        image_delta=tensors.get("image_delta", torch.zeros_like(tensors["image"]))[rows],
        levels=tensors.get("levels", torch.zeros_like(tensors["text_mask"], dtype=torch.long))[
            rows
        ],
        apply_modality_dropout=apply_modality_dropout,
    )


def _hierarchical_logits(
    raw_logits: torch.Tensor,
    task: ParentTask,
    macro_score_weight: float,
    macro_aggregation: str = "independent",
) -> tuple[torch.Tensor, torch.Tensor]:
    if macro_aggregation == "independent":
        macro_positions = torch.as_tensor(
            task.macro_parent_positions,
            dtype=torch.long,
            device=raw_logits.device,
        )
        macro_logits = raw_logits[:, macro_positions]
    elif macro_aggregation == "fine_logsumexp":
        columns: list[torch.Tensor] = []
        for macro_position in task.macro_parent_positions:
            fine_positions = np.flatnonzero(task.candidate_macro_parent_positions == macro_position)
            selected = torch.as_tensor(
                fine_positions,
                dtype=torch.long,
                device=raw_logits.device,
            )
            columns.append(torch.logsumexp(raw_logits[:, selected], dim=1))
        macro_logits = torch.stack(columns, dim=1)
    else:
        raise ValueError(f"Unsupported macro aggregation: {macro_aggregation}")
    if macro_score_weight == 0.0:
        return raw_logits, macro_logits
    mapping = torch.as_tensor(
        task.candidate_macro_local_positions,
        dtype=torch.long,
        device=raw_logits.device,
    )
    valid = mapping >= 0
    mapped = macro_logits[:, mapping.clamp_min(0)]
    return raw_logits + macro_score_weight * mapped * valid.unsqueeze(0), macro_logits


def _fixed_macro_f1(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    scores: list[float] = []
    for label in labels:
        true_positive = int(np.sum((targets == label) & (predictions == label)))
        false_positive = int(np.sum((targets != label) & (predictions == label)))
        false_negative = int(np.sum((targets == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _soft_macro_f1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Macro-F1 over classes represented in the current batch."""
    probabilities = torch.softmax(logits, dim=1)
    truth = F.one_hot(targets, num_classes=logits.shape[1]).to(probabilities.dtype)
    true_positive = (probabilities * truth).sum(dim=0)
    false_positive = (probabilities * (1.0 - truth)).sum(dim=0)
    false_negative = ((1.0 - probabilities) * truth).sum(dim=0)
    support = truth.sum(dim=0) > 0
    if not torch.any(support):
        return logits.sum() * 0.0
    soft_f1 = (2.0 * true_positive + 1e-6) / (
        2.0 * true_positive + false_positive + false_negative + 1e-6
    )
    return 1.0 - soft_f1[support].mean()


def _group_dispersion(
    features: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    """Mean cosine dispersion from the within-capture group centroid."""
    losses: list[torch.Tensor] = []
    normalized = F.normalize(features, dim=-1)
    for group_id in torch.unique(groups):
        mask = groups == group_id
        if int(mask.sum()) < 2:
            continue
        group_features = normalized[mask]
        centroid = F.normalize(group_features.mean(dim=0, keepdim=True), dim=-1)
        losses.append((1.0 - (group_features * centroid).sum(dim=-1)).mean())
    if not losses:
        return features.sum() * 0.0
    return torch.stack(losses).mean()


def _context_regularization_losses(
    model: HierarchicalTreeModel,
    tensors: dict[str, torch.Tensor],
    selected_children: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use safe pose text to canonicalize views without exposing BBCH labels.

    The correct loss makes the four views from one plant/capture group compact.
    The counterfactual term requires the correct camera/pose descriptions to do
    this at least ``margin`` better than descriptions rotated within that group.
    """
    if "group_index" not in tensors:
        zero = tensors["image"][selected_children].sum() * 0.0
        return zero, zero
    selected_groups = torch.unique(tensors["group_index"][selected_children])
    selected_groups = selected_groups[selected_groups >= 0]
    if selected_groups.numel() == 0:
        zero = tensors["image"][selected_children].sum() * 0.0
        return zero, zero

    all_groups = tensors["group_index"]
    group_mask = torch.isin(all_groups, selected_groups)
    rows = torch.nonzero(group_mask, as_tuple=False).squeeze(1)
    groups = all_groups[rows]
    available = (tensors["text_mask"][rows] > 0) & (tensors["image_mask"][rows] > 0)
    rows = rows[available]
    groups = groups[available]
    if rows.numel() < 2:
        zero = tensors["image"][selected_children].sum() * 0.0
        return zero, zero

    wrong_text_rows = rows.clone()
    for group_id in torch.unique(groups):
        positions = torch.nonzero(groups == group_id, as_tuple=False).squeeze(1)
        if positions.numel() >= 2:
            wrong_text_rows[positions] = torch.roll(rows[positions], shifts=1)

    correct = _encode_rows(
        model,
        tensors,
        rows,
        apply_modality_dropout=False,
    )["fused"]
    counterfactual = _encode_rows(
        model,
        tensors,
        rows,
        text_source_rows=wrong_text_rows,
        apply_modality_dropout=False,
    )["fused"]
    correct_dispersion = _group_dispersion(correct, groups)
    counterfactual_dispersion = _group_dispersion(counterfactual, groups)
    counterfactual_loss = F.relu(correct_dispersion - counterfactual_dispersion + margin)
    return correct_dispersion, counterfactual_loss


def _text_counterfactual_rank_loss(
    model: HierarchicalTreeModel,
    task: ParentTask,
    tensors: dict[str, torch.Tensor],
    parent_encoded: dict[str, torch.Tensor],
    selected_children: torch.Tensor,
    selected_targets: torch.Tensor,
    selected_allowed: torch.Tensor,
    macro_score_weight: float,
    macro_aggregation: str,
    margin: float,
) -> torch.Tensor:
    """Favor the true parent under correct rather than within-group rotated text."""
    if "counterfactual_text_row" not in tensors:
        return parent_encoded["fused"].sum() * 0.0
    correct_child = _encode_rows(
        model,
        tensors,
        selected_children,
        apply_modality_dropout=False,
    )

    def target_log_probability(child: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = model.score_candidates(
            parent_encoded["fused"],
            parent_encoded["hyperbolic"],
            child["fused"],
            child["hyperbolic"],
        )
        logits, _ = _hierarchical_logits(
            raw,
            task,
            macro_score_weight,
            macro_aggregation,
        )
        logits = logits.masked_fill(~selected_allowed, -1e9)
        return F.log_softmax(logits, dim=1).gather(1, selected_targets.unsqueeze(1)).squeeze(1)

    correct_log_probability = target_log_probability(correct_child)
    losses: list[torch.Tensor] = []
    for source_key in ("counterfactual_text_row", "constant_text_row"):
        if source_key not in tensors:
            continue
        wrong_child = _encode_rows(
            model,
            tensors,
            selected_children,
            text_source_rows=tensors[source_key][selected_children],
            apply_modality_dropout=False,
        )
        wrong_log_probability = target_log_probability(wrong_child)
        losses.append(F.relu(margin - (correct_log_probability - wrong_log_probability)).mean())
    if not losses:
        return correct_log_probability.sum() * 0.0
    return torch.stack(losses).mean()


def _preflight_backward(
    model: HierarchicalTreeModel,
    task: ParentTask,
    frame: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
    candidate_mode: str,
    depth_margin: float,
    lambda_depth: float,
    lambda_align: float,
    lambda_macro: float,
    lambda_view_consistency: float,
    lambda_text_counterfactual: float,
    text_counterfactual_margin: float,
    lambda_text_rank: float,
    text_rank_margin: float,
    macro_score_weight: float,
    macro_aggregation: str,
    classification_loss_mode: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    lambda_balanced_ce: float,
    balanced_class_weights: torch.Tensor | None,
    lambda_soft_macro_f1: float,
) -> float:
    """Run one tiny forward/backward pass without updating model weights."""
    selected_frame = frame.iloc[: min(2, len(frame))].reset_index(drop=True)
    child_indices, target_positions, allowed = task.batch_arrays(selected_frame, candidate_mode)
    device = tensors["text"].device
    parent_rows = torch.as_tensor(task.parent_pool, dtype=torch.long, device=device)
    child_rows = torch.as_tensor(child_indices, dtype=torch.long, device=device)
    targets = torch.as_tensor(target_positions, dtype=torch.long, device=device)
    allowed_mask = torch.as_tensor(allowed, dtype=torch.bool, device=device)

    model.train()
    model.zero_grad(set_to_none=True)
    parent_encoded = _encode_rows(model, tensors, parent_rows)
    child_encoded = _encode_rows(model, tensors, child_rows)
    raw_logits = model.score_candidates(
        parent_encoded["fused"],
        parent_encoded["hyperbolic"],
        child_encoded["fused"],
        child_encoded["hyperbolic"],
    )
    logits, macro_logits = _hierarchical_logits(
        raw_logits,
        task,
        macro_score_weight,
        macro_aggregation,
    )
    logits = logits.masked_fill(~allowed_mask, -1e9)
    classification_loss = _classification_loss(
        logits,
        targets,
        mode=classification_loss_mode,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
    )
    balanced_ce_loss = (
        F.cross_entropy(logits, targets, weight=balanced_class_weights)
        if balanced_class_weights is not None
        else logits.sum() * 0.0
    )
    soft_macro_f1_loss = _soft_macro_f1_loss(logits, targets)
    depth_loss = model.depth_loss(
        parent_encoded["hyperbolic"][targets],
        child_encoded["hyperbolic"],
        margin=depth_margin,
    )
    alignment_loss = model.alignment_loss(
        child_encoded["text"],
        child_encoded["image"],
        tensors["text_mask"][child_rows],
        tensors["image_mask"][child_rows],
    )
    macro_targets = torch.as_tensor(
        task.candidate_macro_local_positions[target_positions],
        dtype=torch.long,
        device=device,
    )
    macro_loss = F.cross_entropy(macro_logits, macro_targets)
    view_consistency_loss, text_counterfactual_loss = _context_regularization_losses(
        model,
        tensors,
        child_rows,
        margin=text_counterfactual_margin,
    )
    text_rank_loss = _text_counterfactual_rank_loss(
        model,
        task,
        tensors,
        parent_encoded,
        child_rows,
        targets,
        allowed_mask,
        macro_score_weight,
        macro_aggregation,
        margin=text_rank_margin,
    )
    loss = (
        classification_loss
        + lambda_balanced_ce * balanced_ce_loss
        + lambda_soft_macro_f1 * soft_macro_f1_loss
        + lambda_depth * depth_loss
        + lambda_align * alignment_loss
        + lambda_macro * macro_loss
        + lambda_view_consistency * view_consistency_loss
        + lambda_text_counterfactual * text_counterfactual_loss
        + lambda_text_rank * text_rank_loss
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Preflight loss is not finite: {float(loss.detach())}")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("Preflight produced missing, NaN, or infinite gradients")
    value = float(loss.detach())
    model.zero_grad(set_to_none=True)
    return value


def _run_split(
    model: HierarchicalTreeModel,
    task: ParentTask,
    frame: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
    candidate_mode: str,
    batch_size: int,
    depth_margin: float,
    lambda_depth: float,
    lambda_align: float,
    lambda_macro: float,
    lambda_view_consistency: float,
    lambda_text_counterfactual: float,
    text_counterfactual_margin: float,
    lambda_text_rank: float,
    text_rank_margin: float,
    macro_score_weight: float,
    macro_aggregation: str,
    classification_loss_mode: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    lambda_balanced_ce: float,
    balanced_class_weights: torch.Tensor | None,
    lambda_soft_macro_f1: float,
    group_consensus_weight: float,
    inference_logit_adjustment_tau: float,
    class_log_prior: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
) -> tuple[dict[str, float], list[dict]]:
    if frame.empty:
        return {"loss": math.nan, "hits@1": math.nan, "mrr": math.nan}, []
    is_training = optimizer is not None
    model.train(is_training)
    child_indices, target_positions, allowed = task.batch_arrays(frame, candidate_mode)
    parent_rows = torch.as_tensor(task.parent_pool, dtype=torch.long, device=tensors["text"].device)

    total_loss = 0.0
    total_balanced_ce = 0.0
    total_soft_macro_f1 = 0.0
    total_view_consistency = 0.0
    total_text_counterfactual = 0.0
    total_text_rank = 0.0
    total_fusion_gate = 0.0
    total_examples = 0
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    child_parts: list[torch.Tensor] = []
    row_indices: list[int] = []
    for selection in _slices(len(frame), batch_size, shuffle=is_training, rng=rng):
        selected_children = torch.as_tensor(
            child_indices[selection], dtype=torch.long, device=parent_rows.device
        )
        selected_targets = torch.as_tensor(
            target_positions[selection], dtype=torch.long, device=parent_rows.device
        )
        selected_allowed = torch.as_tensor(
            allowed[selection], dtype=torch.bool, device=parent_rows.device
        )

        with torch.set_grad_enabled(is_training):
            parent_encoded = _encode_rows(model, tensors, parent_rows)
            child_encoded = _encode_rows(model, tensors, selected_children)
            raw_logits = model.score_candidates(
                parent_encoded["fused"],
                parent_encoded["hyperbolic"],
                child_encoded["fused"],
                child_encoded["hyperbolic"],
            )
            logits, macro_logits = _hierarchical_logits(
                raw_logits,
                task,
                macro_score_weight,
                macro_aggregation,
            )
            logits = logits.masked_fill(~selected_allowed, -1e9)
            classification_loss = _classification_loss(
                logits,
                selected_targets,
                mode=classification_loss_mode,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
            )
            balanced_ce_loss = (
                F.cross_entropy(
                    logits,
                    selected_targets,
                    weight=balanced_class_weights,
                )
                if balanced_class_weights is not None
                else logits.sum() * 0.0
            )
            soft_macro_f1_loss = _soft_macro_f1_loss(logits, selected_targets)
            positive_parent_h = parent_encoded["hyperbolic"][selected_targets]
            depth_loss = model.depth_loss(
                positive_parent_h, child_encoded["hyperbolic"], margin=depth_margin
            )
            alignment_loss = model.alignment_loss(
                child_encoded["text"],
                child_encoded["image"],
                tensors["text_mask"][selected_children],
                tensors["image_mask"][selected_children],
            )
            macro_targets = torch.as_tensor(
                task.candidate_macro_local_positions[target_positions[selection]],
                dtype=torch.long,
                device=parent_rows.device,
            )
            macro_loss = F.cross_entropy(macro_logits, macro_targets)
            if is_training and (lambda_view_consistency > 0.0 or lambda_text_counterfactual > 0.0):
                (
                    view_consistency_loss,
                    text_counterfactual_loss,
                ) = _context_regularization_losses(
                    model,
                    tensors,
                    selected_children,
                    margin=text_counterfactual_margin,
                )
            else:
                view_consistency_loss = child_encoded["fused"].sum() * 0.0
                text_counterfactual_loss = child_encoded["fused"].sum() * 0.0
            if is_training and lambda_text_rank > 0.0:
                text_rank_loss = _text_counterfactual_rank_loss(
                    model,
                    task,
                    tensors,
                    parent_encoded,
                    selected_children,
                    selected_targets,
                    selected_allowed,
                    macro_score_weight,
                    macro_aggregation,
                    margin=text_rank_margin,
                )
            else:
                text_rank_loss = child_encoded["fused"].sum() * 0.0
            loss = (
                classification_loss
                + lambda_balanced_ce * balanced_ce_loss
                + lambda_soft_macro_f1 * soft_macro_f1_loss
                + lambda_depth * depth_loss
                + lambda_align * alignment_loss
                + lambda_macro * macro_loss
                + lambda_view_consistency * view_consistency_loss
                + lambda_text_counterfactual * text_counterfactual_loss
                + lambda_text_rank * text_rank_loss
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_length = len(selection)
        total_loss += float(loss.detach()) * batch_length
        total_balanced_ce += float(balanced_ce_loss.detach()) * batch_length
        total_soft_macro_f1 += float(soft_macro_f1_loss.detach()) * batch_length
        total_view_consistency += float(view_consistency_loss.detach()) * batch_length
        total_text_counterfactual += float(text_counterfactual_loss.detach()) * batch_length
        total_text_rank += float(text_rank_loss.detach()) * batch_length
        total_fusion_gate += float(child_encoded["fusion_gate"].mean().detach()) * batch_length
        total_examples += batch_length
        logits_parts.append(logits.detach())
        target_parts.append(selected_targets.detach())
        child_parts.append(selected_children.detach())
        row_indices.extend(int(value) for value in selection)

    evaluation_logits = torch.cat(logits_parts, dim=0)
    evaluation_targets = torch.cat(target_parts, dim=0)
    evaluation_children = torch.cat(child_parts, dim=0)
    if not is_training and group_consensus_weight > 0.0:
        if "group_index" not in tensors:
            raise ValueError("Group consensus requires group_index tensors")
        evaluation_logits = _apply_group_consensus(
            evaluation_logits,
            tensors["group_index"][evaluation_children],
            group_consensus_weight,
        )
    if not is_training and inference_logit_adjustment_tau > 0.0:
        evaluation_logits = evaluation_logits - (
            inference_logit_adjustment_tau * class_log_prior.unsqueeze(0)
        )
    target_logits = evaluation_logits.gather(1, evaluation_targets.unsqueeze(1))
    evaluation_ranks = 1 + (evaluation_logits > target_logits).sum(dim=1)
    probabilities = torch.softmax(evaluation_logits, dim=1)
    predicted_positions = evaluation_logits.argmax(dim=1)
    predictions: list[dict] = []
    for local_index, original_index in enumerate(row_indices):
        row = frame.iloc[original_index]
        predicted_node_index = int(task.parent_pool[int(predicted_positions[local_index])])
        predicted_parent = str(task.nodes.iloc[predicted_node_index]["node_id"])
        true_parent = str(row["parent_id"])
        predictions.append(
            {
                "split": str(row["split"]),
                "child_id": str(row["child_id"]),
                "true_parent_id": true_parent,
                "predicted_parent_id": predicted_parent,
                "rank": int(evaluation_ranks[local_index]),
                "confidence": float(probabilities[local_index, predicted_positions[local_index]]),
                "correct": int(predicted_parent == true_parent),
                "same_upper_parent": int(
                    task.fixed_parent.get(predicted_parent) == task.fixed_parent.get(true_parent)
                ),
            }
        )

    rank_array = evaluation_ranks.cpu().numpy().astype(np.float64)
    target_array = evaluation_targets.cpu().numpy().astype(np.int64)
    prediction_array = predicted_positions.cpu().numpy().astype(np.int64)
    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "balanced_ce_loss": total_balanced_ce / max(total_examples, 1),
        "soft_macro_f1_loss": total_soft_macro_f1 / max(total_examples, 1),
        "hits@1": float(np.mean(rank_array <= 1)),
        "hits@3": float(np.mean(rank_array <= 3)),
        "hits@5": float(np.mean(rank_array <= 5)),
        "mrr": float(np.mean(1.0 / rank_array)),
        "upper_parent_accuracy": float(np.mean([p["same_upper_parent"] for p in predictions])),
        "fine_macro_f1_50_classes": _fixed_macro_f1(
            target_array,
            prediction_array,
            task.fine_parent_positions,
        ),
        "view_consistency_loss": total_view_consistency / max(total_examples, 1),
        "text_counterfactual_loss": total_text_counterfactual / max(total_examples, 1),
        "text_rank_loss": total_text_rank / max(total_examples, 1),
        "fusion_gate_mean": total_fusion_gate / max(total_examples, 1),
        "count": int(total_examples),
    }
    return metrics, predictions


def _context_diagnostics(
    model: HierarchicalTreeModel,
    task: ParentTask,
    frame: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
) -> dict[str, float]:
    child_indices, _, _ = task.batch_arrays(frame, "previous_level")
    rows = torch.as_tensor(
        child_indices,
        dtype=torch.long,
        device=tensors["text"].device,
    )
    encoded = _encode_rows(
        model,
        tensors,
        rows,
        apply_modality_dropout=False,
    )
    gates = encoded["fusion_gate"].squeeze(-1)
    result = {
        "fusion_gate_mean": float(gates.mean().item()),
        "fusion_gate_std_population": float(gates.std(unbiased=False).item()),
        "fusion_gate_minimum": float(gates.min().item()),
        "fusion_gate_maximum": float(gates.max().item()),
    }
    if "group_index" in tensors:
        groups = tensors["group_index"][rows]
        result["within_group_cosine_dispersion"] = float(
            _group_dispersion(encoded["fused"], groups).item()
        )
    return result


def train(
    config_path: Path,
    device_override: str | None = None,
    seed_override: int | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, dict[str, float]]:
    config = load_yaml(config_path)
    if seed_override is not None:
        config["seed"] = seed_override
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    rng = np.random.default_rng(seed)
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(data_dir)
    node_ids = tables.nodes["node_id"].astype(str).tolist()
    embeddings_path = Path(config.get("embeddings_path", data_dir / "embeddings.npz"))
    bundle = load_embeddings(embeddings_path, node_ids)
    if not np.isfinite(bundle.text).all() or not np.isfinite(bundle.image).all():
        raise FloatingPointError("Embeddings contain NaN or infinity")
    if (
        not np.isin(bundle.text_mask, [0.0, 1.0]).all()
        or not np.isin(bundle.image_mask, [0.0, 1.0]).all()
    ):
        raise ValueError("Embedding masks must contain only 0 and 1")
    if np.any((bundle.text_mask + bundle.image_mask) == 0):
        missing_count = int(np.sum((bundle.text_mask + bundle.image_mask) == 0))
        raise ValueError(f"{missing_count} knowledge nodes have neither text nor image embeddings")
    configured_dim = int(config.get("embedding_dim", bundle.dim))
    if configured_dim != bundle.dim:
        raise ValueError(
            f"Config embedding_dim={configured_dim}, but embeddings have dimension {bundle.dim}"
        )

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_text_mode = str(config.get("item_text_mode", "enabled"))
    if item_text_mode not in {"enabled", "disabled", "shuffled", "constant"}:
        raise ValueError(f"Unsupported item_text_mode: {item_text_mode}")
    item_rows = tables.nodes.index[tables.nodes["level"].astype(int) == 4].to_numpy()
    use_group_context = bool(config.get("use_group_context", False))
    group_consensus_weight = float(config.get("group_consensus_weight", 0.0))
    if not 0.0 <= group_consensus_weight <= 1.0:
        raise ValueError("group_consensus_weight must be between 0 and 1")
    if group_consensus_weight > 0.0 and not use_group_context:
        raise ValueError("Group consensus requires use_group_context=true")
    lambda_view_consistency = float(config.get("lambda_view_consistency", 0.0))
    lambda_text_counterfactual = float(config.get("lambda_text_counterfactual", 0.0))
    text_counterfactual_margin = float(config.get("text_counterfactual_margin", 0.03))
    lambda_text_rank = float(config.get("lambda_text_rank", 0.0))
    text_rank_margin = float(config.get("text_rank_margin", 0.03))
    text_rank_negative_mode = str(config.get("text_rank_negative_mode", "rotated"))
    if text_rank_negative_mode not in {"rotated", "rotated_constant"}:
        raise ValueError(f"Unsupported text_rank_negative_mode: {text_rank_negative_mode}")
    if lambda_view_consistency < 0.0 or lambda_text_counterfactual < 0.0 or lambda_text_rank < 0.0:
        raise ValueError("Context regularization weights must be non-negative")
    if text_counterfactual_margin < 0.0 or text_rank_margin < 0.0:
        raise ValueError("Counterfactual margins must be non-negative")
    if (
        lambda_view_consistency > 0.0 or lambda_text_counterfactual > 0.0 or lambda_text_rank > 0.0
    ) and not use_group_context:
        raise ValueError("Context regularization requires use_group_context=true")
    group_context_stats: dict[str, int] | None = None
    group_image: np.ndarray | None = None
    image_delta: np.ndarray | None = None
    group_index: np.ndarray | None = None
    if use_group_context:
        group_image, image_delta, group_index, group_context_stats = _group_context_arrays(
            bundle,
            data_dir,
        )
    tensors = _tensor_bundle(
        bundle,
        device,
        item_text_rows=item_rows,
        item_text_mode=item_text_mode,
        item_text_shuffle_seed=int(config.get("item_text_shuffle_seed", 2026)),
        group_image=group_image,
        image_delta=image_delta,
        levels=tables.nodes["level"].to_numpy(dtype=np.int64),
        group_index=group_index,
    )
    if group_index is not None:
        tensors["counterfactual_text_row"] = torch.from_numpy(
            _counterfactual_text_rows(group_index)
        ).to(device)
        if text_rank_negative_mode == "rotated_constant":
            constant_text_rows = np.arange(len(node_ids), dtype=np.int64)
            constant_text_rows[item_rows] = int(item_rows[0])
            tensors["constant_text_row"] = torch.from_numpy(constant_text_rows).to(device)
    task = ParentTask(tables)
    train_frame = task.edge_frame("train")
    val_frame = task.edge_frame("val")
    test_frame = task.edge_frame("test")
    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError("Train, validation, and test edges are all required")
    classification_loss_mode = str(config.get("classification_loss", "cross_entropy"))
    if classification_loss_mode not in {"cross_entropy", "class_weighted", "focal"}:
        raise ValueError(f"Unsupported classification_loss: {classification_loss_mode}")
    class_weights = None
    if classification_loss_mode in {"class_weighted", "focal"}:
        class_weights = _class_weights(
            task,
            train_frame,
            power=float(config.get("class_weight_power", 0.5)),
            device=device,
        )
    lambda_balanced_ce = float(config.get("lambda_balanced_ce", 0.0))
    balanced_ce_power = float(config.get("balanced_ce_power", 0.25))
    if lambda_balanced_ce < 0.0:
        raise ValueError("lambda_balanced_ce must be non-negative")
    if balanced_ce_power < 0.0:
        raise ValueError("balanced_ce_power must be non-negative")
    lambda_soft_macro_f1 = float(config.get("lambda_soft_macro_f1", 0.0))
    if lambda_soft_macro_f1 < 0.0:
        raise ValueError("lambda_soft_macro_f1 must be non-negative")
    balanced_class_weights = (
        _class_weights(
            task,
            train_frame,
            power=balanced_ce_power,
            device=device,
        )
        if lambda_balanced_ce > 0.0
        else None
    )
    focal_gamma = float(config.get("focal_gamma", 2.0))
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    inference_logit_adjustment_tau = float(config.get("inference_logit_adjustment_tau", 0.0))
    if inference_logit_adjustment_tau < 0.0:
        raise ValueError("inference_logit_adjustment_tau must be non-negative")
    class_log_prior = _class_log_prior(task, train_frame, device)
    model = HierarchicalTreeModel(
        embedding_dim=bundle.dim,
        hidden_dim=int(config["hidden_dim"]),
        tree_dim=int(config["tree_dim"]),
        dropout=float(config.get("dropout", 0.1)),
        curvature=float(config.get("curvature", 1.0)),
        fusion_mode=str(config.get("fusion_mode", "gate")),
        geometry=str(config.get("geometry", "poincare")),
        score_mode=str(config.get("score_mode", "directed")),
        controlled_tangent=bool(config.get("controlled_tangent", False)),
        max_tangent_norm=float(config.get("max_tangent_norm", 1.5)),
        level_aware_radius=bool(config.get("level_aware_radius", False)),
        max_level=int(tables.nodes["level"].max()),
        min_tangent_radius=float(config.get("min_tangent_radius", 0.10)),
        max_tangent_radius=float(config.get("max_tangent_radius", 1.20)),
        radius_residual_scale=float(config.get("radius_residual_scale", 0.05)),
        use_group_context=use_group_context,
        group_context_mode=str(config.get("group_context_mode", "residual")),
        use_prototype_head=bool(config.get("use_prototype_head", False)),
        use_geometric_prototypes=bool(config.get("use_geometric_prototypes", False)),
        candidate_count=len(task.parent_pool),
        relation_dim=int(config.get("relation_dim", 256)),
        geometric_residual_scale=float(config.get("geometric_residual_scale", 0.20)),
        distance_scale_init=float(config.get("distance_scale_init", 0.70)),
        context_residual_scale=float(config.get("context_residual_scale", 0.35)),
        context_gate_floor=float(config.get("context_gate_floor", 0.0)),
        text_modality_dropout=float(config.get("text_modality_dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    common = {
        "model": model,
        "task": task,
        "tensors": tensors,
        "candidate_mode": str(config.get("candidate_mode", "previous_level")),
        "batch_size": int(config.get("batch_size", 256)),
        "depth_margin": float(config.get("depth_margin", 0.05)),
        "lambda_depth": float(config.get("lambda_depth", 0.1)),
        "lambda_align": float(config.get("lambda_align", 0.05)),
        "lambda_macro": float(config.get("lambda_macro", 0.0)),
        "lambda_view_consistency": lambda_view_consistency,
        "lambda_text_counterfactual": lambda_text_counterfactual,
        "text_counterfactual_margin": text_counterfactual_margin,
        "lambda_text_rank": lambda_text_rank,
        "text_rank_margin": text_rank_margin,
        "macro_score_weight": float(config.get("macro_score_weight", 0.0)),
        "macro_aggregation": str(config.get("macro_aggregation", "independent")),
        "classification_loss_mode": classification_loss_mode,
        "class_weights": class_weights,
        "focal_gamma": focal_gamma,
        "lambda_balanced_ce": lambda_balanced_ce,
        "balanced_class_weights": balanced_class_weights,
        "lambda_soft_macro_f1": lambda_soft_macro_f1,
        "group_consensus_weight": group_consensus_weight,
        "inference_logit_adjustment_tau": inference_logit_adjustment_tau,
        "class_log_prior": class_log_prior,
        "rng": rng,
    }

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_info = {
            "device": str(device),
            "gpu_name": properties.name,
            "gpu_memory_gb": round(properties.total_memory / (1024**3), 2),
        }
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device_info = {"device": str(device), "gpu_name": None, "gpu_memory_gb": 0.0}
    preflight_loss = _preflight_backward(
        model=model,
        task=task,
        frame=train_frame,
        tensors=tensors,
        candidate_mode=common["candidate_mode"],
        depth_margin=common["depth_margin"],
        lambda_depth=common["lambda_depth"],
        lambda_align=common["lambda_align"],
        lambda_macro=common["lambda_macro"],
        lambda_view_consistency=common["lambda_view_consistency"],
        lambda_text_counterfactual=common["lambda_text_counterfactual"],
        text_counterfactual_margin=common["text_counterfactual_margin"],
        lambda_text_rank=common["lambda_text_rank"],
        text_rank_margin=common["text_rank_margin"],
        macro_score_weight=common["macro_score_weight"],
        macro_aggregation=common["macro_aggregation"],
        classification_loss_mode=classification_loss_mode,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        lambda_balanced_ce=lambda_balanced_ce,
        balanced_class_weights=balanced_class_weights,
        lambda_soft_macro_f1=lambda_soft_macro_f1,
    )
    peak_memory_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
        if device.type == "cuda"
        else 0.0
    )
    preflight_report = {
        **device_info,
        "seed": seed,
        "node_count": len(node_ids),
        "candidate_parent_count": len(task.parent_pool),
        "train_edges": len(train_frame),
        "val_edges": len(val_frame),
        "test_edges": len(test_frame),
        "embedding_dim": bundle.dim,
        "embeddings_path": str(embeddings_path.resolve()),
        "text_node_count": int(tensors["text_mask"].sum().item()),
        "image_node_count": int(tensors["image_mask"].sum().item()),
        "fusion_mode": str(config.get("fusion_mode", "gate")),
        "geometry": str(config.get("geometry", "poincare")),
        "score_mode": str(config.get("score_mode", "directed")),
        "item_text_mode": item_text_mode,
        "item_text_shuffle_seed": int(config.get("item_text_shuffle_seed", 2026)),
        "use_group_context": use_group_context,
        "group_context": group_context_stats,
        "group_context_mode": str(config.get("group_context_mode", "residual")),
        "group_consensus_weight": group_consensus_weight,
        "context_residual_scale": float(config.get("context_residual_scale", 0.35)),
        "context_gate_floor": float(config.get("context_gate_floor", 0.0)),
        "text_modality_dropout": float(config.get("text_modality_dropout", 0.0)),
        "controlled_tangent": bool(config.get("controlled_tangent", False)),
        "level_aware_radius": bool(config.get("level_aware_radius", False)),
        "use_prototype_head": bool(config.get("use_prototype_head", False)),
        "use_geometric_prototypes": bool(config.get("use_geometric_prototypes", False)),
        "classification_loss": classification_loss_mode,
        "class_weight_power": float(config.get("class_weight_power", 0.5)),
        "focal_gamma": focal_gamma,
        "lambda_balanced_ce": lambda_balanced_ce,
        "balanced_ce_power": balanced_ce_power,
        "lambda_soft_macro_f1": lambda_soft_macro_f1,
        "inference_logit_adjustment_tau": inference_logit_adjustment_tau,
        "candidate_mode": str(config.get("candidate_mode", "previous_level")),
        "depth_margin": common["depth_margin"],
        "lambda_depth": common["lambda_depth"],
        "lambda_align": common["lambda_align"],
        "lambda_macro": common["lambda_macro"],
        "lambda_view_consistency": common["lambda_view_consistency"],
        "lambda_text_counterfactual": common["lambda_text_counterfactual"],
        "text_counterfactual_margin": common["text_counterfactual_margin"],
        "lambda_text_rank": common["lambda_text_rank"],
        "text_rank_margin": common["text_rank_margin"],
        "text_rank_negative_mode": text_rank_negative_mode,
        "evaluate_text_controls": bool(config.get("evaluate_text_controls", False)),
        "macro_score_weight": common["macro_score_weight"],
        "macro_aggregation": common["macro_aggregation"],
        "one_batch_loss": preflight_loss,
        "peak_gpu_memory_mb": peak_memory_mb,
        "status": "passed",
    }
    write_json(output_dir / "preflight.json", preflight_report)
    print(f"Preflight passed: {preflight_report}")
    # Keep the formal run reproducible; preflight consumed dropout RNG only.
    set_seed(seed)
    rng = np.random.default_rng(seed)
    common["rng"] = rng

    best_state: dict | None = None
    best_val_score = -math.inf
    stale_epochs = 0
    history: list[dict] = []
    epochs = int(config.get("epochs", 100))
    patience = int(config.get("patience", 12))
    print(f"Training on {device}; {len(train_frame)} train and {len(val_frame)} validation edges")
    for epoch in range(1, epochs + 1):
        train_metrics, _ = _run_split(frame=train_frame, optimizer=optimizer, **common)
        with torch.no_grad():
            val_metrics, _ = _run_split(frame=val_frame, optimizer=None, **common)
        selection_score = (
            float(config.get("selection_mrr_weight", 1.0)) * val_metrics["mrr"]
            + float(config.get("selection_hits1_weight", 0.0)) * val_metrics["hits@1"]
            + float(config.get("selection_macro_f1_weight", 0.0))
            * val_metrics["fine_macro_f1_50_classes"]
        )
        history.append(
            {
                "epoch": epoch,
                "selection_score": selection_score,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_mrr={val_metrics['mrr']:.4f} val_h@1={val_metrics['hits@1']:.4f} "
            f"val_f1={val_metrics['fine_macro_f1_50_classes']:.4f} "
            f"score={selection_score:.4f}"
        )
        if selection_score > best_val_score + 1e-6:
            best_val_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {"model_state": best_state, "config": config, "node_ids": node_ids},
        output_dir / "best_model.pt",
    )
    write_json(output_dir / "history.json", history)

    if bool(config.get("evaluate_text_controls", False)):
        if item_text_mode != "enabled":
            raise ValueError("Text controls require item_text_mode=enabled")
        text_control_report: dict[str, object] = {
            "split": "val",
            "same_checkpoint": True,
            "test_set_used": False,
            "shuffle_seed": int(config.get("item_text_shuffle_seed", 2026)),
            "conditions": {},
        }
        with torch.no_grad():
            for label, control_mode in (
                ("correct_text", "enabled"),
                ("shuffled_text", "shuffled"),
                ("constant_text", "constant"),
                ("no_text", "disabled"),
            ):
                control_tensors = _tensor_bundle(
                    bundle,
                    device,
                    item_text_rows=item_rows,
                    item_text_mode=control_mode,
                    item_text_shuffle_seed=int(config.get("item_text_shuffle_seed", 2026)),
                    group_image=group_image,
                    image_delta=image_delta,
                    levels=tables.nodes["level"].to_numpy(dtype=np.int64),
                    group_index=group_index,
                )
                control_common = {**common, "tensors": control_tensors}
                control_metrics, _ = _run_split(
                    frame=val_frame,
                    optimizer=None,
                    **control_common,
                )
                text_control_report["conditions"][label] = {
                    "metrics": control_metrics,
                    "diagnostics": _context_diagnostics(
                        model,
                        task,
                        val_frame,
                        control_tensors,
                    ),
                }
        conditions = text_control_report["conditions"]
        assert isinstance(conditions, dict)
        correct_metrics = conditions["correct_text"]["metrics"]
        text_control_report["correct_minus_control"] = {
            label: {
                metric: float(correct_metrics[metric] - values["metrics"][metric])
                for metric in ("hits@1", "mrr", "fine_macro_f1_50_classes")
            }
            for label, values in conditions.items()
            if label != "correct_text"
        }
        write_json(output_dir / "text_control_metrics.json", text_control_report)

    final_metrics: dict[str, dict[str, float]] = {}
    all_predictions: list[dict] = []
    evaluate_test = bool(config.get("evaluate_test", True))
    evaluation_splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    with torch.no_grad():
        for split in evaluation_splits:
            metrics, predictions = _run_split(
                frame=task.edge_frame(split), optimizer=None, **common
            )
            final_metrics[split] = metrics
            all_predictions.extend(predictions)
    write_json(output_dir / "metrics.json", final_metrics)
    pd.DataFrame(all_predictions).sort_values(["split", "child_id"]).to_csv(
        output_dir / "predicted_edges.csv", index=False, encoding="utf-8-sig"
    )

    if evaluate_test:
        fixed = tables.edges[tables.edges["is_fixed"] == 1][
            ["parent_id", "child_id", "relation"]
        ].copy()
        predicted = pd.DataFrame(all_predictions)[["predicted_parent_id", "child_id"]].rename(
            columns={"predicted_parent_id": "parent_id"}
        )
        predicted["relation"] = "predicted_parent"
        pd.concat([fixed, predicted], ignore_index=True).to_csv(
            output_dir / "constructed_tree_edges.csv", index=False, encoding="utf-8-sig"
        )
        print(f"Final test metrics: {final_metrics['test']}")
    else:
        print(f"Validation-only metrics: {final_metrics['val']}")
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multimodal hierarchical parent model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    train(
        args.config,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
