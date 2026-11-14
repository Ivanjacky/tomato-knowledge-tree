from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .evaluate import evaluate
from .model import ModalityGate
from .schema import load_embeddings, load_tables
from .train import _apply_group_consensus, _fixed_macro_f1, _group_context_arrays
from .utils import set_seed, write_json


class SupervisedParentClassifier(nn.Module):
    """Matched flat and hierarchical classifiers over frozen multimodal features."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        fine_count: int,
        macro_count: int,
        mode: str,
        dropout: float,
        use_group_context: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"linear_image", "gated_mlp", "hierarchical_gated"}:
            raise ValueError(f"Unsupported classifier mode: {mode}")
        self.mode = mode
        self.use_group_context = bool(use_group_context)
        self.context_image_projection = (
            nn.Sequential(
                nn.LayerNorm(embedding_dim * 3),
                nn.Linear(embedding_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if self.use_group_context
            else None
        )
        if mode == "linear_image":
            self.fusion = None
            self.fine_head = nn.Linear(embedding_dim, fine_count)
        else:
            self.fusion = ModalityGate(
                embedding_dim,
                hidden_dim,
                dropout,
                fusion_mode="gate",
            )
            self.fine_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, fine_count),
            )
        self.macro_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, macro_count),
            )
            if mode == "hierarchical_gated"
            else None
        )

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_mask: torch.Tensor,
        image_mask: torch.Tensor,
        group_image: torch.Tensor | None = None,
        image_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mode == "linear_image":
            representation = F.normalize(image, dim=-1)
        else:
            assert self.fusion is not None
            context_image_hidden = None
            if self.use_group_context:
                if group_image is None or image_delta is None:
                    raise ValueError(
                        "Group-aware classifier requires group_image and image_delta"
                    )
                assert self.context_image_projection is not None
                context_image_hidden = self.context_image_projection(
                    torch.cat([image, group_image, image_delta], dim=-1)
                )
            representation, _, _ = self.fusion(
                text,
                image,
                text_mask,
                image_mask,
                image_hidden_override=context_image_hidden,
            )
        fine_logits = self.fine_head(representation)
        macro_logits = self.macro_head(representation) if self.macro_head is not None else None
        return fine_logits, macro_logits


def _rank_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    target_logits = logits.gather(1, targets.unsqueeze(1))
    ranks = 1 + (logits > target_logits).sum(dim=1)
    ranks_np = ranks.detach().cpu().numpy().astype(np.float64)
    predicted = logits.argmax(dim=1)
    return {
        "hits@1": float(np.mean(ranks_np <= 1)),
        "hits@3": float(np.mean(ranks_np <= 3)),
        "hits@5": float(np.mean(ranks_np <= 5)),
        "mrr": float(np.mean(1.0 / ranks_np)),
        "fine_macro_f1_50_classes": _fixed_macro_f1(
            targets.detach().cpu().numpy(),
            predicted.detach().cpu().numpy(),
            np.arange(logits.shape[1], dtype=np.int64),
        ),
        "count": len(ranks_np),
    }


def _batch_indices(
    length: int,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    indices = np.arange(length)
    if shuffle:
        rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, length, batch_size)]


def run_supervised_baseline(
    *,
    data_dir: Path,
    embeddings_path: Path,
    output_dir: Path,
    mode: str,
    seed: int,
    device_name: str | None = None,
    hidden_dim: int = 512,
    dropout: float = 0.2,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 100,
    patience: int = 12,
    lambda_macro: float = 0.5,
    macro_score_weight: float = 1.0,
    use_group_context: bool = False,
    evaluate_test: bool = True,
    selection_mrr_weight: float = 1.0,
    selection_hits1_weight: float = 0.0,
    selection_macro_f1_weight: float = 0.0,
    group_consensus_weight: float = 0.0,
) -> dict[str, Any]:
    if not 0.0 <= group_consensus_weight <= 1.0:
        raise ValueError("group_consensus_weight must be between 0 and 1")
    if group_consensus_weight > 0.0 and not use_group_context:
        raise ValueError("Group consensus requires use_group_context=true")
    set_seed(seed)
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(data_dir)
    nodes = tables.nodes.reset_index(drop=True)
    node_ids = nodes["node_id"].astype(str).tolist()
    bundle = load_embeddings(embeddings_path, node_ids)
    row_by_id = {node_id: index for index, node_id in enumerate(node_ids)}

    fine_ids = sorted(
        nodes.loc[
            (nodes["level"].astype(int) == 3)
            & (nodes["is_candidate_parent"].astype(int) == 1),
            "node_id",
        ].astype(str)
    )
    macro_ids = sorted(
        nodes.loc[
            (nodes["level"].astype(int) == 2)
            & (nodes["is_candidate_parent"].astype(int) == 1),
            "node_id",
        ].astype(str)
    )
    fine_position = {node_id: index for index, node_id in enumerate(fine_ids)}
    macro_position = {node_id: index for index, node_id in enumerate(macro_ids)}
    fixed_parent = {
        str(row.child_id): str(row.parent_id)
        for row in tables.edges[tables.edges["is_fixed"].astype(int) == 1].itertuples(
            index=False
        )
    }
    fine_macro_positions = torch.as_tensor(
        [macro_position[fixed_parent[node_id]] for node_id in fine_ids],
        dtype=torch.long,
    )

    edge_frame = tables.edges[tables.edges["is_fixed"].astype(int) == 0].copy()
    edge_frame = edge_frame.sort_values(["split", "child_id"]).reset_index(drop=True)
    edge_frame["node_row"] = edge_frame["child_id"].astype(str).map(row_by_id)
    edge_frame["fine_target"] = edge_frame["parent_id"].astype(str).map(fine_position)
    edge_frame["macro_target"] = (
        edge_frame["parent_id"].astype(str).map(fixed_parent).map(macro_position)
    )
    if edge_frame[["node_row", "fine_target", "macro_target"]].isna().any().any():
        raise ValueError("Baseline task contains an unmapped node or parent")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensors = {
        "text": torch.from_numpy(bundle.text).to(device),
        "image": torch.from_numpy(bundle.image).to(device),
        "text_mask": torch.from_numpy(bundle.text_mask).to(device),
        "image_mask": torch.from_numpy(bundle.image_mask).to(device),
    }
    group_context_stats = None
    if use_group_context:
        group_image, image_delta, group_index, group_context_stats = _group_context_arrays(
            bundle,
            data_dir,
        )
        tensors["group_image"] = torch.from_numpy(group_image).to(device)
        tensors["image_delta"] = torch.from_numpy(image_delta).to(device)
        tensors["group_index"] = torch.from_numpy(group_index).to(device)
    model = SupervisedParentClassifier(
        embedding_dim=bundle.dim,
        hidden_dim=hidden_dim,
        fine_count=len(fine_ids),
        macro_count=len(macro_ids),
        mode=mode,
        dropout=dropout,
        use_group_context=use_group_context,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    fine_macro_positions = fine_macro_positions.to(device)

    def forward_frame(
        frame: pd.DataFrame,
        *,
        training: bool,
    ) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
        model.train(training)
        total_loss = 0.0
        total = 0
        logits_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        row_parts: list[pd.DataFrame] = []
        for selection in _batch_indices(len(frame), batch_size, training, rng):
            selected = frame.iloc[selection]
            rows = torch.as_tensor(
                selected["node_row"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            fine_targets = torch.as_tensor(
                selected["fine_target"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            macro_targets = torch.as_tensor(
                selected["macro_target"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            with torch.set_grad_enabled(training):
                fine_logits, macro_logits = model(
                    tensors["text"][rows],
                    tensors["image"][rows],
                    tensors["text_mask"][rows],
                    tensors["image_mask"][rows],
                    tensors.get("group_image", tensors["image"])[rows],
                    tensors.get("image_delta", torch.zeros_like(tensors["image"]))[
                        rows
                    ],
                )
                loss = F.cross_entropy(fine_logits, fine_targets)
                ranking_logits = fine_logits
                if macro_logits is not None:
                    loss = loss + lambda_macro * F.cross_entropy(macro_logits, macro_targets)
                    ranking_logits = (
                        fine_logits
                        + macro_score_weight * macro_logits[:, fine_macro_positions]
                    )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
            total += len(selected)
            logits_parts.append(ranking_logits.detach())
            target_parts.append(fine_targets.detach())
            row_parts.append(selected.reset_index(drop=True))

        logits = torch.cat(logits_parts, dim=0)
        targets = torch.cat(target_parts, dim=0)
        rows = pd.concat(row_parts, ignore_index=True)
        if not training and group_consensus_weight > 0.0:
            node_rows = torch.as_tensor(
                rows["node_row"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            logits = _apply_group_consensus(
                logits,
                tensors["group_index"][node_rows],
                group_consensus_weight,
            )
        metrics = {"loss": total_loss / max(total, 1), **_rank_metrics(logits, targets)}
        probabilities = torch.softmax(logits, dim=1)
        predicted = logits.argmax(dim=1)
        target_logits = logits.gather(1, targets.unsqueeze(1))
        ranks = 1 + (logits > target_logits).sum(dim=1)
        predictions: list[dict[str, Any]] = []
        upper_correct: list[int] = []
        for index, row in enumerate(rows.itertuples(index=False)):
            predicted_parent = fine_ids[int(predicted[index])]
            true_parent = str(row.parent_id)
            same_macro = int(fixed_parent[predicted_parent] == fixed_parent[true_parent])
            upper_correct.append(same_macro)
            predictions.append(
                {
                    "split": str(row.split),
                    "child_id": str(row.child_id),
                    "true_parent_id": true_parent,
                    "predicted_parent_id": predicted_parent,
                    "rank": int(ranks[index]),
                    "confidence": float(probabilities[index, predicted[index]]),
                    "correct": int(predicted_parent == true_parent),
                    "same_upper_parent": same_macro,
                }
            )
        metrics["upper_parent_accuracy"] = float(np.mean(upper_correct))
        return metrics, predictions

    train_frame = edge_frame[edge_frame["split"].astype(str) == "train"].reset_index(drop=True)
    val_frame = edge_frame[edge_frame["split"].astype(str) == "val"].reset_index(drop=True)
    if train_frame.empty or val_frame.empty:
        raise ValueError("Training and validation rows are required")

    model.train()
    sample = train_frame.iloc[: min(2, len(train_frame))]
    rows = torch.as_tensor(sample["node_row"].to_numpy(dtype=np.int64), device=device)
    targets = torch.as_tensor(sample["fine_target"].to_numpy(dtype=np.int64), device=device)
    fine_logits, macro_logits = model(
        tensors["text"][rows],
        tensors["image"][rows],
        tensors["text_mask"][rows],
        tensors["image_mask"][rows],
        tensors.get("group_image", tensors["image"])[rows],
        tensors.get("image_delta", torch.zeros_like(tensors["image"]))[rows],
    )
    preflight_loss = F.cross_entropy(fine_logits, targets)
    if macro_logits is not None:
        macro_targets = torch.as_tensor(
            sample["macro_target"].to_numpy(dtype=np.int64), device=device
        )
        preflight_loss = preflight_loss + lambda_macro * F.cross_entropy(
            macro_logits, macro_targets
        )
    preflight_loss.backward()
    if not torch.isfinite(preflight_loss):
        raise FloatingPointError("Supervised baseline preflight loss is not finite")
    model.zero_grad(set_to_none=True)
    write_json(
        output_dir / "preflight.json",
        {
            "status": "passed",
            "seed": seed,
            "mode": mode,
            "embedding_dim": bundle.dim,
            "one_batch_loss": float(preflight_loss.detach()),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    )
    set_seed(seed)
    rng = np.random.default_rng(seed)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_score = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics, _ = forward_frame(train_frame, training=True)
        with torch.no_grad():
            val_metrics, _ = forward_frame(val_frame, training=False)
        selection_score = (
            selection_mrr_weight * float(val_metrics["mrr"])
            + selection_hits1_weight * float(val_metrics["hits@1"])
            + selection_macro_f1_weight
            * float(val_metrics["fine_macro_f1_50_classes"])
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
            f"mode={mode} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_mrr={val_metrics['mrr']:.4f} "
            f"score={selection_score:.4f}"
        )
        if selection_score > best_val_score + 1e-6:
            best_val_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Supervised baseline did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "mode": mode,
            "seed": seed,
            "node_ids": node_ids,
            "fine_ids": fine_ids,
        },
        output_dir / "best_model.pt",
    )
    write_json(output_dir / "history.json", history)

    metrics: dict[str, dict[str, float | int]] = {}
    predictions: list[dict[str, Any]] = []
    evaluation_splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    with torch.no_grad():
        for split in evaluation_splits:
            split_frame = edge_frame[edge_frame["split"].astype(str) == split].reset_index(
                drop=True
            )
            split_metrics, split_predictions = forward_frame(split_frame, training=False)
            metrics[split] = split_metrics
            predictions.extend(split_predictions)
    write_json(output_dir / "metrics.json", metrics)
    prediction_frame = pd.DataFrame(predictions).sort_values(["split", "child_id"])
    prediction_frame.to_csv(
        output_dir / "predicted_edges.csv", index=False, encoding="utf-8-sig"
    )
    supplementary = None
    if evaluate_test:
        fixed = tables.edges[tables.edges["is_fixed"].astype(int) == 1][
            ["parent_id", "child_id", "relation"]
        ].copy()
        predicted_edges = prediction_frame[["predicted_parent_id", "child_id"]].rename(
            columns={"predicted_parent_id": "parent_id"}
        )
        predicted_edges["relation"] = "predicted_parent"
        pd.concat([fixed, predicted_edges], ignore_index=True).to_csv(
            output_dir / "constructed_tree_edges.csv", index=False, encoding="utf-8-sig"
        )
        supplementary = evaluate(data_dir, output_dir)
    run_info = {
        "method": mode,
        "seed": seed,
        "embedding_dim": bundle.dim,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "embeddings_path": str(embeddings_path.resolve()),
        "data_dir": str(data_dir.resolve()),
        "selection_split": "val",
        "selection_metric": "weighted_mrr_hits1_macro_f1",
        "selection_weights": {
            "mrr": selection_mrr_weight,
            "hits@1": selection_hits1_weight,
            "macro_f1": selection_macro_f1_weight,
        },
        "lambda_macro": lambda_macro if mode == "hierarchical_gated" else 0.0,
        "macro_score_weight": macro_score_weight,
        "use_group_context": use_group_context,
        "group_context": group_context_stats,
        "group_consensus_weight": group_consensus_weight,
        "test_evaluated": evaluate_test,
        "structure_valid": (
            bool(supplementary["structure"]["is_valid_tree"])
            if supplementary is not None
            else None
        ),
    }
    write_json(output_dir / "run_info.json", run_info)
    print(
        json.dumps(
            {
                "run_info": run_info,
                "test": metrics.get("test"),
                "val": metrics["val"],
            },
            indent=2,
        )
    )
    return {"run_info": run_info, "metrics": metrics, "supplementary": supplementary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched supervised parent classifiers")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["linear_image", "gated_mlp", "hierarchical_gated"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lambda-macro", type=float, default=0.5)
    parser.add_argument("--macro-score-weight", type=float, default=1.0)
    parser.add_argument("--use-group-context", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--selection-mrr-weight", type=float, default=1.0)
    parser.add_argument("--selection-hits1-weight", type=float, default=0.0)
    parser.add_argument("--selection-macro-f1-weight", type=float, default=0.0)
    parser.add_argument("--group-consensus-weight", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_supervised_baseline(
        data_dir=args.data_dir,
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        device_name=args.device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lambda_macro=args.lambda_macro,
        macro_score_weight=args.macro_score_weight,
        use_group_context=args.use_group_context,
        evaluate_test=not args.validation_only,
        selection_mrr_weight=args.selection_mrr_weight,
        selection_hits1_weight=args.selection_hits1_weight,
        selection_macro_f1_weight=args.selection_macro_f1_weight,
        group_consensus_weight=args.group_consensus_weight,
    )


if __name__ == "__main__":
    main()
