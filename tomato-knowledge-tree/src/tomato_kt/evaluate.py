from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .schema import load_tables
from .utils import write_json

REQUIRED_PREDICTION_COLUMNS = {
    "split",
    "child_id",
    "true_parent_id",
    "predicted_parent_id",
    "rank",
}


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float | None:
    if not y_true or not labels:
        return None
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def _prediction_metrics(
    frame: pd.DataFrame,
    *,
    reference_fine_labels: list[str],
    macro_labels: list[str],
    fine_to_macro: dict[str, str],
) -> dict[str, Any]:
    count = len(frame)
    if count == 0:
        return {
            "count": 0,
            "hits_at_1": None,
            "hits_at_3": None,
            "hits_at_5": None,
            "mrr": None,
            "fine_accuracy": None,
            "fine_macro_f1_reference_classes": None,
            "fine_macro_f1_observed_classes": None,
            "macro_stage_accuracy": None,
            "macro_stage_macro_f1": None,
            "invalid_parent_level_rate": None,
        }

    true_fine = frame["true_parent_id"].astype(str).tolist()
    pred_fine = frame["predicted_parent_id"].astype(str).tolist()
    ranks = pd.to_numeric(frame["rank"], errors="raise").to_numpy(dtype=np.float64)
    if np.any(ranks < 1):
        raise ValueError("All prediction ranks must be positive integers")

    missing_true = sorted(set(true_fine) - set(fine_to_macro))
    if missing_true:
        raise ValueError(f"True fine-stage nodes missing fixed macro parents: {missing_true[:10]}")

    true_macro = [fine_to_macro[node_id] for node_id in true_fine]
    invalid_macro = "__invalid_parent_level__"
    pred_macro = [fine_to_macro.get(node_id, invalid_macro) for node_id in pred_fine]
    observed_fine = sorted(set(true_fine) | set(pred_fine))

    return {
        "count": count,
        "hits_at_1": float(np.mean(ranks <= 1)),
        "hits_at_3": float(np.mean(ranks <= 3)),
        "hits_at_5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "fine_accuracy": float(np.mean(np.asarray(true_fine) == np.asarray(pred_fine))),
        "fine_macro_f1_reference_classes": _macro_f1(
            true_fine, pred_fine, reference_fine_labels
        ),
        "fine_macro_f1_observed_classes": _macro_f1(
            true_fine, pred_fine, observed_fine
        ),
        "macro_stage_accuracy": float(
            np.mean(np.asarray(true_macro) == np.asarray(pred_macro))
        ),
        "macro_stage_macro_f1": _macro_f1(true_macro, pred_macro, macro_labels),
        "invalid_parent_level_rate": float(
            np.mean([node_id not in fine_to_macro for node_id in pred_fine])
        ),
    }


def _fixed_parent_map(edges: pd.DataFrame, fine_ids: set[str]) -> dict[str, str]:
    fixed = edges[edges["is_fixed"].astype(int) == 1]
    parent_map: dict[str, str] = {}
    for row in fixed.itertuples(index=False):
        child_id = str(row.child_id)
        if child_id not in fine_ids:
            continue
        if child_id in parent_map:
            raise ValueError(f"Fine-stage node has multiple fixed parents: {child_id}")
        parent_map[child_id] = str(row.parent_id)
    missing = sorted(fine_ids - set(parent_map))
    if missing:
        raise ValueError(f"Fine-stage nodes missing fixed parents: {missing[:10]}")
    return parent_map


def _per_class_metrics(
    predictions: pd.DataFrame,
    nodes: pd.DataFrame,
    fine_ids: list[str],
    fine_to_macro: dict[str, str],
) -> pd.DataFrame:
    name_by_id = dict(zip(nodes["node_id"].astype(str), nodes["name"].astype(str)))
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        frame = predictions[predictions["split"].astype(str) == split]
        y_true = frame["true_parent_id"].astype(str).tolist()
        y_pred = frame["predicted_parent_id"].astype(str).tolist()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=fine_ids,
            zero_division=0,
        )
        predicted_support = pd.Series(y_pred).value_counts()
        for index, node_id in enumerate(fine_ids):
            macro_id = fine_to_macro[node_id]
            rows.append(
                {
                    "split": split,
                    "parent_id": node_id,
                    "parent_name": name_by_id[node_id],
                    "macro_parent_id": macro_id,
                    "macro_parent_name": name_by_id[macro_id],
                    "true_support": int(support[index]),
                    "predicted_support": int(predicted_support.get(node_id, 0)),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                }
            )
    return pd.DataFrame(rows)


def _structure_report(
    nodes: pd.DataFrame,
    reference_edges: pd.DataFrame,
    predictions: pd.DataFrame,
    constructed_edges: pd.DataFrame,
) -> dict[str, Any]:
    node_ids = set(nodes["node_id"].astype(str))
    item_ids = set(
        nodes.loc[nodes["level"].astype(int) == 4, "node_id"].astype(str)
    )
    fine_ids = set(
        nodes.loc[nodes["level"].astype(int) == 3, "node_id"].astype(str)
    )
    level_by_id = dict(zip(nodes["node_id"].astype(str), nodes["level"].astype(int)))

    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(
        zip(
            constructed_edges["parent_id"].astype(str),
            constructed_edges["child_id"].astype(str),
        )
    )

    constructed_pairs = list(
        zip(
            constructed_edges["parent_id"].astype(str),
            constructed_edges["child_id"].astype(str),
        )
    )
    unknown_refs = sorted(
        {
            node_id
            for pair in constructed_pairs
            for node_id in pair
            if node_id not in node_ids
        }
    )
    roots = sorted(node_id for node_id in node_ids if graph.in_degree(node_id) == 0)
    non_roots = node_ids - set(roots)
    previous_level_edges = sum(
        1
        for parent_id, child_id in constructed_pairs
        if parent_id in level_by_id
        and child_id in level_by_id
        and level_by_id[child_id] == level_by_id[parent_id] + 1
    )

    fixed_reference = {
        (str(row.parent_id), str(row.child_id))
        for row in reference_edges.itertuples(index=False)
        if int(row.is_fixed) == 1
    }
    fixed_constructed = {
        (str(row.parent_id), str(row.child_id))
        for row in constructed_edges.itertuples(index=False)
        if str(row.relation) != "predicted_parent"
    }

    child_counts = predictions["child_id"].astype(str).value_counts()
    valid_image_ids: set[str] = set()
    for row in predictions.itertuples(index=False):
        child_id = str(row.child_id)
        parent_id = str(row.predicted_parent_id)
        if (
            child_counts.get(child_id, 0) == 1
            and child_id in item_ids
            and parent_id in fine_ids
        ):
            valid_image_ids.add(child_id)

    checks = {
        "all_edge_nodes_known": not unknown_refs,
        "no_duplicate_edges": len(constructed_pairs) == len(set(constructed_pairs)),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "is_weakly_connected": nx.is_weakly_connected(graph) if node_ids else False,
        "single_root": roots == ["tomato"],
        "every_non_root_has_one_parent": all(
            graph.in_degree(node_id) == 1 for node_id in non_roots
        ),
        "all_edges_connect_adjacent_levels": previous_level_edges
        == len(constructed_pairs),
        "fixed_edges_preserved": fixed_constructed == fixed_reference,
        "all_image_nodes_predicted_once": set(child_counts.index) == item_ids
        and bool((child_counts == 1).all()),
        "edge_count_is_nodes_minus_one": len(constructed_pairs) == len(node_ids) - 1,
    }

    return {
        "node_count": len(node_ids),
        "edge_count": len(constructed_pairs),
        "fixed_edge_count": len(fixed_constructed),
        "predicted_edge_count": int(
            (constructed_edges["relation"].astype(str) == "predicted_parent").sum()
        ),
        "root_ids": roots,
        "unknown_node_references": unknown_refs,
        "valid_image_node_count": len(valid_image_ids),
        "image_node_count": len(item_ids),
        "structure_validity_rate": (
            float(len(valid_image_ids) / len(item_ids)) if item_ids else None
        ),
        "global_checks": checks,
        "global_check_pass_rate": float(np.mean(list(checks.values()))),
        "is_valid_tree": bool(all(checks.values())),
    }


def evaluate(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    tables = load_tables(data_dir)
    prediction_path = output_dir / "predicted_edges.csv"
    tree_path = output_dir / "constructed_tree_edges.csv"
    if not prediction_path.exists() or not tree_path.exists():
        raise FileNotFoundError(
            "Formal prediction outputs are required: predicted_edges.csv and "
            "constructed_tree_edges.csv"
        )

    predictions = pd.read_csv(prediction_path)
    constructed_edges = pd.read_csv(tree_path)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file missing columns: {missing}")
    if not {"parent_id", "child_id", "relation"}.issubset(constructed_edges.columns):
        raise ValueError("Constructed tree file has an invalid schema")

    nodes = tables.nodes.copy()
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
    fine_to_macro = _fixed_parent_map(tables.edges, set(fine_ids))
    train_classes = sorted(
        set(
            predictions.loc[
                predictions["split"].astype(str) == "train", "true_parent_id"
            ].astype(str)
        )
    )

    split_reports: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        frame = predictions[predictions["split"].astype(str) == split].copy()
        seen_mask = frame["true_parent_id"].astype(str).isin(train_classes)
        unseen_classes = sorted(
            set(frame.loc[~seen_mask, "true_parent_id"].astype(str))
        )
        split_reports[split] = {
            "overall": _prediction_metrics(
                frame,
                reference_fine_labels=fine_ids,
                macro_labels=macro_ids,
                fine_to_macro=fine_to_macro,
            ),
            "training_seen_classes": _prediction_metrics(
                frame[seen_mask],
                reference_fine_labels=train_classes,
                macro_labels=macro_ids,
                fine_to_macro=fine_to_macro,
            ),
            "training_unseen_classes": {
                "class_ids": unseen_classes,
                **_prediction_metrics(
                    frame[~seen_mask],
                    reference_fine_labels=unseen_classes,
                    macro_labels=macro_ids,
                    fine_to_macro=fine_to_macro,
                ),
            },
        }

    structure = _structure_report(
        nodes, tables.edges, predictions, constructed_edges
    )
    report = {
        "metadata": {
            "source_predictions": str(prediction_path),
            "source_tree": str(tree_path),
            "candidate_fine_stage_count": len(fine_ids),
            "candidate_macro_stage_count": len(macro_ids),
            "training_seen_fine_stage_count": len(train_classes),
            "metric_definitions": {
                "fine_macro_f1_reference_classes": (
                    "Macro-F1 over all 50 candidate fine stages for overall rows, "
                    "training-seen stages for seen rows, and the true unseen stages "
                    "for unseen rows."
                ),
                "fine_macro_f1_observed_classes": (
                    "Macro-F1 over the union of true and predicted labels present "
                    "in the evaluated subset."
                ),
                "structure_validity_rate": (
                    "Fraction of level-4 image nodes having exactly one prediction "
                    "from a known level-3 fine-stage parent."
                ),
                "invalid_parent_level_rate": (
                    "Fraction of predictions whose parent is not a level-3 "
                    "fine-stage node."
                ),
            },
        },
        "splits": split_reports,
        "structure": structure,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "supplementary_metrics.json", report)
    write_json(output_dir / "structure_validation.json", structure)
    _per_class_metrics(
        predictions, nodes, fine_ids, fine_to_macro
    ).to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute supplementary metrics from an existing formal run."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/tomatomap_kt5k")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/tomatomap_qwen3vl")
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate(args.data_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
