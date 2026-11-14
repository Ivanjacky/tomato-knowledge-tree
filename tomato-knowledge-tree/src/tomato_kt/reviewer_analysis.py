from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F

from .model import HierarchicalTreeModel
from .schema import load_embeddings, load_tables


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _run_dirs(root: Path) -> dict[int, Path]:
    runs = {42: root / "outputs" / "tomatomap_qwen3vl"}
    runs.update(
        {
            seed: root / "outputs" / f"tomatomap_qwen3vl_seed{seed}"
            for seed in (43, 44, 45, 46)
        }
    )
    for seed, path in runs.items():
        if not (path / "predicted_edges.csv").exists():
            raise FileNotFoundError(f"Missing formal run for seed {seed}: {path}")
    return runs


def analyze_errors_and_uncertainty(
    *,
    root: Path,
    output_dir: Path,
    runs: dict[int, Path] | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    data_dir = root / "data" / "processed" / "tomatomap_kt5k"
    tables = load_tables(data_dir)
    fine_ids = sorted(
        tables.nodes.loc[tables.nodes["level"].astype(int) == 3, "node_id"].astype(str),
        key=lambda value: int(value.removeprefix("bbch_")),
    )
    order = {node_id: index for index, node_id in enumerate(fine_ids)}
    stage_codes = [int(node_id.removeprefix("bbch_")) for node_id in fine_ids]
    run_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    confusion_counts = np.zeros((len(fine_ids), len(fine_ids)), dtype=np.int64)
    for seed, run_dir in (runs or _run_dirs(root)).items():
        predictions = pd.read_csv(run_dir / "predicted_edges.csv")
        test = predictions[predictions["split"].astype(str) == "test"].copy()
        true_ids = test["true_parent_id"].astype(str).tolist()
        predicted_ids = test["predicted_parent_id"].astype(str).tolist()
        supported = sorted(set(true_ids), key=lambda value: order[value])
        distances = np.asarray(
            [abs(order[truth] - order[prediction]) for truth, prediction in zip(true_ids, predicted_ids, strict=True)],
            dtype=np.int64,
        )
        correct = np.asarray(true_ids) == np.asarray(predicted_ids)
        confidence = pd.to_numeric(test["confidence"], errors="raise").to_numpy(dtype=float)
        ece = 0.0
        for left in np.linspace(0.0, 0.9, 10):
            right = left + 0.1
            mask = (confidence >= left) & (
                confidence <= right if math.isclose(right, 1.0) else confidence < right
            )
            if np.any(mask):
                ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        sorted_indices = np.argsort(-confidence)
        risks: list[float] = []
        for coverage in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
            retained = max(1, math.ceil(len(test) * coverage))
            selected = sorted_indices[:retained]
            accuracy = float(correct[selected].mean())
            risk = 1.0 - accuracy
            risks.append(risk)
            risk_rows.append(
                {
                    "seed": seed,
                    "coverage": coverage,
                    "retained_count": retained,
                    "accuracy": accuracy,
                    "risk": risk,
                    "minimum_confidence": float(confidence[selected].min()),
                }
            )
        run_rows.append(
            {
                "seed": seed,
                "test_count": len(test),
                "supported_stage_count": len(supported),
                "macro_f1_fixed_50": float(
                    f1_score(true_ids, predicted_ids, labels=fine_ids, average="macro", zero_division=0)
                ),
                "macro_f1_supported_stages": float(
                    f1_score(true_ids, predicted_ids, labels=supported, average="macro", zero_division=0)
                ),
                "mean_ordinal_stage_distance_all": float(distances.mean()),
                "median_ordinal_stage_distance_all": float(np.median(distances)),
                "mean_ordinal_stage_distance_errors": float(distances[~correct].mean()),
                "within_one_stage": float(np.mean(distances <= 1)),
                "within_two_stages": float(np.mean(distances <= 2)),
                "expected_calibration_error_10_bins": ece,
                "mean_confidence": float(confidence.mean()),
                "aurc_six_coverages": float(np.mean(risks)),
            }
        )
        confusion_counts += confusion_matrix(true_ids, predicted_ids, labels=fine_ids)

    run_frame = pd.DataFrame(run_rows)
    run_frame.to_csv(
        output_dir / f"{prefix}error_uncertainty_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_rows: list[dict[str, Any]] = []
    for column in run_frame.columns:
        if column == "seed":
            continue
        values = run_frame[column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": column,
                "n": len(values),
                "mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"{prefix}error_uncertainty_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(risk_rows).to_csv(
        output_dir / f"{prefix}risk_coverage.csv", index=False, encoding="utf-8-sig"
    )
    confusion_frame = pd.DataFrame(confusion_counts, index=stage_codes, columns=stage_codes)
    confusion_frame.index.name = "true_bbch"
    confusion_frame.to_csv(
        output_dir / f"{prefix}confusion_matrix_counts.csv", encoding="utf-8-sig"
    )
    row_sums = confusion_counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion_counts,
        row_sums,
        out=np.zeros_like(confusion_counts, dtype=float),
        where=row_sums > 0,
    )
    normalized_frame = pd.DataFrame(normalized, index=stage_codes, columns=stage_codes)
    normalized_frame.index.name = "true_bbch"
    normalized_frame.to_csv(
        output_dir / f"{prefix}confusion_matrix_row_normalized.csv", encoding="utf-8-sig"
    )
    report = {
        "seeds": run_frame["seed"].astype(int).tolist(),
        "candidate_stage_count": len(fine_ids),
        "test_supported_stage_count": int(run_frame["supported_stage_count"].iloc[0]),
        "summary_file": str(
            (output_dir / f"{prefix}error_uncertainty_summary.csv").resolve()
        ),
        "confusion_count_total": int(confusion_counts.sum()),
    }
    _write_json(output_dir / f"{prefix}error_uncertainty_report.json", report)
    return report


def _model_from_checkpoint(path: Path, device: torch.device) -> tuple[HierarchicalTreeModel, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = HierarchicalTreeModel(
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        tree_dim=int(config["tree_dim"]),
        dropout=float(config["dropout"]),
        curvature=float(config["curvature"]),
        fusion_mode=str(config["fusion_mode"]),
        geometry=str(config["geometry"]),
        score_mode=str(config.get("score_mode", "directed")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def analyze_gate_and_geometry(*, root: Path, output_dir: Path) -> dict[str, Any]:
    data_dir = root / "data" / "processed" / "tomatomap_kt5k"
    tables = load_tables(data_dir)
    nodes = tables.nodes.reset_index(drop=True)
    node_ids = nodes["node_id"].astype(str).tolist()
    row_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    bundle = load_embeddings(data_dir / "embeddings.npz", node_ids)
    text = torch.from_numpy(bundle.text)
    image = torch.from_numpy(bundle.image)
    text_mask = torch.from_numpy(bundle.text_mask)
    image_mask = torch.from_numpy(bundle.image_mask)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    pose_by_id = dict(zip(manifest["image_id"].astype(str), manifest["pose_id"].astype(int)))
    item_rows = nodes.index[nodes["level"].astype(int) == 4].to_numpy(dtype=np.int64)
    fixed_edges = tables.edges[tables.edges["is_fixed"].astype(int) == 1]
    graph = nx.Graph()
    graph.add_edges_from(
        zip(fixed_edges["parent_id"].astype(str), fixed_edges["child_id"].astype(str))
    )
    concept_ids = nodes.loc[nodes["level"].astype(int) <= 3, "node_id"].astype(str).tolist()
    graph_distances: list[float] = []
    concept_pairs: list[tuple[int, int]] = []
    for first in range(len(concept_ids)):
        for second in range(first + 1, len(concept_ids)):
            graph_distances.append(float(nx.shortest_path_length(graph, concept_ids[first], concept_ids[second])))
            concept_pairs.append((row_by_id[concept_ids[first]], row_by_id[concept_ids[second]]))

    gate_rows: list[dict[str, Any]] = []
    radius_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for seed, run_dir in _run_dirs(root).items():
        device = torch.device("cpu")
        model, config = _model_from_checkpoint(run_dir / "best_model.pt", device)
        with torch.no_grad():
            encoded = model.encode(text, image, text_mask, image_mask)
            coordinates = encoded["hyperbolic"]
            text_hidden = F.normalize(model.fusion.text_projection(text[item_rows]), dim=-1)
            image_hidden = F.normalize(model.fusion.image_projection(image[item_rows]), dim=-1)
            item_text_mask = text_mask[item_rows].float().unsqueeze(1)
            item_image_mask = image_mask[item_rows].float().unsqueeze(1)
            gates = model.fusion.gate(
                torch.cat([text_hidden, image_hidden, item_text_mask, item_image_mask], dim=-1)
            ).squeeze(1)
        curvature = float(config["curvature"])
        euclidean_radius = coordinates.norm(dim=1).cpu().numpy()
        scaled = np.clip(np.sqrt(curvature) * euclidean_radius, 0.0, 1.0 - 1e-7)
        geodesic_radius = 2.0 * np.arctanh(scaled) / np.sqrt(curvature)
        for level in sorted(nodes["level"].astype(int).unique()):
            mask = nodes["level"].astype(int).to_numpy() == level
            values = geodesic_radius[mask]
            radius_rows.append(
                {
                    "seed": seed,
                    "level": int(level),
                    "node_count": int(mask.sum()),
                    "geodesic_radius_mean": float(values.mean()),
                    "geodesic_radius_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "euclidean_radius_mean": float(euclidean_radius[mask].mean()),
                    "boundary_fraction_gt_0p99": float(
                        np.mean(euclidean_radius[mask] > 0.99 / np.sqrt(curvature))
                    ),
                }
            )
        for local, node_row in enumerate(item_rows):
            node_id = node_ids[node_row]
            gate_rows.append(
                {
                    "seed": seed,
                    "node_id": node_id,
                    "pose_id": int(pose_by_id[node_id]),
                    "text_gate_weight": float(gates[local]),
                    "image_gate_weight": float(1.0 - gates[local]),
                }
            )
        item_edges = tables.edges[tables.edges["is_fixed"].astype(int) == 0]
        parent_rows = np.asarray([row_by_id[str(value)] for value in item_edges["parent_id"]])
        child_rows = np.asarray([row_by_id[str(value)] for value in item_edges["child_id"]])
        gaps = geodesic_radius[child_rows] - geodesic_radius[parent_rows]
        model_distances: list[float] = []
        with torch.no_grad():
            for first, second in concept_pairs:
                model_distances.append(
                    float(model.ball.distance(coordinates[first : first + 1], coordinates[second : second + 1]))
                )
        graph_array = np.asarray(graph_distances)
        model_array = np.asarray(model_distances)
        scale = float(np.dot(graph_array, model_array) / max(np.dot(model_array, model_array), 1e-12))
        stress = float(
            np.sqrt(np.sum((graph_array - scale * model_array) ** 2) / np.sum(graph_array**2))
        )
        geometry_rows.append(
            {
                "seed": seed,
                "depth_margin": float(config["depth_margin"]),
                "mean_child_parent_geodesic_radius_gap": float(gaps.mean()),
                "median_child_parent_geodesic_radius_gap": float(np.median(gaps)),
                "positive_radius_order_fraction": float(np.mean(gaps > 0)),
                "margin_satisfaction_fraction": float(np.mean(gaps >= float(config["depth_margin"]))),
                "concept_graph_distance_spearman": float(spearmanr(graph_array, model_array).statistic),
                "concept_distance_normalized_stress": stress,
            }
        )

    gate_frame = pd.DataFrame(gate_rows)
    gate_frame.to_csv(output_dir / "gate_weights_by_item.csv", index=False, encoding="utf-8-sig")
    gate_summary = (
        gate_frame.groupby(["seed", "pose_id"])["text_gate_weight"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    gate_summary.to_csv(output_dir / "gate_weights_by_pose.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(radius_rows).to_csv(
        output_dir / "poincare_radii_by_level.csv", index=False, encoding="utf-8-sig"
    )
    geometry_frame = pd.DataFrame(geometry_rows)
    geometry_frame.to_csv(
        output_dir / "poincare_depth_and_distortion.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "gate_text_weight_mean": float(gate_frame["text_gate_weight"].mean()),
        "gate_text_weight_std": float(gate_frame["text_gate_weight"].std(ddof=1)),
        "positive_radius_order_fraction_mean": float(
            geometry_frame["positive_radius_order_fraction"].mean()
        ),
        "margin_satisfaction_fraction_mean": float(
            geometry_frame["margin_satisfaction_fraction"].mean()
        ),
        "concept_graph_distance_spearman_mean": float(
            geometry_frame["concept_graph_distance_spearman"].mean()
        ),
    }
    _write_json(output_dir / "gate_geometry_report.json", report)
    return report


def analyze_selection(
    *,
    dataset_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    import fiftyone as fo

    if dataset_name not in fo.list_datasets():
        raise ValueError(f"FiftyOne dataset not found: {dataset_name}")
    dataset = fo.load_dataset(dataset_name)
    view = dataset.select_group_slices(["pos_2"], flat=True)
    rows: list[dict[str, int]] = []
    for sample in view.iter_samples(progress=True):
        tags = set(sample.tags or [])
        if tags and "det" not in tags:
            continue
        rows.append(
            {
                "bbch_stage": int(sample["bbch_stage"]),
                "pose_id": int(sample["pose_id"]),
            }
        )
    frame = pd.DataFrame(rows)
    selected = frame[frame["pose_id"].isin([1, 4, 7, 10])]
    stages = sorted(frame["bbch_stage"].unique().tolist())
    full_counts = frame["bbch_stage"].value_counts()
    selected_counts = selected["bbch_stage"].value_counts()
    full_share = np.asarray([full_counts.get(stage, 0) for stage in stages], dtype=float)
    selected_share = np.asarray([selected_counts.get(stage, 0) for stage in stages], dtype=float)
    full_share /= full_share.sum()
    selected_share /= selected_share.sum()
    comparison = pd.DataFrame(
        {
            "bbch_stage": stages,
            "pos2_all_poses_count": [int(full_counts.get(stage, 0)) for stage in stages],
            "pos2_selected_poses_count": [int(selected_counts.get(stage, 0)) for stage in stages],
            "pos2_all_poses_share": full_share,
            "pos2_selected_poses_share": selected_share,
            "share_difference": selected_share - full_share,
        }
    )
    comparison.to_csv(
        output_dir / "camera_pose_selection_stage_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "dataset": dataset_name,
        "pos2_all_pose_count": len(frame),
        "pos2_selected_pose_count": len(selected),
        "stage_count": len(stages),
        "total_variation_distance": float(0.5 * np.abs(selected_share - full_share).sum()),
        "maximum_absolute_share_difference": float(np.abs(selected_share - full_share).max()),
    }
    _write_json(output_dir / "camera_pose_selection_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer-requested post-hoc analyses")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fiftyone-dataset", default="tomato-map-local")
    parser.add_argument("--skip-selection", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "errors_and_uncertainty": analyze_errors_and_uncertainty(
            root=args.root.resolve(), output_dir=output_dir
        ),
        "gate_and_geometry": analyze_gate_and_geometry(
            root=args.root.resolve(), output_dir=output_dir
        ),
    }
    if not args.skip_selection:
        report["selection"] = analyze_selection(
            dataset_name=args.fiftyone_dataset,
            output_dir=output_dir,
        )
    else:
        selection_report_path = output_dir / "camera_pose_selection_report.json"
        if selection_report_path.exists():
            with selection_report_path.open("r", encoding="utf-8") as handle:
                report["selection"] = json.load(handle)
    _write_json(output_dir / "reviewer_analysis_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
