from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import load_tables
from .utils import write_json

TRUTH_COLUMNS = ["child_id", "split", "true_parent_id"]


def _sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _parse_run_spec(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text or not path_text:
        raise argparse.ArgumentTypeError("Run must use SEED=OUTPUT_DIR syntax")
    try:
        return int(seed_text), Path(path_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed: {seed_text}") from exc


def _fine_code(node_id: str) -> int | None:
    prefix = "bbch_"
    if not node_id.startswith(prefix):
        return None
    try:
        return int(node_id.removeprefix(prefix))
    except ValueError:
        return None


def _rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2:
        return None
    left_rank = left.rank(method="average").to_numpy(dtype=np.float64)
    right_rank = right.rank(method="average").to_numpy(dtype=np.float64)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return strict-JSON-compatible records, converting NaN values to null."""
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _metric_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_columns = [
        "hits_at_1",
        "hits_at_3",
        "hits_at_5",
        "mrr",
        "macro_stage_accuracy",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        row["items_per_seed"] = int(group["count"].iloc[0])
        if group["count"].nunique() != 1:
            raise ValueError(f"Group has inconsistent item counts across seeds: {row}")
        row["seed_count"] = int(group["seed"].nunique())
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std_sample"] = _sample_std(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_runs(
    runs: dict[int, Path], data_dir: Path, output_dir: Path
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("At least two runs are required for cross-seed analysis")
    tables = load_tables(data_dir)
    manifest = pd.read_csv(data_dir / "manifest.csv", dtype={"image_id": str})
    required_manifest = {"image_id", "pose_id", "pose_degrees", "camera_label"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"manifest.csv is missing columns: {missing_manifest}")

    nodes = tables.nodes.copy()
    node_name = dict(zip(nodes["node_id"].astype(str), nodes["name"].astype(str)))
    fine_ids = set(
        nodes.loc[nodes["level"].astype(int) == 3, "node_id"].astype(str)
    )
    fine_to_macro = {
        str(row.child_id): str(row.parent_id)
        for row in tables.edges[tables.edges["is_fixed"].astype(int) == 1].itertuples(
            index=False
        )
        if str(row.child_id) in fine_ids
    }
    if set(fine_to_macro) != fine_ids:
        raise ValueError("Every fine-stage node must have exactly one fixed macro parent")

    prediction_frames: list[pd.DataFrame] = []
    class_frames: list[pd.DataFrame] = []
    reference_truth: pd.DataFrame | None = None
    for seed, run_dir in sorted(runs.items()):
        predictions = pd.read_csv(run_dir / "predicted_edges.csv", dtype=str)
        missing = sorted(set(TRUTH_COLUMNS + ["predicted_parent_id", "rank"]) - set(predictions))
        if missing:
            raise ValueError(f"Seed {seed} predictions are missing columns: {missing}")
        truth = predictions[TRUTH_COLUMNS].sort_values("child_id").reset_index(drop=True)
        if reference_truth is None:
            reference_truth = truth
        elif not truth.equals(reference_truth):
            raise ValueError(f"Seed {seed} does not use identical truth rows and splits")
        predictions["seed"] = seed
        predictions["rank"] = pd.to_numeric(predictions["rank"], errors="raise")
        prediction_frames.append(predictions)

        per_class = pd.read_csv(run_dir / "per_class_metrics.csv", dtype={"parent_id": str})
        per_class = per_class[per_class["split"].astype(str) == "test"].copy()
        per_class["seed"] = seed
        class_frames.append(per_class)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    test = all_predictions[all_predictions["split"].astype(str) == "test"].copy()
    test["true_macro_id"] = test["true_parent_id"].map(fine_to_macro)
    test["predicted_macro_id"] = test["predicted_parent_id"].map(fine_to_macro)
    if test[["true_macro_id", "predicted_macro_id"]].isna().any().any():
        raise ValueError("Predictions contain unknown fine-stage nodes")

    class_all = pd.concat(class_frames, ignore_index=True)
    support_consistency = class_all.groupby("parent_id")["true_support"].nunique()
    if bool((support_consistency > 1).any()):
        raise ValueError("Per-class test support differs across seeds")
    class_summary = (
        class_all.groupby(
            ["parent_id", "parent_name", "macro_parent_id", "macro_parent_name"],
            as_index=False,
            sort=True,
        )
        .agg(
            true_support=("true_support", "first"),
            predicted_support_mean=("predicted_support", "mean"),
            predicted_support_std_sample=("predicted_support", _sample_std),
            precision_mean=("precision", "mean"),
            precision_std_sample=("precision", _sample_std),
            recall_mean=("recall", "mean"),
            recall_std_sample=("recall", _sample_std),
            f1_mean=("f1", "mean"),
            f1_std_sample=("f1", _sample_std),
        )
    )
    supported_classes = class_summary[class_summary["true_support"] > 0].copy()
    support_f1_spearman = _rank_correlation(
        supported_classes["true_support"], supported_classes["f1_mean"]
    )

    errors = test[test["true_parent_id"] != test["predicted_parent_id"]].copy()
    confusion = (
        errors.groupby(["true_parent_id", "predicted_parent_id"], as_index=False)
        .size()
        .rename(columns={"size": "total_count_across_seeds"})
        .sort_values(
            ["total_count_across_seeds", "true_parent_id", "predicted_parent_id"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    confusion["mean_count_per_seed"] = confusion["total_count_across_seeds"] / len(runs)
    confusion["error_share"] = confusion["total_count_across_seeds"] / max(len(errors), 1)
    confusion["true_name"] = confusion["true_parent_id"].map(node_name)
    confusion["predicted_name"] = confusion["predicted_parent_id"].map(node_name)
    confusion["true_macro_id"] = confusion["true_parent_id"].map(fine_to_macro)
    confusion["predicted_macro_id"] = confusion["predicted_parent_id"].map(fine_to_macro)
    confusion["same_macro_stage"] = (
        confusion["true_macro_id"] == confusion["predicted_macro_id"]
    )
    confusion["fine_stage_distance"] = [
        abs(true_code - pred_code)
        if same_macro and true_code is not None and pred_code is not None
        else np.nan
        for true_code, pred_code, same_macro in zip(
            confusion["true_parent_id"].map(_fine_code),
            confusion["predicted_parent_id"].map(_fine_code),
            confusion["same_macro_stage"],
        )
    ]

    manifest_meta = manifest[
        ["image_id", "pose_id", "pose_degrees", "camera_label"]
    ].rename(columns={"image_id": "child_id"})
    test = test.merge(manifest_meta, on="child_id", how="left", validate="many_to_one")
    if test["pose_id"].isna().any():
        raise ValueError("At least one test prediction is absent from manifest.csv")
    pose_rows: list[dict[str, Any]] = []
    for (seed, pose_id, pose_degrees), group in test.groupby(
        ["seed", "pose_id", "pose_degrees"], sort=True
    ):
        ranks = group["rank"].to_numpy(dtype=np.float64)
        pose_rows.append(
            {
                "seed": int(seed),
                "pose_id": int(pose_id),
                "pose_degrees": int(pose_degrees),
                "count": len(group),
                "hits_at_1": float(np.mean(ranks <= 1)),
                "hits_at_3": float(np.mean(ranks <= 3)),
                "hits_at_5": float(np.mean(ranks <= 5)),
                "mrr": float(np.mean(1.0 / ranks)),
                "macro_stage_accuracy": float(
                    np.mean(group["true_macro_id"] == group["predicted_macro_id"])
                ),
            }
        )
    pose_summary = _metric_summary(
        pd.DataFrame(pose_rows), ["pose_id", "pose_degrees"]
    )

    stability_rows: list[dict[str, Any]] = []
    for (child_id, true_parent_id), group in test.groupby(
        ["child_id", "true_parent_id"], sort=True
    ):
        predictions = group["predicted_parent_id"].astype(str).tolist()
        if len(predictions) != len(runs) or group["seed"].nunique() != len(runs):
            raise ValueError(f"Incomplete seed predictions for {child_id}")
        counts = Counter(predictions)
        max_count = max(counts.values())
        consensus = min(label for label, count in counts.items() if count == max_count)
        macro_predictions = [fine_to_macro[label] for label in predictions]
        macro_counts = Counter(macro_predictions)
        macro_max_count = max(macro_counts.values())
        metadata = group.iloc[0]
        stability_rows.append(
            {
                "child_id": child_id,
                "true_parent_id": true_parent_id,
                "true_parent_name": node_name[true_parent_id],
                "pose_id": int(metadata["pose_id"]),
                "pose_degrees": int(metadata["pose_degrees"]),
                "unique_fine_predictions": len(counts),
                "fine_agreement_rate": max_count / len(runs),
                "unique_macro_predictions": len(macro_counts),
                "macro_agreement_rate": macro_max_count / len(runs),
                "correct_seed_count": int(sum(label == true_parent_id for label in predictions)),
                "consensus_parent_id": consensus,
                "consensus_parent_name": node_name[consensus],
                "consensus_correct": int(consensus == true_parent_id),
            }
        )
    stability = pd.DataFrame(stability_rows)

    same_macro_errors = errors["true_macro_id"] == errors["predicted_macro_id"]
    true_codes = errors["true_parent_id"].map(_fine_code)
    pred_codes = errors["predicted_parent_id"].map(_fine_code)
    nearby_errors = same_macro_errors & (true_codes - pred_codes).abs().le(2)
    top_classes = supported_classes.sort_values(
        ["f1_mean", "true_support"], ascending=[False, False]
    ).head(10)
    bottom_classes = supported_classes.sort_values(
        ["f1_mean", "true_support"], ascending=[True, False]
    ).head(10)

    report = {
        "metadata": {
            "seeds": sorted(runs),
            "seed_count": len(runs),
            "test_items_per_seed": int(test["child_id"].nunique()),
            "test_prediction_rows": len(test),
        },
        "class_imbalance": {
            "candidate_fine_stage_count": len(class_summary),
            "test_supported_fine_stage_count": len(supported_classes),
            "test_zero_support_stage_ids": class_summary.loc[
                class_summary["true_support"] == 0, "parent_id"
            ].tolist(),
            "zero_mean_f1_supported_stage_count": int(
                ((supported_classes["f1_mean"] == 0)).sum()
            ),
            "support_vs_mean_f1_spearman": support_f1_spearman,
            "top_10_stages_by_mean_f1": top_classes[
                ["parent_id", "parent_name", "true_support", "f1_mean"]
            ].pipe(_json_records),
            "bottom_10_stages_by_mean_f1": bottom_classes[
                ["parent_id", "parent_name", "true_support", "f1_mean"]
            ].pipe(_json_records),
        },
        "hierarchical_errors": {
            "total_errors_across_seeds": len(errors),
            "same_macro_stage_error_rate": float(same_macro_errors.mean()),
            "same_macro_within_two_bbch_codes_error_rate": float(nearby_errors.mean()),
            "top_10_confusion_pairs": _json_records(confusion.head(10)),
        },
        "prediction_stability": {
            "all_seeds_same_fine_parent_rate": float(
                np.mean(stability["unique_fine_predictions"] == 1)
            ),
            "at_least_four_of_five_same_fine_parent_rate": float(
                np.mean(stability["fine_agreement_rate"] >= 0.8)
            ),
            "all_seeds_same_macro_parent_rate": float(
                np.mean(stability["unique_macro_predictions"] == 1)
            ),
            "consensus_fine_parent_accuracy": float(stability["consensus_correct"].mean()),
            "mean_correct_seed_fraction": float(
                (stability["correct_seed_count"] / len(runs)).mean()
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "error_analysis.json", report)
    class_summary.to_csv(
        output_dir / "test_stage_metrics.csv", index=False, encoding="utf-8-sig"
    )
    confusion.to_csv(
        output_dir / "test_confusion_pairs.csv", index=False, encoding="utf-8-sig"
    )
    pose_summary.to_csv(
        output_dir / "test_pose_metrics.csv", index=False, encoding="utf-8-sig"
    )
    stability.to_csv(
        output_dir / "test_prediction_stability.csv", index=False, encoding="utf-8-sig"
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze class, hierarchy, pose, and stability errors across runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_spec,
        required=True,
        metavar="SEED=OUTPUT_DIR",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/tomatomap_kt5k")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/tomatomap_qwen3vl_multiseed")
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runs = dict(args.run)
    if len(runs) != len(args.run):
        raise ValueError("Each seed may be provided only once")
    report = analyze_runs(runs, args.data_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
