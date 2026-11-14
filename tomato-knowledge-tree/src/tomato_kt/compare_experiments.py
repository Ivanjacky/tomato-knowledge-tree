from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import load_tables
from .utils import write_json

SUMMARY_METRICS = {
    "hits_at_1": ("test", "hits_at_1"),
    "hits_at_3": ("test", "hits_at_3"),
    "hits_at_5": ("test", "hits_at_5"),
    "mrr": ("test", "mrr"),
    "macro_stage_accuracy": ("test", "macro_stage_accuracy"),
    "macro_stage_macro_f1": ("test", "macro_stage_macro_f1"),
    "fine_macro_f1_50_classes": ("test", "fine_macro_f1_50_classes"),
    "invalid_parent_level_rate": ("test", "invalid_parent_level_rate"),
    "structure_validity_rate": ("all", "structure_validity_rate"),
}
SIGNIFICANCE_METRICS = ("hits_at_1", "mrr", "macro_stage_accuracy")
DESCRIPTIONS = {
    "full": "完整模型",
    "euclidean": "移除双曲层级建模（欧氏空间且无双曲半径损失）",
    "mean_fusion": "门控融合替换为均值融合",
    "no_depth": "移除双曲半径深度顺序损失",
    "no_align": "移除图文投影对齐损失",
    "symmetric_score": "方向性父子评分替换为对称距离评分",
    "no_item_text": "移除第4层图像节点的相机与姿态文本",
    "relaxed_candidates": "上一层候选约束放宽为任意祖先层候选",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _parse_experiment(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise argparse.ArgumentTypeError("Experiment must use NAME=AGGREGATE_DIR syntax")
    return name, Path(path_text)


def _zero_summary(seed_count: int) -> dict[str, float | int]:
    return {"n": seed_count, "mean": 0.0, "std_sample": 0.0, "min": 0.0, "max": 0.0}


def _summary_metric(
    report: dict[str, Any], split: str, metric: str
) -> dict[str, float | int]:
    value = report.get("summary", {}).get(split, {}).get(metric)
    if value is not None:
        return value
    if metric == "invalid_parent_level_rate" and all(
        bool(run.get("is_valid_tree", False)) for run in report["runs"].values()
    ):
        return _zero_summary(len(report["runs"]))
    raise ValueError(f"Aggregate report is missing {split}/{metric}")


def _resolve_run_dir(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _fine_to_macro(data_dir: Path) -> dict[str, str]:
    tables = load_tables(data_dir)
    fine_ids = set(
        tables.nodes.loc[
            tables.nodes["level"].astype(int) == 3, "node_id"
        ].astype(str)
    )
    mapping: dict[str, str] = {}
    for row in tables.edges[tables.edges["is_fixed"].astype(int) == 1].itertuples(
        index=False
    ):
        child_id = str(row.child_id)
        if child_id in fine_ids:
            if child_id in mapping:
                raise ValueError(f"Fine-stage node has multiple parents: {child_id}")
            mapping[child_id] = str(row.parent_id)
    if set(mapping) != fine_ids:
        raise ValueError("Every fine-stage node must have one fixed macro parent")
    return mapping


def _plant_metrics(
    report: dict[str, Any],
    manifest: pd.DataFrame,
    fine_to_macro: dict[str, str],
    project_root: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed_text, run in sorted(report["runs"].items(), key=lambda item: int(item[0])):
        run_dir = _resolve_run_dir(str(run["output_dir"]), project_root)
        predictions = pd.read_csv(run_dir / "predicted_edges.csv", dtype=str)
        required = {"split", "child_id", "true_parent_id", "predicted_parent_id", "rank"}
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"{run_dir} predictions are missing columns: {missing}")
        test = predictions[predictions["split"] == "test"].copy()
        test["seed"] = int(seed_text)
        test["rank"] = pd.to_numeric(test["rank"], errors="raise")
        frames.append(test)

    predictions = pd.concat(frames, ignore_index=True)
    metadata = manifest[["image_id", "plant_id"]].rename(columns={"image_id": "child_id"})
    predictions = predictions.merge(metadata, on="child_id", how="left", validate="many_to_one")
    if predictions["plant_id"].isna().any():
        raise ValueError("At least one test prediction has no plant_id in manifest.csv")
    predictions["hits_at_1"] = (
        predictions["true_parent_id"] == predictions["predicted_parent_id"]
    ).astype(float)
    predictions["mrr"] = 1.0 / predictions["rank"].astype(float)
    true_macro = predictions["true_parent_id"].map(fine_to_macro)
    pred_macro = predictions["predicted_parent_id"].map(fine_to_macro)
    if true_macro.isna().any():
        raise ValueError("A true parent is not a level-3 fine-stage node")
    predictions["macro_stage_accuracy"] = (true_macro == pred_macro).astype(float)
    return (
        predictions.groupby("plant_id", as_index=False)[list(SIGNIFICANCE_METRICS)]
        .mean()
        .sort_values("plant_id")
        .reset_index(drop=True)
    )


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or len(differences) == 0:
        raise ValueError("Paired differences must be a non-empty one-dimensional array")
    if len(differences) > 20:
        raise ValueError("Exact sign-flip test is limited to at most 20 paired units")
    observed = abs(float(differences.mean()))
    assignments = np.arange(1 << len(differences), dtype=np.uint32)[:, None]
    bits = (assignments >> np.arange(len(differences), dtype=np.uint32)) & 1
    signs = 1.0 - 2.0 * bits.astype(np.float64)
    permuted = np.abs((signs * differences[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-12))


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[int(original_index)])
        running = max(running, candidate)
        adjusted[int(original_index)] = running
    return adjusted.tolist()


def compare_experiments(
    experiments: dict[str, Path],
    reference: str,
    data_dir: Path,
    output_dir: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    if reference not in experiments:
        raise ValueError(f"Reference experiment is absent: {reference}")
    if len(experiments) < 2:
        raise ValueError("At least two experiments are required")
    root = (project_root or Path.cwd()).resolve()
    reports = {
        name: _read_json(path / "multiseed_metrics.json")
        for name, path in experiments.items()
    }
    reference_report = reports[reference]
    reference_seeds = reference_report["metadata"]["seeds"]
    reference_truth = reference_report["metadata"]["ground_truth_sha256"]
    for name, report in reports.items():
        if report["metadata"]["seeds"] != reference_seeds:
            raise ValueError(f"{name} does not use the reference seeds")
        if report["metadata"]["ground_truth_sha256"] != reference_truth:
            raise ValueError(f"{name} does not use the reference ground-truth rows")

    reference_values = {
        metric: float(_summary_metric(reference_report, split, source)["mean"])
        for metric, (split, source) in SUMMARY_METRICS.items()
    }
    summary_rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        row: dict[str, Any] = {
            "experiment": name,
            "description": DESCRIPTIONS.get(name, name),
            "is_reference": name == reference,
            "seed_count": len(report["metadata"]["seeds"]),
        }
        for metric, (split, source) in SUMMARY_METRICS.items():
            metric_summary = _summary_metric(report, split, source)
            row[f"{metric}_mean"] = float(metric_summary["mean"])
            row[f"{metric}_std_sample"] = float(metric_summary["std_sample"])
            row[f"{metric}_delta_vs_{reference}"] = (
                float(metric_summary["mean"]) - reference_values[metric]
            )
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)

    manifest = pd.read_csv(data_dir / "manifest.csv", dtype={"image_id": str, "plant_id": str})
    missing_manifest = sorted({"image_id", "plant_id"} - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"manifest.csv is missing columns: {missing_manifest}")
    fine_to_macro = _fine_to_macro(data_dir)
    plant_frames = {
        name: _plant_metrics(report, manifest, fine_to_macro, root)
        for name, report in reports.items()
    }
    reference_plants = plant_frames[reference]
    significance_rows: list[dict[str, Any]] = []
    for name, frame in plant_frames.items():
        if name == reference:
            continue
        paired = reference_plants.merge(
            frame,
            on="plant_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_reference", "_ablation"),
        )
        if len(paired) != len(reference_plants) or len(paired) != len(frame):
            raise ValueError(f"{name} does not contain the same test plants as {reference}")
        for metric in SIGNIFICANCE_METRICS:
            differences = (
                paired[f"{metric}_ablation"] - paired[f"{metric}_reference"]
            ).to_numpy(dtype=np.float64)
            significance_rows.append(
                {
                    "experiment": name,
                    "reference": reference,
                    "metric": metric,
                    "paired_unit": "plant_id",
                    "plant_count": len(paired),
                    "ablation_minus_reference_mean": float(differences.mean()),
                    "p_value_exact_two_sided": exact_sign_flip_pvalue(differences),
                }
            )
    significance_frame = pd.DataFrame(significance_rows)
    adjusted = holm_adjust(significance_frame["p_value_exact_two_sided"].tolist())
    significance_frame["p_value_holm"] = adjusted
    significance_frame["significant_at_0_05"] = significance_frame["p_value_holm"] < 0.05

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(
        output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig"
    )
    significance_frame.to_csv(
        output_dir / "ablation_significance.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "metadata": {
            "reference": reference,
            "experiments": list(experiments),
            "seeds": reference_seeds,
            "ground_truth_sha256": reference_truth,
            "significance_test": (
                "Exact two-sided paired sign-flip test on per-plant metrics; each plant "
                "metric averages all its test images and all five seeds."
            ),
            "multiple_testing_correction": "Holm correction over all ablation-metric tests",
        },
        "summary": json.loads(summary_frame.to_json(orient="records")),
        "significance": json.loads(significance_frame.to_json(orient="records")),
    }
    write_json(output_dir / "ablation_comparison.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multi-seed ablations with a paired plant-level test."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        type=_parse_experiment,
        required=True,
        metavar="NAME=AGGREGATE_DIR",
    )
    parser.add_argument("--reference", default="full")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/tomatomap_kt5k")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ablations/summary")
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    experiments = dict(args.experiment)
    if len(experiments) != len(args.experiment):
        raise ValueError("Each experiment name may be provided only once")
    report = compare_experiments(
        experiments, args.reference, args.data_dir, args.output_dir
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
