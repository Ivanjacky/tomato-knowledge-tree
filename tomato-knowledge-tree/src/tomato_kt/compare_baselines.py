from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compare_experiments import exact_sign_flip_pvalue, holm_adjust
from .schema import load_tables
from .utils import write_json

METRICS = (
    "hits_at_1",
    "hits_at_3",
    "hits_at_5",
    "mrr",
    "upper_parent_accuracy",
    "macro_stage_accuracy",
    "macro_stage_macro_f1",
    "fine_macro_f1_50_classes",
    "structure_validity_rate",
)
SIGNIFICANCE_METRICS = ("hits_at_1", "mrr", "macro_stage_accuracy")
FORMAL_SINGLE_KEYS = {
    "hits_at_1": "hits@1",
    "hits_at_3": "hits@3",
    "hits_at_5": "hits@5",
    "mrr": "mrr",
    "upper_parent_accuracy": "upper_parent_accuracy",
}
SUPPLEMENTARY_SINGLE_KEYS = {
    "macro_stage_accuracy": "macro_stage_accuracy",
    "macro_stage_macro_f1": "macro_stage_macro_f1",
    "fine_macro_f1_50_classes": "fine_macro_f1_reference_classes",
}
DESCRIPTIONS = {
    "qwen_cosine": "Qwen3-VL冻结向量 + 余弦父节点检索",
    "clip": "OpenAI CLIP ViT-B/32 + 余弦父节点检索",
    "siglip2": "Google SigLIP2 Base/224 + 余弦父节点检索",
    "qwen_euclidean": "Qwen3-VL + 门控融合 + 欧氏父节点评分",
    "qwen_mean": "Qwen3-VL + 均值融合 + 双曲父节点评分",
    "qwen_gate_hyperbolic": "Qwen3-VL + 门控融合 + 双曲父节点评分（无深度损失）",
    "full": "完整知识树模型",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise argparse.ArgumentTypeError("Method must use NAME=OUTPUT_DIR syntax")
    return name, Path(path_text)


def _single_summary(path: Path) -> tuple[dict[str, float], str, dict[str, Any]]:
    metrics = _read_json(path / "metrics.json")["test"]
    supplementary = _read_json(path / "supplementary_metrics.json")
    test_extra = supplementary["splits"]["test"]["overall"]
    run_info = _read_json(path / "run_info.json")
    values = {
        metric: float(metrics[source]) for metric, source in FORMAL_SINGLE_KEYS.items()
    }
    values.update(
        {
            metric: float(test_extra[source])
            for metric, source in SUPPLEMENTARY_SINGLE_KEYS.items()
        }
    )
    values["structure_validity_rate"] = float(
        supplementary["structure"]["structure_validity_rate"]
    )
    return values, str(run_info["ground_truth_sha256"]), run_info


def _multi_summary(path: Path) -> tuple[dict[str, float], dict[str, float], str, dict[str, Any]]:
    report = _read_json(path / "multiseed_metrics.json")
    test = report["summary"]["test"]
    all_metrics = report["summary"]["all"]
    means = {metric: float(test[metric]["mean"]) for metric in METRICS if metric in test}
    means["structure_validity_rate"] = float(
        all_metrics["structure_validity_rate"]["mean"]
    )
    standard_deviations = {
        metric: float(test[metric]["std_sample"])
        for metric in METRICS
        if metric in test
    }
    standard_deviations["structure_validity_rate"] = float(
        all_metrics["structure_validity_rate"]["std_sample"]
    )
    missing = sorted(set(METRICS) - set(means))
    if missing:
        raise ValueError(f"Aggregate is missing metrics: {missing}")
    return (
        means,
        standard_deviations,
        str(report["metadata"]["ground_truth_sha256"]),
        report,
    )


def _fine_to_macro(data_dir: Path) -> dict[str, str]:
    tables = load_tables(data_dir)
    fine_ids = set(
        tables.nodes.loc[tables.nodes["level"].astype(int) == 3, "node_id"].astype(str)
    )
    mapping = {
        str(row.child_id): str(row.parent_id)
        for row in tables.edges[tables.edges["is_fixed"].astype(int) == 1].itertuples(
            index=False
        )
        if str(row.child_id) in fine_ids
    }
    if set(mapping) != fine_ids:
        raise ValueError("Every fine-stage node must have exactly one fixed macro parent")
    return mapping


def _prediction_paths(kind: str, path: Path, report: dict[str, Any], root: Path) -> list[Path]:
    if kind == "single":
        return [path / "predicted_edges.csv"]
    paths: list[Path] = []
    for _, run in sorted(report["runs"].items(), key=lambda item: int(item[0])):
        run_dir = Path(str(run["output_dir"]))
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        paths.append(run_dir / "predicted_edges.csv")
    return paths


def _plant_metrics(
    prediction_paths: list[Path],
    manifest: pd.DataFrame,
    fine_to_macro: dict[str, str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_index, path in enumerate(prediction_paths):
        predictions = pd.read_csv(path, dtype=str)
        required = {"split", "child_id", "true_parent_id", "predicted_parent_id", "rank"}
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"{path} is missing prediction columns: {missing}")
        test = predictions[predictions["split"] == "test"].copy()
        test["run_index"] = run_index
        test["rank"] = pd.to_numeric(test["rank"], errors="raise")
        frames.append(test)
    combined = pd.concat(frames, ignore_index=True)
    metadata = manifest[["image_id", "plant_id"]].rename(columns={"image_id": "child_id"})
    combined = combined.merge(metadata, on="child_id", how="left", validate="many_to_one")
    if combined["plant_id"].isna().any():
        raise ValueError("A test prediction has no plant ID in manifest.csv")
    combined["hits_at_1"] = (
        combined["true_parent_id"] == combined["predicted_parent_id"]
    ).astype(float)
    combined["mrr"] = 1.0 / combined["rank"].astype(float)
    true_macro = combined["true_parent_id"].map(fine_to_macro)
    pred_macro = combined["predicted_parent_id"].map(fine_to_macro)
    if true_macro.isna().any() or pred_macro.isna().any():
        raise ValueError("A prediction does not use a known level-3 parent")
    combined["macro_stage_accuracy"] = (true_macro == pred_macro).astype(float)
    return (
        combined.groupby("plant_id", as_index=False)[list(SIGNIFICANCE_METRICS)]
        .mean()
        .sort_values("plant_id")
        .reset_index(drop=True)
    )


def compare_baselines(
    *,
    single_methods: dict[str, Path],
    multi_methods: dict[str, Path],
    reference: str,
    data_dir: Path,
    output_dir: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    methods = {
        **{name: ("single", path) for name, path in single_methods.items()},
        **{name: ("multi", path) for name, path in multi_methods.items()},
    }
    if len(methods) != len(single_methods) + len(multi_methods):
        raise ValueError("A method name cannot be both single-run and multi-seed")
    if reference not in methods:
        raise ValueError(f"Reference method is absent: {reference}")
    root = (project_root or Path.cwd()).resolve()

    method_data: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for name, (kind, path) in methods.items():
        if kind == "single":
            means, fingerprint, report = _single_summary(path)
            standard_deviations = {metric: None for metric in METRICS}
            run_count = 1
        else:
            means, standard_deviations, fingerprint, report = _multi_summary(path)
            run_count = int(report["metadata"]["run_count"])
        fingerprints.add(fingerprint)
        method_data[name] = {
            "kind": kind,
            "path": path,
            "means": means,
            "standard_deviations": standard_deviations,
            "fingerprint": fingerprint,
            "report": report,
            "run_count": run_count,
        }
    if len(fingerprints) != 1:
        raise ValueError(f"Methods do not use identical ground truth: {sorted(fingerprints)}")

    reference_values = method_data[reference]["means"]
    summary_rows: list[dict[str, Any]] = []
    for name in methods:
        data = method_data[name]
        row: dict[str, Any] = {
            "method": name,
            "description": DESCRIPTIONS.get(name, name),
            "run_type": data["kind"],
            "run_count": data["run_count"],
            "is_reference": name == reference,
        }
        if data["kind"] == "single":
            row["model"] = data["report"].get("model")
            row["selected_prompt"] = data["report"].get("selected_prompt")
        else:
            row["model"] = "Qwen3-VL-Embedding-2B"
            row["selected_prompt"] = None
        for metric in METRICS:
            row[f"{metric}_mean"] = float(data["means"][metric])
            row[f"{metric}_std_sample"] = data["standard_deviations"][metric]
            row[f"{metric}_delta_vs_{reference}"] = (
                float(data["means"][metric]) - float(reference_values[metric])
            )
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)

    manifest = pd.read_csv(data_dir / "manifest.csv", dtype={"image_id": str, "plant_id": str})
    fine_to_macro = _fine_to_macro(data_dir)
    plants = {
        name: _plant_metrics(
            _prediction_paths(data["kind"], data["path"], data["report"], root),
            manifest,
            fine_to_macro,
        )
        for name, data in method_data.items()
    }
    reference_plants = plants[reference]
    significance_rows: list[dict[str, Any]] = []
    for name, frame in plants.items():
        if name == reference:
            continue
        paired = reference_plants.merge(
            frame,
            on="plant_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_reference", "_method"),
        )
        if len(paired) != len(reference_plants) or len(paired) != len(frame):
            raise ValueError(f"{name} does not contain the same test plants as {reference}")
        for metric in SIGNIFICANCE_METRICS:
            differences = (
                paired[f"{metric}_method"] - paired[f"{metric}_reference"]
            ).to_numpy(dtype=np.float64)
            significance_rows.append(
                {
                    "method": name,
                    "reference": reference,
                    "metric": metric,
                    "paired_unit": "plant_id",
                    "plant_count": len(paired),
                    "method_minus_reference_mean": float(differences.mean()),
                    "p_value_exact_two_sided": exact_sign_flip_pvalue(differences),
                }
            )
    significance_frame = pd.DataFrame(significance_rows)
    significance_frame["p_value_holm"] = holm_adjust(
        significance_frame["p_value_exact_two_sided"].tolist()
    )
    significance_frame["significant_at_0_05"] = (
        significance_frame["p_value_holm"] < 0.05
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    significance_frame.to_csv(
        output_dir / "comparison_significance.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "metadata": {
            "reference": reference,
            "methods": list(methods),
            "ground_truth_sha256": next(iter(fingerprints)),
            "deterministic_standard_deviation": None,
            "significance_test": (
                "Exact two-sided paired sign-flip test on 15 test plants. Per-plant values "
                "average all images and, for multi-seed methods, all five seeds."
            ),
            "multiple_testing_correction": "Holm correction over all method-metric tests",
        },
        "summary": json.loads(summary_frame.to_json(orient="records")),
        "significance": json.loads(significance_frame.to_json(orient="records")),
    }
    write_json(output_dir / "comparison_results.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare deterministic and multi-seed baselines")
    parser.add_argument("--single", action="append", type=_parse_named_path, default=[])
    parser.add_argument("--multi", action="append", type=_parse_named_path, default=[])
    parser.add_argument("--reference", default="full")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/tomatomap_kt5k")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/comparisons/summary")
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    single = dict(args.single)
    multi = dict(args.multi)
    if len(single) != len(args.single) or len(multi) != len(args.multi):
        raise ValueError("Each method name may be provided only once per run type")
    result = compare_baselines(
        single_methods=single,
        multi_methods=multi,
        reference=args.reference,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
