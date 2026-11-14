from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRICS = [
    "hits_at_1",
    "hits_at_3",
    "hits_at_5",
    "mrr",
    "upper_parent_accuracy",
    "macro_stage_macro_f1",
    "fine_macro_f1_50_classes",
    "fine_macro_f1_observed_classes",
]


def _read_summary(path: Path, method: str, group: str, source: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    frame = frame[(frame["split"].astype(str) == "test") & frame["metric"].isin(METRICS)]
    return [
        {
            "group": group,
            "method": method,
            "source": source,
            "metric": str(row.metric),
            "n": int(row.n),
            "mean": float(row.mean),
            "std_sample": float(row.std_sample),
            "min": float(row.min),
            "max": float(row.max),
        }
        for row in frame.itertuples(index=False)
    ]


def _original_runs(root: Path) -> dict[int, Path]:
    return {
        42: root / "outputs" / "tomatomap_qwen3vl",
        **{
            seed: root / "outputs" / f"tomatomap_qwen3vl_seed{seed}"
            for seed in (43, 44, 45, 46)
        },
    }


def _new_runs(output_root: Path, group: str, name: str, seeds: list[int]) -> dict[int, Path]:
    return {seed: output_root / group / name / f"seed{seed}" for seed in seeds}


def _plant_metrics(
    runs: dict[int, Path],
    plant_by_child: dict[str, int],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed, run_dir in runs.items():
        predictions = pd.read_csv(run_dir / "predicted_edges.csv")
        test = predictions[predictions["split"].astype(str) == "test"].copy()
        test["plant_id"] = test["child_id"].astype(str).map(plant_by_child)
        if test["plant_id"].isna().any():
            raise ValueError(f"Predictions contain unmapped child IDs: {run_dir}")
        test["hits_at_1"] = (test["rank"].astype(int) <= 1).astype(float)
        test["mrr"] = 1.0 / test["rank"].astype(float)
        test["upper_parent_accuracy"] = test["same_upper_parent"].astype(float)
        grouped = (
            test.groupby("plant_id")[["hits_at_1", "mrr", "upper_parent_accuracy"]]
            .mean()
            .reset_index()
        )
        grouped["seed"] = seed
        rows.append(grouped)
    combined = pd.concat(rows, ignore_index=True)
    return (
        combined.groupby("plant_id")[["hits_at_1", "mrr", "upper_parent_accuracy"]]
        .mean()
        .sort_index()
    )


def _bootstrap_ci(values: np.ndarray, seed: int = 2026, samples: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-15)
        total += 1
    return float(exceed / total)


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def build_effect_sizes(
    *,
    root: Path,
    output_root: Path,
    method_runs: dict[str, dict[int, Path]],
) -> pd.DataFrame:
    manifest = pd.read_csv(root / "data" / "processed" / "tomatomap_kt5k" / "manifest.csv")
    plant_by_child = dict(zip(manifest["image_id"].astype(str), manifest["plant_id"].astype(int)))
    reference = _plant_metrics(_original_runs(root), plant_by_child)
    rows: list[dict[str, Any]] = []
    for method, runs in method_runs.items():
        candidate = _plant_metrics(runs, plant_by_child)
        common = reference.index.intersection(candidate.index)
        if len(common) != 15:
            raise ValueError(f"Expected 15 paired test plants for {method}, found {len(common)}")
        for metric in ("hits_at_1", "mrr", "upper_parent_accuracy"):
            differences = candidate.loc[common, metric].to_numpy() - reference.loc[common, metric].to_numpy()
            lower, upper = _bootstrap_ci(differences)
            standard_deviation = float(differences.std(ddof=1))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "plant_count": len(common),
                    "mean_difference_vs_original_full": float(differences.mean()),
                    "bootstrap_ci_95_lower": lower,
                    "bootstrap_ci_95_upper": upper,
                    "cohens_dz": (
                        float(differences.mean() / standard_deviation)
                        if standard_deviation > 0
                        else 0.0
                    ),
                    "exact_sign_flip_p": _exact_sign_flip_p(differences),
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p"] = _holm(frame["exact_sign_flip_p"].astype(float).tolist())
    frame.to_csv(output_root / "plant_level_effect_sizes.csv", index=False, encoding="utf-8-sig")
    return frame


def _sensitivity_table(root: Path, output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    default_metrics = json.loads(
        (root / "outputs" / "tomatomap_qwen3vl" / "metrics.json").read_text(encoding="utf-8")
    )["val"]
    default_history = json.loads(
        (root / "outputs" / "tomatomap_qwen3vl" / "history.json").read_text(encoding="utf-8")
    )
    rows.append(
        {
            "setting": "default",
            "best_epoch": max(default_history, key=lambda row: row["val"]["mrr"])["epoch"],
            **default_metrics,
        }
    )
    sensitivity_root = output_root / "sensitivity"
    for setting_dir in sorted(path for path in sensitivity_root.iterdir() if path.is_dir()):
        run_dir = setting_dir / "seed42"
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["val"]
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "setting": setting_dir.name,
                "best_epoch": max(history, key=lambda row: row["val"]["mrr"])["epoch"],
                **metrics,
            }
        )
    frame = pd.DataFrame(rows).sort_values("mrr", ascending=False)
    frame.to_csv(output_root / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def synthesize(root: Path, output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    original_summary = root / "outputs" / "tomatomap_qwen3vl_multiseed" / "multiseed_summary.csv"
    rows.extend(
        _read_summary(
            original_summary,
            "Original full implementation",
            "reference",
            "existing five-seed formal output",
        )
    )
    summary_specs = {
        "supervised_baselines": ["linear_image", "gated_mlp", "hierarchical_gated"],
        "text_controls": ["item_text_shuffled", "item_text_constant"],
        "imbalance": ["class_weighted", "focal_loss"],
        "encoder_control": ["encoder_qwen", "encoder_clip", "encoder_siglip2"],
        "prompt_sensitivity": ["prompt_official", "prompt_contextual", "prompt_concise"],
        "split_robustness": ["plant_split_43", "plant_split_44"],
    }
    for group, names in summary_specs.items():
        for name in names:
            rows.extend(
                _read_summary(
                    output_root / group / name / "summary" / "multiseed_summary.csv",
                    name,
                    group,
                    "new reviewer experiment",
                )
            )
    rows.extend(
        _read_summary(
            root / "outputs" / "ablations" / "no_item_text" / "multiseed" / "multiseed_summary.csv",
            "no_item_text",
            "text_controls",
            "existing five-seed ablation",
        )
    )
    method_summary = pd.DataFrame(rows)
    method_summary.to_csv(output_root / "reviewer_method_summary_long.csv", index=False, encoding="utf-8-sig")
    pivot = method_summary.pivot_table(
        index=["group", "method", "source", "n"],
        columns="metric",
        values=["mean", "std_sample"],
    )
    pivot.columns = [f"{stat}_{metric}" for stat, metric in pivot.columns]
    pivot.reset_index().to_csv(
        output_root / "reviewer_method_summary_wide.csv", index=False, encoding="utf-8-sig"
    )

    method_runs = {
        "linear_image": _new_runs(output_root, "supervised_baselines", "linear_image", [42, 43, 44, 45, 46]),
        "gated_mlp": _new_runs(output_root, "supervised_baselines", "gated_mlp", [42, 43, 44, 45, 46]),
        "hierarchical_gated": _new_runs(output_root, "supervised_baselines", "hierarchical_gated", [42, 43, 44, 45, 46]),
        "item_text_shuffled": _new_runs(output_root, "text_controls", "item_text_shuffled", [42, 43, 44, 45, 46]),
        "item_text_constant": _new_runs(output_root, "text_controls", "item_text_constant", [42, 43, 44]),
        "class_weighted": _new_runs(output_root, "imbalance", "class_weighted", [42, 43, 44]),
        "focal_loss": _new_runs(output_root, "imbalance", "focal_loss", [42, 43, 44]),
        "encoder_qwen": _new_runs(output_root, "encoder_control", "encoder_qwen", [42, 43, 44]),
        "encoder_clip": _new_runs(output_root, "encoder_control", "encoder_clip", [42, 43, 44]),
        "encoder_siglip2": _new_runs(output_root, "encoder_control", "encoder_siglip2", [42, 43, 44]),
        "prompt_official": _new_runs(output_root, "prompt_sensitivity", "prompt_official", [42, 43, 44]),
        "prompt_contextual": _new_runs(output_root, "prompt_sensitivity", "prompt_contextual", [42, 43, 44]),
        "prompt_concise": _new_runs(output_root, "prompt_sensitivity", "prompt_concise", [42, 43, 44]),
    }
    effects = build_effect_sizes(
        root=root,
        output_root=output_root,
        method_runs=method_runs,
    )
    sensitivity = _sensitivity_table(root, output_root)
    report = {
        "new_formal_training_runs": 53,
        "new_validation_only_runs": 11,
        "total_new_runs": 64,
        "method_summary_rows": len(method_summary),
        "effect_size_tests": len(effects),
        "sensitivity_settings_including_default": len(sensitivity),
    }
    with (output_root / "reviewer_results_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize reviewer experiment results")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/reviewer_experiments"),
    )
    args = parser.parse_args()
    synthesize(args.root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
