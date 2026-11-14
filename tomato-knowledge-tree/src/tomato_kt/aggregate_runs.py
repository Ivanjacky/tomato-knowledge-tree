from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import write_json

FORMAL_METRICS = {
    "loss": "loss",
    "hits_at_1": "hits@1",
    "hits_at_3": "hits@3",
    "hits_at_5": "hits@5",
    "mrr": "mrr",
    "upper_parent_accuracy": "upper_parent_accuracy",
}
SUPPLEMENTARY_METRICS = {
    "fine_macro_f1_50_classes": "fine_macro_f1_reference_classes",
    "fine_macro_f1_observed_classes": "fine_macro_f1_observed_classes",
    "macro_stage_accuracy": "macro_stage_accuracy",
    "macro_stage_macro_f1": "macro_stage_macro_f1",
    "invalid_parent_level_rate": "invalid_parent_level_rate",
}
TRUTH_COLUMNS = ["child_id", "split", "true_parent_id"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _truth_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str)
    missing = sorted(set(TRUTH_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    truth = frame[TRUTH_COLUMNS].sort_values("child_id").reset_index(drop=True)
    if truth["child_id"].duplicated().any():
        raise ValueError(f"Duplicate child predictions in {path}")
    return truth


def _truth_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "std_sample": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_runs(
    runs: dict[int, Path], output_dir: Path, require_valid_tree: bool = True
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("At least two distinct seeds are required for aggregation")

    long_rows: list[dict[str, Any]] = []
    run_reports: dict[str, Any] = {}
    reference_truth: pd.DataFrame | None = None
    reference_seed: int | None = None
    truth_fingerprint: str | None = None

    for seed, run_dir in sorted(runs.items()):
        metrics = _read_json(run_dir / "metrics.json")
        supplementary = _read_json(run_dir / "supplementary_metrics.json")
        truth = _truth_table(run_dir / "predicted_edges.csv")
        if reference_truth is None:
            reference_truth = truth
            reference_seed = seed
            truth_fingerprint = _truth_fingerprint(truth)
        elif not truth.equals(reference_truth):
            raise ValueError(
                f"Seed {seed} does not use the same child/split/ground-truth rows "
                f"as reference seed {reference_seed}"
            )

        split_counts = truth["split"].value_counts().sort_index().to_dict()
        structure = supplementary.get("structure", {})
        if require_valid_tree and not structure.get("is_valid_tree", False):
            raise ValueError(f"Seed {seed} did not produce a valid knowledge tree")
        run_reports[str(seed)] = {
            "output_dir": _portable_path(run_dir),
            "split_counts": {str(key): int(value) for key, value in split_counts.items()},
            "is_valid_tree": bool(structure.get("is_valid_tree", False)),
            "structure_validity_rate": float(structure["structure_validity_rate"]),
        }

        for split in ("train", "val", "test"):
            if split not in metrics or split not in supplementary.get("splits", {}):
                raise ValueError(f"Seed {seed} is missing split metrics for {split}")
            for output_name, source_name in FORMAL_METRICS.items():
                long_rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "metric": output_name,
                        "value": float(metrics[split][source_name]),
                    }
                )
            overall = supplementary["splits"][split]["overall"]
            for output_name, source_name in SUPPLEMENTARY_METRICS.items():
                if source_name in overall:
                    value = float(overall[source_name])
                elif source_name == "invalid_parent_level_rate" and structure.get(
                    "is_valid_tree", False
                ):
                    # Backward compatibility for completed, verified full-model runs
                    # produced before this diagnostic metric was introduced.
                    value = 0.0
                else:
                    raise ValueError(
                        f"Seed {seed} is missing supplementary metric {source_name} "
                        f"for split {split}"
                    )
                long_rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "metric": output_name,
                        "value": value,
                    }
                )
        long_rows.append(
            {
                "seed": seed,
                "split": "all",
                "metric": "structure_validity_rate",
                "value": float(structure["structure_validity_rate"]),
            }
        )

    long_frame = pd.DataFrame(long_rows)
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for (split, metric), group in long_frame.groupby(["split", "metric"], sort=True):
        metric_summary = _summary(
            group["value"].astype(float).tolist()
        )
        summary.setdefault(str(split), {})[str(metric)] = metric_summary
        summary_rows.append(
            {"split": split, "metric": metric, **metric_summary}
        )

    report = {
        "metadata": {
            "seeds": sorted(runs),
            "run_count": len(runs),
            "standard_deviation": "sample standard deviation (ddof=1)",
            "identical_ground_truth_rows": True,
            "ground_truth_sha256": truth_fingerprint,
            "required_valid_tree": require_valid_tree,
        },
        "runs": run_reports,
        "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "multiseed_metrics.json", report)
    long_frame.sort_values(["split", "metric", "seed"]).to_csv(
        output_dir / "multiseed_metrics_long.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "multiseed_summary.csv", index=False, encoding="utf-8-sig"
    )
    return report


def _parse_run_spec(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text or not path_text:
        raise argparse.ArgumentTypeError("Run must use SEED=OUTPUT_DIR syntax")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed: {seed_text}") from exc
    return seed, Path(path_text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate comparable full-model runs across random seeds."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_spec,
        required=True,
        metavar="SEED=OUTPUT_DIR",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/tomatomap_qwen3vl_multiseed")
    )
    parser.add_argument("--allow-invalid-tree", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runs = dict(args.run)
    if len(runs) != len(args.run):
        raise ValueError("Each seed may be provided only once")
    report = aggregate_runs(
        runs, args.output_dir, require_valid_tree=not args.allow_invalid_tree
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
