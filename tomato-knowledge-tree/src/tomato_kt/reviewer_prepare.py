from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .reviewer_splits import build_split
from .utils import load_yaml


def _write_config(path: Path, base: dict, changes: dict) -> None:
    config = dict(base)
    config.update(changes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def prepare(root: Path) -> dict[str, object]:
    base = load_yaml(root / "configs" / "tomatomap.yaml")
    source_data = root / "data" / "processed" / "tomatomap_kt5k"
    original_embeddings = source_data / "embeddings.npz"
    split_root = root / "data" / "processed" / "reviewer_splits"
    for split_seed in (43, 44):
        output = split_root / f"split_seed{split_seed}"
        if not (output / "split_validation.json").exists():
            build_split(
                source_data_dir=source_data,
                output_dir=output,
                seed=split_seed,
            )

    config_dir = root / "configs" / "reviewer"
    output_root = root / "outputs" / "reviewer_experiments"
    embedding_root = root / "data" / "processed" / "reviewer_embeddings"
    specs: list[dict[str, object]] = []

    def add(name: str, group: str, seeds: list[int], changes: dict) -> None:
        config_path = config_dir / f"{name}.yaml"
        _write_config(
            config_path,
            base,
            {
                **changes,
                "output_dir": str(output_root / group / name / "seed42"),
            },
        )
        specs.append(
            {
                "name": name,
                "group": group,
                "seeds": ",".join(str(seed) for seed in seeds),
                "config": str(config_path),
            }
        )

    add(
        "item_text_shuffled",
        "text_controls",
        [42, 43, 44, 45, 46],
        {"item_text_mode": "shuffled", "item_text_shuffle_seed": 2026},
    )
    add(
        "item_text_constant",
        "text_controls",
        [42, 43, 44],
        {"item_text_mode": "constant"},
    )
    add(
        "class_weighted",
        "imbalance",
        [42, 43, 44],
        {"classification_loss": "class_weighted", "class_weight_power": 0.5},
    )
    add(
        "focal_loss",
        "imbalance",
        [42, 43, 44],
        {
            "classification_loss": "focal",
            "class_weight_power": 0.0,
            "focal_gamma": 2.0,
        },
    )
    encoder_dims = {"qwen": 2048, "clip": 512, "siglip2": 768}
    for encoder, dimension in encoder_dims.items():
        add(
            f"encoder_{encoder}",
            "encoder_control",
            [42, 43, 44],
            {
                "embedding_dim": dimension,
                "embeddings_path": str(embedding_root / f"{encoder}_official.npz"),
                "item_text_mode": "disabled",
            },
        )
    for variant in ("official", "contextual", "concise"):
        add(
            f"prompt_{variant}",
            "prompt_sensitivity",
            [42, 43, 44],
            {
                "embedding_dim": 2048,
                "embeddings_path": str(embedding_root / f"qwen_{variant}.npz"),
                "item_text_mode": "enabled",
            },
        )
    for split_seed in (43, 44):
        add(
            f"plant_split_{split_seed}",
            "split_robustness",
            [42, 43, 44],
            {
                "data_dir": str(split_root / f"split_seed{split_seed}"),
                "embeddings_path": str(original_embeddings),
            },
        )

    sensitivity = {
        "curvature_0p5": {"curvature": 0.5},
        "curvature_2p0": {"curvature": 2.0},
        "depth_margin_0p04": {"depth_margin": 0.04},
        "depth_margin_0p12": {"depth_margin": 0.12},
        "lambda_depth_0p10": {"lambda_depth": 0.10},
        "lambda_depth_0p40": {"lambda_depth": 0.40},
        "lambda_align_0p10": {"lambda_align": 0.10},
        "learning_rate_0p0001": {"learning_rate": 0.0001},
        "learning_rate_0p001": {"learning_rate": 0.001},
        "dropout_0p10": {"dropout": 0.10},
        "dropout_0p30": {"dropout": 0.30},
    }
    for name, changes in sensitivity.items():
        add(
            name,
            "sensitivity",
            [42],
            {**changes, "evaluate_test": False},
        )

    plan = pd.DataFrame(specs)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "run_plan.csv"
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    report = {
        "config_count": len(plan),
        "planned_tree_runs": int(
            sum(len(str(value).split(",")) for value in plan["seeds"].tolist())
        ),
        "plan": str(plan_path.resolve()),
        "config_dir": str(config_dir.resolve()),
        "split_root": str(split_root.resolve()),
        "embedding_root": str(embedding_root.resolve()),
    }
    with (output_root / "preparation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reviewer experiment configs and splits")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    prepare(args.root.resolve())


if __name__ == "__main__":
    main()
