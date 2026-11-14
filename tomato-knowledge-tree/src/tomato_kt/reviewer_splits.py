from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_tomatomap import _write_knowledge_tables, assign_plant_splits


def _distribution(frame: pd.DataFrame, stages: list[int]) -> np.ndarray:
    counts = frame["bbch_stage"].astype(int).value_counts()
    values = np.asarray([counts.get(stage, 0) for stage in stages], dtype=np.float64)
    return values / values.sum()


def _total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def build_split(
    *,
    source_data_dir: Path,
    output_dir: Path,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, object]:
    manifest = pd.read_csv(source_data_dir / "manifest.csv")
    required = {"plant_id", "bbch_stage", "image_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Source manifest is missing columns: {missing}")
    split_map = assign_plant_splits(
        manifest["plant_id"].astype(int).tolist(),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    new_manifest = manifest.copy()
    new_manifest["split"] = new_manifest["plant_id"].astype(int).map(split_map)
    if new_manifest["split"].isna().any():
        raise ValueError("At least one plant did not receive a split")
    _write_knowledge_tables(new_manifest, output_dir)

    plants = {
        split: sorted(
            new_manifest.loc[
                new_manifest["split"].astype(str) == split, "plant_id"
            ].astype(int).unique().tolist()
        )
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": sorted(set(plants["train"]) & set(plants["val"])),
        "train_test": sorted(set(plants["train"]) & set(plants["test"])),
        "val_test": sorted(set(plants["val"]) & set(plants["test"])),
    }
    if any(overlaps.values()):
        raise ValueError(f"Plant leakage detected: {overlaps}")
    stages = sorted(new_manifest["bbch_stage"].astype(int).unique().tolist())
    overall_distribution = _distribution(new_manifest, stages)
    split_rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        frame = new_manifest[new_manifest["split"].astype(str) == split]
        split_rows.append(
            {
                "split": split,
                "plant_count": int(frame["plant_id"].nunique()),
                "image_count": len(frame),
                "stage_count": int(frame["bbch_stage"].nunique()),
                "total_variation_from_overall": _total_variation(
                    _distribution(frame, stages), overall_distribution
                ),
            }
        )
    pd.DataFrame(split_rows).to_csv(
        output_dir / "split_robustness_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {"plant_id": plant_id, "split": split}
            for split, plant_ids in plants.items()
            for plant_id in plant_ids
        ]
    ).sort_values("plant_id").to_csv(
        output_dir / "plant_split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "plant_count": int(new_manifest["plant_id"].nunique()),
        "image_count": len(new_manifest),
        "stage_count": len(stages),
        "plant_overlap": overlaps,
        "splits": split_rows,
        "training_stage_ids": sorted(
            new_manifest.loc[
                new_manifest["split"].astype(str) == "train", "bbch_stage"
            ].astype(int).unique().tolist()
        ),
        "validation_stage_ids": sorted(
            new_manifest.loc[
                new_manifest["split"].astype(str) == "val", "bbch_stage"
            ].astype(int).unique().tolist()
        ),
        "test_stage_ids": sorted(
            new_manifest.loc[
                new_manifest["split"].astype(str) == "test", "bbch_stage"
            ].astype(int).unique().tolist()
        ),
    }
    with (output_dir / "split_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent plant-level reviewer splits")
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()
    build_split(
        source_data_dir=args.source_data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
