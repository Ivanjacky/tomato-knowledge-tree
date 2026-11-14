from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

IMAGE_PATTERN = re.compile(
    r"^pi(?P<camera>\d+)_(?P<sequence>\d+)_(?P<plant>\d+)_"
    r"(?P<pose>\d+)_(?P<timestamp>\d{14})$",
    flags=re.IGNORECASE,
)
GROUP_PATTERN = re.compile(r"^group\s*(?P<group>\d+)$", flags=re.IGNORECASE)

MAJOR_STAGE_NAMES = {
    1: ("叶片发育", "BBCH 10—19：番茄叶片发育阶段"),
    2: ("侧枝形成", "BBCH 20—29：番茄侧枝形成阶段"),
    5: ("花序出现", "BBCH 50—59：番茄花序出现阶段"),
    6: ("开花", "BBCH 60—69：番茄开花阶段"),
    7: ("果实发育", "BBCH 70—79：番茄果实发育阶段"),
    8: ("果实成熟", "BBCH 80—89：番茄果实成熟阶段"),
}


def _clean_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    text = str(value).strip().replace(".0", "")
    digits = re.sub(r"\D", "", text)
    return digits if len(digits) == 8 else None


def load_bbch_labels(path: Path) -> dict[tuple[int, int, str], int]:
    """Loads the official five-sheet BBCH workbook.

    Keys are ``(group_number, plant_id, YYYYMMDD)``. The implementation mirrors
    the official TomatoMAP builder but produces a direct lookup table.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    lookup: dict[tuple[int, int, str], int] = {}

    for sheet_name, frame in sheets.items():
        match = GROUP_PATTERN.match(str(sheet_name).replace(" ", ""))
        if match is None:
            match = re.search(r"(\d+)", str(sheet_name))
        if match is None or frame.empty:
            continue
        group_id = int(match.group(1) if match.lastindex else match.group(0))
        date_column = frame.columns[0]

        for _, row in frame.iterrows():
            date = _clean_date(row[date_column])
            if date is None:
                continue
            for column in frame.columns[1:]:
                plant_match = re.match(r"BBCH\s*plant\s*(\d+)", str(column), re.IGNORECASE)
                if plant_match is None or pd.isna(row[column]):
                    continue
                plant_id = int(plant_match.group(1))
                lookup[(group_id, plant_id, date)] = int(float(row[column]))

    if not lookup:
        raise ValueError(f"No BBCH labels were found in {path}")
    return lookup


def parse_image_filename(path: Path) -> dict[str, int | str] | None:
    match = IMAGE_PATTERN.match(path.stem)
    if match is None:
        return None
    values = {key: int(value) for key, value in match.groupdict().items() if key != "timestamp"}
    values["timestamp"] = match.group("timestamp")
    values["capture_date"] = match.group("timestamp")[:8]
    return values


def parse_image(path: Path) -> dict[str, int | str] | None:
    values = parse_image_filename(path)
    if values is None:
        return None
    group_id = None
    for part in path.parts:
        group_match = GROUP_PATTERN.match(part)
        if group_match:
            group_id = int(group_match.group("group"))
            break
    if group_id is None:
        return None
    values["group"] = group_id
    return values


def assign_plant_splits(
    plant_ids: list[int], seed: int, train_ratio: float, val_ratio: float
) -> dict[int, str]:
    unique = np.asarray(sorted(set(plant_ids)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_count = round(len(unique) * train_ratio)
    val_count = round(len(unique) * val_ratio)
    train_count = min(train_count, len(unique))
    val_count = min(val_count, len(unique) - train_count)
    result: dict[int, str] = {}
    for plant in unique[:train_count]:
        result[int(plant)] = "train"
    for plant in unique[train_count : train_count + val_count]:
        result[int(plant)] = "val"
    for plant in unique[train_count + val_count :]:
        result[int(plant)] = "test"
    return result


def _candidate_image_root(raw_root: Path) -> Path:
    candidates = [raw_root / "img", raw_root / "TomatoMAP" / "img", raw_root]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("*.jpg")):
            return candidate
    raise FileNotFoundError(f"Cannot locate TomatoMAP images below {raw_root}")


def _candidate_label_root(raw_root: Path) -> Path | None:
    candidates = [
        raw_root / "labels" / "labels",
        raw_root / "TomatoMAP" / "labels" / "labels",
        raw_root / "labels",
    ]
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def _find_label(image: Path, image_root: Path, label_root: Path | None) -> str:
    if label_root is None:
        return ""
    try:
        relative = image.relative_to(image_root).with_suffix(".txt")
    except ValueError:
        return ""
    candidate = label_root / relative
    return str(candidate.resolve()) if candidate.exists() else ""


def _write_knowledge_tables(manifest: pd.DataFrame, output: Path) -> None:
    required = {
        "image_id",
        "image_path",
        "relative_path",
        "plant_id",
        "capture_date",
        "capture_timestamp",
        "camera_id",
        "pose_id",
        "bbch_stage",
        "split",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")

    manifest = manifest.sort_values(
        ["split", "plant_id", "capture_date", "camera_id", "pose_id"]
    ).reset_index(drop=True)
    node_rows: list[dict] = [
        {
            "node_id": "tomato",
            "name": "番茄知识",
            "description": "番茄作物全生命周期多媒体知识根节点",
            "node_kind": "concept",
            "level": 0,
            "split": "fixed",
            "is_candidate_parent": 1,
        },
        {
            "node_id": "growth_stage",
            "name": "生长发育",
            "description": "番茄BBCH生育期和物候发育知识",
            "node_kind": "concept",
            "level": 1,
            "split": "fixed",
            "is_candidate_parent": 1,
        },
    ]
    edge_rows: list[dict] = [
        {
            "parent_id": "tomato",
            "child_id": "growth_stage",
            "relation": "has_branch",
            "split": "fixed",
            "is_fixed": 1,
        }
    ]
    media_rows: list[dict] = []

    stages = sorted(int(value) for value in manifest["bbch_stage"].unique())
    major_codes = sorted({stage // 10 for stage in stages})
    for major in major_codes:
        name, description = MAJOR_STAGE_NAMES.get(
            major, (f"BBCH {major}0—{major}9", f"番茄BBCH {major}0至{major}9阶段")
        )
        major_id = f"bbch_major_{major}"
        node_rows.append(
            {
                "node_id": major_id,
                "name": name,
                "description": description,
                "node_kind": "concept",
                "level": 2,
                "split": "fixed",
                "is_candidate_parent": 1,
            }
        )
        edge_rows.append(
            {
                "parent_id": "growth_stage",
                "child_id": major_id,
                "relation": "has_stage_group",
                "split": "fixed",
                "is_fixed": 1,
            }
        )

    for stage in stages:
        stage_id = f"bbch_{stage:02d}"
        description = f"番茄BBCH {stage:02d}细粒度生育阶段"
        node_rows.append(
            {
                "node_id": stage_id,
                "name": f"BBCH {stage:02d}",
                "description": description,
                "node_kind": "concept",
                "level": 3,
                "split": "fixed",
                "is_candidate_parent": 1,
            }
        )
        edge_rows.append(
            {
                "parent_id": f"bbch_major_{stage // 10}",
                "child_id": stage_id,
                "relation": "has_fine_stage",
                "split": "fixed",
                "is_fixed": 1,
            }
        )
        media_rows.append(
            {
                "media_id": f"text:{stage_id}",
                "node_id": stage_id,
                "modality": "text",
                "content": description,
                "path": "",
                "split": "fixed",
            }
        )

    for record in manifest.to_dict(orient="records"):
        item_id = str(record["image_id"])
        split = str(record["split"])
        stage = int(record["bbch_stage"])
        camera_text = str(record.get("camera_label", "")).strip()
        if not camera_text or camera_text.lower() == "nan":
            camera_text = f"相机位置{int(record['camera_id'])}"
        pose_degrees = record.get("pose_degrees", "")
        if pd.isna(pose_degrees) or str(pose_degrees).strip() == "":
            pose_text = f"旋转姿态{int(record['pose_id'])}"
        else:
            pose_text = f"转台角度{int(pose_degrees)}度"
        # Exclude date, plant ID, and BBCH target so text cannot leak the label.
        item_description = f"番茄植株多视角RGB表型图像；{camera_text}；{pose_text}。"
        node_rows.append(
            {
                "node_id": item_id,
                "name": Path(str(record["relative_path"])).stem,
                "description": item_description,
                "node_kind": "item",
                "level": 4,
                "split": split,
                "is_candidate_parent": 0,
            }
        )
        edge_rows.append(
            {
                "parent_id": f"bbch_{stage:02d}",
                "child_id": item_id,
                "relation": "visual_evidence_of",
                "split": split,
                "is_fixed": 0,
            }
        )
        media_rows.append(
            {
                "media_id": f"media:{item_id}",
                "node_id": item_id,
                "modality": "image",
                "content": "",
                "path": str(record["image_path"]),
                "split": split,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(node_rows).to_csv(output / "nodes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(edge_rows).to_csv(output / "edges.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(media_rows).to_csv(output / "media.csv", index=False, encoding="utf-8-sig")
    summary = (
        manifest.groupby(["split", "bbch_stage"], dropna=False)
        .size()
        .rename("image_count")
        .reset_index()
    )
    summary.to_csv(output / "split_summary.csv", index=False, encoding="utf-8-sig")
    print(
        f"Prepared {len(manifest)} images, {manifest['plant_id'].nunique()} plants, "
        f"and {len(stages)} BBCH stages at {output.resolve()}"
    )


def _sample_value(sample: Any, field: str, default: Any = None) -> Any:
    try:
        value = sample[field]
    except (KeyError, AttributeError, TypeError):
        value = getattr(sample, field, default)
    return default if value is None else value


def build_knowledge_files_from_fiftyone(
    dataset_name: str,
    output: Path,
    camera_ids: set[int],
    pose_ids: set[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Build knowledge tables from the Voxel51 TomatoMAP grouped dataset."""
    try:
        import fiftyone as fo
    except ImportError as exc:
        raise RuntimeError(
            "FiftyOne is required for --fiftyone-dataset. Run "
            "scripts/bootstrap_windows.ps1 -Device cu128 -WithQwen -WithFiftyOne."
        ) from exc

    available_datasets = fo.list_datasets()
    if dataset_name not in available_datasets:
        tomato_names = [name for name in available_datasets if "tomato" in name.lower()]
        hint = f" Available Tomato datasets: {tomato_names}" if tomato_names else ""
        raise ValueError(f"FiftyOne dataset {dataset_name!r} was not found.{hint}")

    dataset = fo.load_dataset(dataset_name)
    required_fields = {
        "filepath",
        "bbch_stage",
        "plant_id",
        "piid",
        "pose_id",
        "image_set_id",
    }
    schema = set(dataset.get_field_schema().keys())
    missing_fields = sorted(required_fields - schema)
    if missing_fields:
        raise ValueError(
            "The selected dataset is not the Voxel51 TomatoMAP det/cls release; "
            f"missing fields: {', '.join(missing_fields)}"
        )

    if getattr(dataset, "media_type", None) == "group":
        requested_slices = [f"pos_{camera_id}" for camera_id in sorted(camera_ids)]
        available_slices = set(dataset.group_slices)
        missing_slices = sorted(set(requested_slices) - available_slices)
        if missing_slices:
            raise ValueError(
                f"Camera slices do not exist: {missing_slices}; available: {sorted(available_slices)}"
            )
        view = dataset.select_group_slices(requested_slices, flat=True)
    else:
        view = dataset

    rows: list[dict] = []
    for sample in tqdm(view.iter_samples(), total=len(view), desc="Reading FiftyOne TomatoMAP"):
        tags = set(_sample_value(sample, "tags", []))
        if tags and "det" not in tags:
            continue
        image_path = Path(str(_sample_value(sample, "filepath"))).resolve()
        parsed = parse_image_filename(image_path)
        camera_id = int(_sample_value(sample, "piid", parsed["camera"] if parsed else -1))
        pose_id = int(_sample_value(sample, "pose_id", parsed["pose"] if parsed else -1))
        if camera_id not in camera_ids or pose_id not in pose_ids:
            continue
        if not image_path.is_file():
            raise FileNotFoundError(
                f"FiftyOne metadata points to a missing image: {image_path}. "
                "Run load_from_hub again on this computer to restore media paths."
            )

        timestamp = str(parsed["timestamp"]) if parsed else ""
        capture_value = _sample_value(sample, "capture_datetime")
        if isinstance(capture_value, datetime):
            timestamp = capture_value.strftime("%Y%m%d%H%M%S")
        if not timestamp:
            raise ValueError(f"Cannot determine capture timestamp for {image_path}")
        image_set_id = int(
            _sample_value(sample, "image_set_id", parsed["sequence"] if parsed else -1)
        )
        group_value = _sample_value(sample, "group")
        group_id = str(getattr(group_value, "id", image_set_id))
        rows.append(
            {
                "image_id": f"tomatomap:{image_set_id}:{camera_id}",
                "image_path": str(image_path),
                "relative_path": image_path.name,
                "label_path": "",
                "group_id": group_id,
                "image_set_id": image_set_id,
                "plant_id": int(
                    _sample_value(sample, "plant_id", parsed["plant"] if parsed else -1)
                ),
                "capture_date": timestamp[:8],
                "capture_timestamp": timestamp,
                "camera_id": camera_id,
                "camera_label": str(
                    _sample_value(sample, "camera_label", f"相机位置{camera_id}")
                ),
                "pose_id": pose_id,
                "pose_degrees": int(
                    _sample_value(sample, "pose_degrees", (pose_id - 1) * 30)
                ),
                "bbch_stage": int(_sample_value(sample, "bbch_stage")),
            }
        )

    if not rows:
        raise RuntimeError(
            "No matching det/cls images were found in FiftyOne. Check dataset name, camera IDs, "
            "and pose IDs."
        )
    manifest = pd.DataFrame(rows)
    if manifest["image_id"].duplicated().any():
        duplicate = manifest.loc[manifest["image_id"].duplicated(), "image_id"].iloc[0]
        raise ValueError(f"Duplicate image ID found in FiftyOne data: {duplicate}")
    split_by_plant = assign_plant_splits(
        manifest["plant_id"].tolist(), seed=seed, train_ratio=train_ratio, val_ratio=val_ratio
    )
    manifest["split"] = manifest["plant_id"].map(split_by_plant)
    _write_knowledge_tables(manifest, output)


def build_knowledge_files(
    raw_root: Path,
    bbch_xlsx: Path,
    output: Path,
    camera_ids: set[int],
    pose_ids: set[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    image_root = _candidate_image_root(raw_root)
    label_root = _candidate_label_root(raw_root)
    labels = load_bbch_labels(bbch_xlsx)

    rows: list[dict] = []
    image_paths = list(image_root.rglob("*.jpg")) + list(image_root.rglob("*.JPG"))
    for image_path in sorted(set(image_paths)):
        parsed = parse_image(image_path)
        if parsed is None:
            continue
        if int(parsed["camera"]) not in camera_ids or int(parsed["pose"]) not in pose_ids:
            continue
        key = (int(parsed["group"]), int(parsed["plant"]), str(parsed["capture_date"]))
        bbch_stage = labels.get(key)
        if bbch_stage is None:
            continue
        relative = image_path.relative_to(image_root).as_posix()
        rows.append(
            {
                "image_id": f"tomatomap:{relative}",
                "image_path": str(image_path.resolve()),
                "relative_path": relative,
                "label_path": _find_label(image_path, image_root, label_root),
                "group_id": int(parsed["group"]),
                "plant_id": int(parsed["plant"]),
                "capture_date": str(parsed["capture_date"]),
                "capture_timestamp": str(parsed["timestamp"]),
                "camera_id": int(parsed["camera"]),
                "pose_id": int(parsed["pose"]),
                "bbch_stage": int(bbch_stage),
            }
        )

    if not rows:
        raise RuntimeError(
            "No matching images were found. Check raw_root, camera IDs, pose IDs, and workbook path."
        )

    manifest = pd.DataFrame(rows)
    split_by_plant = assign_plant_splits(
        manifest["plant_id"].tolist(), seed=seed, train_ratio=train_ratio, val_ratio=val_ratio
    )
    manifest["split"] = manifest["plant_id"].map(split_by_plant)
    _write_knowledge_tables(manifest, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe TomatoMAP knowledge dataset")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-root", type=Path)
    source.add_argument("--fiftyone-dataset", type=str)
    parser.add_argument("--bbch-xlsx", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/tomatomap_kt5k"))
    parser.add_argument("--camera-ids", type=int, nargs="+", default=[2])
    parser.add_argument("--pose-ids", type=int, nargs="+", default=[1, 4, 7, 10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Ratios must satisfy train > 0, val >= 0, and train + val < 1")
    common = {
        "output": args.output,
        "camera_ids": set(args.camera_ids),
        "pose_ids": set(args.pose_ids),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
    }
    if args.fiftyone_dataset:
        build_knowledge_files_from_fiftyone(dataset_name=args.fiftyone_dataset, **common)
    else:
        if args.bbch_xlsx is None:
            parser.error("--bbch-xlsx is required when --raw-root is used")
        build_knowledge_files(raw_root=args.raw_root, bbch_xlsx=args.bbch_xlsx, **common)


if __name__ == "__main__":
    main()
