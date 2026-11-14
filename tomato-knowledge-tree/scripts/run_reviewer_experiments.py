from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".conda-env" / "python.exe"
OUTPUT_ROOT = ROOT / "outputs" / "reviewer_experiments"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def _run(arguments: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), *arguments]
    print(f"RUN {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nCOMMAND: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if "Final test metrics" in line or "Validation-only metrics" in line:
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}; see {log_path}")


def prepare() -> None:
    _run(
        ["-m", "tomato_kt.reviewer_prepare", "--root", str(ROOT)],
        OUTPUT_ROOT / "prepare.log",
    )


def build_embeddings() -> None:
    embedding_root = ROOT / "data" / "processed" / "reviewer_embeddings"
    embedding_root.mkdir(parents=True, exist_ok=True)
    common = [
        "--data-dir",
        str(ROOT / "data" / "processed" / "tomatomap_kt5k"),
        "--descriptions",
        str(ROOT / "configs" / "bbch_descriptions_en.json"),
        "--original-qwen-embeddings",
        str(ROOT / "data" / "processed" / "tomatomap_kt5k" / "embeddings.npz"),
    ]
    for variant in ("official", "contextual", "concise"):
        target = embedding_root / f"qwen_{variant}.npz"
        if target.exists():
            print(f"SKIP existing {target}", flush=True)
            continue
        _run(
            [
                "-m",
                "tomato_kt.reviewer_embeddings",
                "--encoder",
                "qwen",
                "--output",
                str(target),
                "--prompt-variant",
                variant,
                "--model-path",
                str(ROOT / "models" / "Qwen3-VL-Embedding-2B"),
                *common,
            ],
            OUTPUT_ROOT / "embedding_logs" / f"qwen_{variant}.log",
        )
    hf_specs = {
        "clip": (
            ROOT / "models" / "CLIP-ViT-B-32",
            ROOT
            / "data"
            / "processed"
            / "tomatomap_kt5k"
            / "baseline_cache"
            / "clip_vit_b32",
        ),
        "siglip2": (
            ROOT / "models" / "SigLIP2-base-patch16-224",
            ROOT
            / "data"
            / "processed"
            / "tomatomap_kt5k"
            / "baseline_cache"
            / "siglip2_base_224",
        ),
    }
    for encoder, (model_path, cache_dir) in hf_specs.items():
        target = embedding_root / f"{encoder}_official.npz"
        if target.exists():
            print(f"SKIP existing {target}", flush=True)
            continue
        _run(
            [
                "-m",
                "tomato_kt.reviewer_embeddings",
                "--encoder",
                encoder,
                "--output",
                str(target),
                "--prompt-variant",
                "official",
                "--model-path",
                str(model_path),
                "--image-cache-dir",
                str(cache_dir),
                *common,
            ],
            OUTPUT_ROOT / "embedding_logs" / f"{encoder}_official.log",
        )


def _complete_tree_run(output_dir: Path, validation_only: bool) -> bool:
    required = ["best_model.pt", "history.json", "metrics.json", "predicted_edges.csv"]
    if not validation_only:
        required.extend(["constructed_tree_edges.csv", "supplementary_metrics.json"])
    return all((output_dir / name).exists() for name in required)


def run_tree_groups(groups: set[str] | None = None) -> None:
    plan = pd.read_csv(OUTPUT_ROOT / "run_plan.csv")
    if groups:
        plan = plan[plan["group"].astype(str).isin(groups)]
    for row in plan.itertuples(index=False):
        config_path = Path(str(row.config))
        validation_only = str(row.group) == "sensitivity"
        for seed in [int(value) for value in str(row.seeds).split(",")]:
            output_dir = OUTPUT_ROOT / str(row.group) / str(row.name) / f"seed{seed}"
            if _complete_tree_run(output_dir, validation_only):
                print(f"SKIP complete {output_dir}", flush=True)
                continue
            _run(
                [
                    "-m",
                    "tomato_kt.train",
                    "--config",
                    str(config_path),
                    "--device",
                    "cuda",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(output_dir),
                ],
                output_dir / "run.log",
            )
            if not validation_only:
                data_dir = _config_data_dir(config_path)
                _run(
                    [
                        "-m",
                        "tomato_kt.evaluate",
                        "--data-dir",
                        str(data_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                    output_dir / "evaluate.log",
                )
        if not validation_only:
            seeds = [int(value) for value in str(row.seeds).split(",")]
            _aggregate(
                {seed: OUTPUT_ROOT / str(row.group) / str(row.name) / f"seed{seed}" for seed in seeds},
                OUTPUT_ROOT / str(row.group) / str(row.name) / "summary",
            )


def _config_data_dir(config_path: Path) -> Path:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    value = Path(str(config["data_dir"]))
    return value if value.is_absolute() else ROOT / value


def run_supervised() -> None:
    data_dir = ROOT / "data" / "processed" / "tomatomap_kt5k"
    embeddings = data_dir / "embeddings.npz"
    for mode in ("linear_image", "gated_mlp", "hierarchical_gated"):
        runs: dict[int, Path] = {}
        for seed in (42, 43, 44, 45, 46):
            output_dir = OUTPUT_ROOT / "supervised_baselines" / mode / f"seed{seed}"
            runs[seed] = output_dir
            if _complete_tree_run(output_dir, validation_only=False):
                print(f"SKIP complete {output_dir}", flush=True)
                continue
            _run(
                [
                    "-m",
                    "tomato_kt.supervised_baseline",
                    "--data-dir",
                    str(data_dir),
                    "--embeddings",
                    str(embeddings),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--device",
                    "cuda",
                ],
                output_dir / "run.log",
            )
        _aggregate(runs, OUTPUT_ROOT / "supervised_baselines" / mode / "summary")


def _aggregate(runs: dict[int, Path], output_dir: Path) -> None:
    arguments = ["-m", "tomato_kt.aggregate_runs"]
    for seed, run_dir in sorted(runs.items()):
        arguments.extend(["--run", f"{seed}={run_dir}"])
    arguments.extend(["--output-dir", str(output_dir)])
    _run(arguments, output_dir / "aggregate.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-requested experiment suite")
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "embeddings",
            "supervised",
            "tree_controls",
            "splits",
            "sensitivity",
            "all",
        ],
    )
    args = parser.parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"embeddings", "all"}:
        build_embeddings()
    if args.stage in {"supervised", "all"}:
        run_supervised()
    if args.stage in {"tree_controls", "all"}:
        run_tree_groups(
            {"text_controls", "imbalance", "encoder_control", "prompt_sensitivity"}
        )
    if args.stage in {"splits", "all"}:
        run_tree_groups({"split_robustness"})
    if args.stage in {"sensitivity", "all"}:
        run_tree_groups({"sensitivity"})
    report = {"stage": args.stage, "status": "completed"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
