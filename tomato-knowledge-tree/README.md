# A Multimodal Knowledge Tree Construction Model for Tomato Phenological Knowledge

This repository implements supervised parent placement for tomato images within a five-level BBCH knowledge tree. A frozen Qwen3-VL-Embedding-2B encoder supplies the text and image vectors. The trainable module combines modality-specific projections, modality masks, gated fusion, a Poincaré representation, and directional parent scoring. The decoder links each image to one eligible fine-stage BBCH parent and combines the predicted links with the fixed concept hierarchy.

## Data

The experiments use the [FiftyOne release of TomatoMAP on Hugging Face](https://huggingface.co/datasets/Voxel51/tomato-map). The dataset was created by Yujie Zhang, Sabine Struckmeyer, Andreas Kolb, and Sven Reichardt and is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the [original data deposit](https://doi.org/10.5447/ipk/2025/14) and [dataset paper](https://doi.org/10.1038/s41597-026-06926-9) for attribution.

The preparation script selects camera `pos_2` and pose IDs `1`, `4`, `7`, and `10`. The resulting subset contains 5,372 images from 101 plants across 50 fine-grained BBCH stages. Split seed 42 assigns each plant to one partition, with 3,812 training, 808 validation, and 752 test images. Use the locally generated `split_summary.csv` to check these counts.

The commands below generate fresh experimental outputs locally. Agreement with every numerical result in the manuscript has not been established for this code release.

## Setup on Windows

Open PowerShell in the repository root. Use Conda, Python 3.11, and an NVIDIA GPU with a driver that supports CUDA 12.8. The scripts install PyTorch 2.8, FiftyOne, and the Qwen dependencies in `.conda-env`. The archived study environment used PyTorch 2.8.0+cu128 and FiftyOne 1.21.0.

```powershell
.\scripts\bootstrap_windows.ps1 -Device cu128 -WithQwen -WithFiftyOne
$python = Join-Path $PWD '.conda-env\python.exe'
$env:PYTHONPATH = Join-Path $PWD 'src'
& $python scripts/download_tomatomap.py
```

The download command reuses a registered `tomato-map-local` dataset. For a new import, allow at least 50 GiB of free storage for the dataset and metadata. The optional `--revision` argument accepts a Hugging Face commit ID. When importing the dataset under a different name, pass that name to `-FiftyOneDataset` in the next step.

## Train and evaluate

```powershell
.\scripts\run_full_windows.ps1 -FiftyOneDataset 'tomato-map-local' -ModelSource huggingface -SkipInstall -EmbeddingBatchSize 2 -MaxPixels 262144
& $python -m tomato_kt.evaluate --data-dir data/processed/tomatomap_kt5k --output-dir outputs/tomatomap_qwen3vl
.\scripts\run_multiseed_full_windows.ps1
```

The first command builds the study subset, extracts frozen embeddings, and trains the complete knowledge-tree module with training seed 42. Before training, the script runs a two-sample forward and backward check. The checkpoint is selected by validation MRR. The multiseed script trains seeds 43–46 on the same plant-level partition and aggregates all five runs. Use `-ModelSource modelscope` to obtain the [same pretrained model](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) from ModelScope, or use `-ModelSource existing` with `-ModelPath` to use a local copy of the weights.

Vector extraction saves a per-item cache. After an interruption, use the same project directory, model, and extraction settings to reuse that cache. Generated data go under `data/processed/`, and training, metrics, and tree files go under `outputs/`. The evaluator reports rank-based retrieval metrics, macro-stage performance, fine-stage Macro-F1 over the fixed 50-stage label set, and tree validity.

## Comparisons and controls

After training the five full-model runs, run the ablations and frozen-encoder comparisons in this order. The frozen comparison script uses the Hugging Face CLI installed with the model dependencies.

```powershell
.\scripts\run_ablations_windows.ps1
.\scripts\run_comparisons_windows.ps1
& $python scripts/run_reviewer_experiments.py all
& $python -m tomato_kt.reviewer_results
& $python -m tomato_kt.reviewer_analysis --output-dir outputs/reviewer_experiments/analysis
```

The control runner evaluates supervised classifiers, matched encoder comparisons, camera and pose text controls, concept wording, class-imbalance losses, alternative plant-level partitions, and validation sensitivity. The analysis commands consume locally generated predictions and checkpoints. They calculate plant-level paired differences, confidence intervals, adjusted p-values, error patterns, and gate and geometry diagnostics.

## Code layout

| Path | Contents |
|---|---|
| `src/tomato_kt/` | Data preparation, embedding extraction, parent scoring, training, evaluation, and analysis |
| `configs/tomatomap.yaml` | Complete model settings |
| `configs/ablation_*.yaml`, `configs/reviewer/` | Ablations and supplementary controls |
| `scripts/` | Dataset import and experiment commands |
| `vendor/qwen_embed/` | [Upstream Qwen embedding implementation](https://github.com/QwenLM/Qwen3-VL-Embedding) and its Apache 2.0 license |
