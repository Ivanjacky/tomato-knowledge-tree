from __future__ import annotations

import argparse
from pathlib import Path


def download_model(source: str, model_id: str, output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install modelscope before using --source modelscope") from exc
        downloaded = snapshot_download(model_id, local_dir=str(output))
    elif source == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Install huggingface-hub before using --source huggingface"
            ) from exc
        downloaded = snapshot_download(repo_id=model_id, local_dir=str(output))
    else:
        raise ValueError(f"Unknown model source: {source}")

    model_root = Path(downloaded)
    if not (output / "config.json").is_file() and (model_root / "config.json").is_file():
        output = model_root.resolve()
    if not (output / "config.json").is_file():
        raise FileNotFoundError(f"Model download finished but config.json is missing below {output}")
    print(f"Model ready at {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-VL-Embedding model weights")
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    download_model(args.source, args.model_id, args.output)


if __name__ == "__main__":
    main()
