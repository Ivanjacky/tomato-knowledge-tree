"""Import the public TomatoMAP dataset into a persistent FiftyOne dataset."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="tomato-map-local", help="Local FiftyOne dataset name")
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision")
    args = parser.parse_args()

    import fiftyone as fo
    from fiftyone.utils.huggingface import load_from_hub

    if args.name in fo.list_datasets():
        print(f"Using existing FiftyOne dataset {args.name!r}.")
        return

    dataset = load_from_hub(
        "Voxel51/tomato-map",
        name=args.name,
        revision=args.revision,
        persistent=True,
        overwrite=False,
    )
    print(f"Imported {dataset.name!r} with {len(dataset)} entries in its default view.")


if __name__ == "__main__":
    main()
