#!/usr/bin/env python3
"""
Upload the StrawberryBench dataset to HuggingFace Hub.

Prerequisites:
  1. Generate the dataset:  uv run python data/generate_dataset.py
  2. Login to HuggingFace:  huggingface-cli login
  3. Create the dataset repo on HF Hub (or let this script create it)

Run:
    uv run python data/upload_to_hf.py
    uv run python data/upload_to_hf.py --repo-id my-org/strawberry-bench
"""

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import json

REPO_ID_DEFAULT = "floleuerer/strawberry-bench"
DATA_DIR = Path(__file__).parent


def load_dataset_from_json(path: Path) -> Dataset:
    data = json.loads(path.read_text())
    return Dataset.from_list(data)


def upload(repo_id: str, private: bool = False) -> None:
    jsonl_path = DATA_DIR / "dataset.jsonl"
    json_path = DATA_DIR / "dataset.json"
    card_path = DATA_DIR / "dataset_card.md"

    if jsonl_path.exists():
        data = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line]
    elif json_path.exists():
        data = json.loads(json_path.read_text())
    else:
        raise FileNotFoundError(
            "No dataset found. Run `uv run python data/generate_dataset.py` first."
        )

    print(f"Loaded {len(data)} examples")

    ds = Dataset.from_list(data)
    ds_dict = DatasetDict({"test": ds})

    print(f"Pushing to {repo_id} ...")
    ds_dict.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Add StrawberryBench dataset",
    )
    print(f"Dataset pushed: https://huggingface.co/datasets/{repo_id}")

    # Upload the dataset card as README.md
    if card_path.exists():
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print("Dataset card uploaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload StrawberryBench to HuggingFace")
    parser.add_argument("--repo-id", default=REPO_ID_DEFAULT, help="HuggingFace repo ID")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()

    upload(repo_id=args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
