"""Upload the thoughts/ folder to a new Hugging Face dataset repo.

Usage:
    python scripts/upload_thoughts_to_hf.py <username_or_org>/<repo_name> [--private]
"""

import argparse

from huggingface_hub import HfApi

from src.config import hf_token, THOUGHTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="e.g. myuser/thoughts")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(THOUGHTS),
    )

    print(f"Uploaded {THOUGHTS} to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
