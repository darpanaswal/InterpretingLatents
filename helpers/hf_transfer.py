"""Unified Hugging Face Hub transfer helper: upload checkpoints, upload the
thoughts/ dataset folder, and download models/checkpoints.

Usage:
    python helpers/hf_transfer.py upload-checkpoint --local_dir model/gsm/codi/ --repo_id darpanaswal/gsm-codi
    python helpers/hf_transfer.py upload-thoughts --repo_id myuser/thoughts [--private]
    python helpers/hf_transfer.py download --repo_id thomas-ferraz/model_name-pause-GSM8kAug-Jun8-ckpt19 --local_dir model/gsm/llama/pause
"""

import argparse
import os
import re
import shutil

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import HfApi, login, snapshot_download, list_repo_files
from huggingface_hub.errors import HFValidationError

from src.config import hf_token, BASE_DIR, THOUGHTS


def upload_checkpoint(local_dir: str, repo_id: str):
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, exist_ok=True)

    folders = [f for f in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, f))]
    for folder in folders:
        print(f"Uploading {folder}...")
        api.upload_folder(
            folder_path=os.path.join(local_dir, folder),
            path_in_repo=folder,
            repo_id=repo_id,
            commit_message=f"Upload {folder}",
        )


def upload_thoughts(repo_id: str, private: bool):
    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(THOUGHTS),
    )
    print(f"Uploaded {THOUGHTS} to https://huggingface.co/datasets/{repo_id}")


def smart_download(repo_id: str, local_dir: str, **kw):
    # strip scheme + host if a full URL was passed instead of a bare repo id
    repo_id = re.sub(r"^https?://(huggingface\.co|hf\.co)/", "", repo_id.strip())

    try:
        # valid repo id (namespace/repo_name) -> whole repo
        return snapshot_download(repo_id=repo_id, local_dir=local_dir, **kw)
    except HFValidationError:
        ns, name, *rest = repo_id.split("/")
        real_repo = f"{ns}/{name}"
        sub = "/".join(rest)

        # ground truth: actual files in repo
        files = list_repo_files(real_repo)

        # match files whose path is exactly `sub` (file) or under `sub/` (folder)
        matched = [f for f in files if f == sub or f.startswith(sub + "/")]
        if not matched:
            raise FileNotFoundError(
                f"'{sub}' not found in {real_repo}. Repo contains: {files}"
            )

        snapshot_download(
            repo_id=real_repo,
            allow_patterns=matched,
            local_dir=local_dir,
            **kw,
        )

        # flatten: strip leading `sub/` prefix, move up to local_dir
        for f in matched:
            downloaded = os.path.join(local_dir, f)
            rel = os.path.relpath(f, sub) if f != sub else os.path.basename(f)
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True) if os.path.dirname(rel) else None
            shutil.move(downloaded, dest)
        # drop empty top nesting dir
        top = os.path.join(local_dir, rest[0])
        if os.path.isdir(top):
            shutil.rmtree(top)
        return local_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload_ckpt = sub.add_parser("upload-checkpoint", help="Upload all checkpoint folders under a local directory")
    p_upload_ckpt.add_argument("--local_dir", required=True, help="e.g. model/gsm/codi/")
    p_upload_ckpt.add_argument("--repo_id", required=True, help="e.g. darpanaswal/gsm-codi")

    p_upload_thoughts = sub.add_parser("upload-thoughts", help="Upload the thoughts/ folder as a HF dataset")
    p_upload_thoughts.add_argument("--repo_id", required=True, help="e.g. myuser/thoughts")
    p_upload_thoughts.add_argument("--private", action="store_true")

    p_download = sub.add_parser("download", help="Download a model/checkpoint (repo or subfolder) from the Hub")
    p_download.add_argument("--repo_id", required=True, help="repo id, URL, or repo/subpath")
    p_download.add_argument("--local_dir", required=True, help="e.g. model/gsm/llama/pause")

    args = parser.parse_args()

    login(token=hf_token)

    if args.command == "upload-checkpoint":
        upload_checkpoint(args.local_dir, args.repo_id)
    elif args.command == "upload-thoughts":
        upload_thoughts(args.repo_id, args.private)
    elif args.command == "download":
        print(BASE_DIR)
        smart_download(args.repo_id, args.local_dir)


if __name__ == "__main__":
    main()
