import os
import re

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from src.config import hf_token, BASE_DIR
from huggingface_hub import login, snapshot_download, list_repo_files
from huggingface_hub.errors import HFValidationError

print(BASE_DIR)

login(token=hf_token)

REPO_ID = "https://huggingface.co/thomas-ferraz/model_name-pause-GSM8kAug-Jun8-ckpt19"
LOCAL_DIR = f"{BASE_DIR}/model/gsm/llama/pause"


def smart_download(repo_id, local_dir, **kw):
    # strip scheme + host if a full URL was passed instead of a bare repo id
    repo_id = re.sub(r"^https?://(huggingface\.co|hf\.co)/", "", repo_id.strip())

    try:
        # valid repo id (namespace/repo_name) → whole repo
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
        import shutil
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


smart_download(REPO_ID, LOCAL_DIR)