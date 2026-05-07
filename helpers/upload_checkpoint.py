import os
from src.config import hf_token
from huggingface_hub import HfApi, login

# Initialize API
login(token=hf_token)
api = HfApi()


# Define repository and local path
repo_id = "darpanaswal/gsm-codi"
local_base = "model/gsm/codi/"

# Create repo if needed
api.create_repo(repo_id=repo_id, exist_ok=True)

# Get all checkpoint folders
folders = [f for f in os.listdir(local_base) if os.path.isdir(os.path.join(local_base, f))]

for folder in folders:
    print(f"Uploading {folder}...")
    api.upload_folder(
        folder_path=os.path.join(local_base, folder),
        path_in_repo=folder,
        repo_id=repo_id,
        commit_message=f"Upload {folder}"
    )