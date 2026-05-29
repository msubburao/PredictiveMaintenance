from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "msubburao/predictivemaintenanceds"
repo_type = "dataset"

hf_token=os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN not found. Set it from Colab before running this file.")

# Initialize API client

#login(token=hf_token)
api = HfApi(token=hf_token)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="/content/predictive_maintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
