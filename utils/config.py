import os
import numpy as np
import random, torch, os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BASE_DIR = Path(__file__).parent.parent
dotenv_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=dotenv_path)

openai_token = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

if not openai_token or not hf_token or not wandb_token:
    raise ValueError("API keys are not set in environment variables")

# ─────────────────────────── Paths ─────────────────────────────────
DATA_DIR = BASE_DIR / "data"
BASE_GPT2 = BASE_DIR / "model/gpt2"
COT_GPT2 = BASE_DIR / "model/prosqa-cot"
PAUSE_GPT2 = BASE_DIR / "model/pause"
COCONUT_GPT2 = BASE_DIR / "model/prosqa-coconut"
COCONUT_GPT2_U = BASE_DIR / "model/prosqa-coconut-u0.3"
PROSQA_TRAIN = BASE_DIR / "data/prosqa_train.json"
PROSQA_VAL = BASE_DIR / "data/prosqa_valid.json"
PROSQA_TEST = BASE_DIR / "data/prosqa_test.json"
OUTPUTS = BASE_DIR / "outputs"
CONTROL_EXPT = OUTPUTS / "recursionControl"
VQVAE = OUTPUTS / "vqvae"