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
PROSQA_TRAIN = DATA_DIR / "prosqa_train.json"
PROSQA_VAL = DATA_DIR / "prosqa_valid.json"
PROSQA_TEST = DATA_DIR / "prosqa_test.json"
GSM_TRAIN = DATA_DIR / "gsm_train.json"
GSM_VAL = DATA_DIR / "gsm_valid.json"
GSM_TEST = DATA_DIR / "gsm_test.json"


MODEL_DIR = BASE_DIR / "model"
BASE_GPT2 = MODEL_DIR / "gpt2"
BASE_LLAMA = MODEL_DIR / "llama"

### Finetuned models
PROSQA_MODELS = MODEL_DIR / "prosqa"
GSM_MODELS = MODEL_DIR / "gsm"

OUTPUTS = BASE_DIR / "outputs"
CONTROL_EXPT = OUTPUTS / "superposition"
THOUGHTS = BASE_DIR / "thoughts"
INLP = OUTPUTS / "inlp"
VARIANCE_DECOMPOSITION = OUTPUTS / "variance_decomposition"
THOUGHT_ABLATION = OUTPUTS / "remove_thoughts"