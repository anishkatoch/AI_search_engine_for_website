"""Configuration loader.

Secrets come from `.env` (via python-dotenv); everything else comes from
`config.yaml`. Any value can be overridden by an environment variable of the
same UPPER_CASE name (handy for Docker / CI).
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repo root (this file lives in <root>/app/config.py).
BASE_DIR = Path(__file__).resolve().parent.parent

# Load secrets from .env into the environment.
load_dotenv(BASE_DIR / ".env")


def _load_yaml() -> dict:
    path = Path(os.getenv("CONFIG_FILE", BASE_DIR / "config.yaml"))
    if path.is_file():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


_cfg = _load_yaml()
_app = _cfg.get("app", {})
_data = _cfg.get("data", {})
_models = _cfg.get("models", {})
_search = _cfg.get("search", {})
_hybrid = _search.get("hybrid", {})
_recommend = _cfg.get("recommend", {})
_upload = _cfg.get("upload", {})


def _path(value: str) -> str:
    """Resolve a path relative to the repo root, unless it's absolute or a URL."""
    if value.startswith(("http://", "https://")) or os.path.isabs(value):
        return value
    return str(BASE_DIR / value)


# ----- Secrets (environment / .env only) -----
OPEN_API_KEY = os.getenv("OPEN_API_KEY", "")
USE_OPENAI = bool(OPEN_API_KEY)

# HuggingFace token is optional. huggingface_hub reads HF_TOKEN from the env
# automatically, but an *empty* value can trigger auth errors — so keep the var
# only when it holds a real token, otherwise remove it (anonymous access).
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    os.environ.pop("HF_TOKEN", None)

# ----- Server -----
HOST = os.getenv("HOST", str(_app.get("host", "0.0.0.0")))
PORT = int(os.getenv("PORT", _app.get("port", 8005)))

# ----- Data -----
PRODUCT_CSV = _path(os.getenv("PRODUCT_CSV", _data.get("product_csv", "amazon_products.csv")))
EMBEDDINGS_FILE = _path(os.getenv("EMBEDDINGS_FILE", _data.get("embeddings_file", "product_embeddings.npy")))

# ----- Models -----
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", _models.get("huggingface", "all-MiniLM-L6-v2"))
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", _models.get("openai_embedding", "text-embedding-3-large"))

# ----- Search -----
TOP_K = int(os.getenv("TOP_K", _search.get("top_k", 80)))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", _search.get("similarity_threshold", 0.4)))
RESULTS_PER_PAGE = int(os.getenv("RESULTS_PER_PAGE", _search.get("results_per_page", 10)))
SEMANTIC_WEIGHT = float(_hybrid.get("semantic_weight", 0.7))
KEYWORD_WEIGHT = float(_hybrid.get("keyword_weight", 0.3))

# ----- Recommendations -----
REC_SIMILAR_COUNT = int(_recommend.get("similar_count", 8))
REC_CATEGORY_COUNT = int(_recommend.get("category_count", 8))
REC_TRENDING_COUNT = int(_recommend.get("trending_count", 12))
REC_CATEGORY_BOOST = float(_recommend.get("category_boost", 0.05))

# ----- Upload -----
UPLOAD_MAX_ROWS = int(_upload.get("max_rows", 20000))

# ----- Frontend -----
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
