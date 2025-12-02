from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_DB = BASE_DIR / "data" / "catalog.db"
DEFAULT_LLM_MODEL = os.getenv("PANDASAI_LLM_MODEL", "gpt-5-mini")
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", 25))

class AppSettings:
    """Simple settings accessor to keep imports tidy."""

    base_dir: Path = BASE_DIR
    upload_dir: Path = UPLOAD_DIR
    catalog_db: Path = CATALOG_DB
    default_llm_model: str = DEFAULT_LLM_MODEL
    upload_max_mb: int = UPLOAD_MAX_MB
    
    # Security
    app_password: str = "cGFzc3dvcmQgcHVyY2hhc2luZw=="
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
