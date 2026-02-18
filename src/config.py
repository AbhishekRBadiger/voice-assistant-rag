"""
config.py - Loads and validates configuration from config.yaml
Python 3.10 · Windows
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Optional


# ── Project Root ──────────────────────────────────────────────
# Search upward from src/ until we find config.yaml.
# Works regardless of what the project folder is named.
def _find_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "config.yaml").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent.parent


ROOT_DIR = _find_root()
CONFIG_PATH = ROOT_DIR / "config.yaml"

# Keys whose values are HuggingFace model names, NOT local paths.
_MODEL_NAME_KEYS = {"embedding_model", "spacy_model"}

# Path prefixes that are always relative to project root
_LOCAL_PREFIXES = ("models", "data", "logs", "src", "tests")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML config and resolve relative file paths to absolute."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    def resolve_paths(node: Any, parent_key: str = "") -> Any:
        if isinstance(node, dict):
            return {k: resolve_paths(v, k) for k, v in node.items()}
        if isinstance(node, str):
            # Never touch HuggingFace model name fields
            if parent_key in _MODEL_NAME_KEYS:
                return node
            # Never touch URLs or ~ paths
            if node.startswith("http") or node.startswith("~"):
                return node
            # Only resolve strings that start with known local folder names
            if any(node.startswith(p) for p in _LOCAL_PREFIXES):
                return str(ROOT_DIR / node)
        return node

    cfg = resolve_paths(cfg)

    # Ensure critical directories exist
    for dir_path in [
        cfg["paths"]["actions_dir"],
        cfg["paths"]["logs_dir"],
        cfg["stt"]["model_dir"],
    ]:
        os.makedirs(dir_path, exist_ok=True)

    return cfg


# Singleton config instance
_config: Optional[dict] = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config
