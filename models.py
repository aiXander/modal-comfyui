"""Model download and cache utilities for ComfyUI on Modal.

All models are downloaded to the /cache volume and symlinked into
ComfyUI's expected model directories. This ensures models are
downloaded once and reused across container restarts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CACHE_DIR = "/cache"


def hf_download(repo_id: str, filename: str, model_dir: str):
    """Download a model from Hugging Face Hub to the cache volume and symlink it."""
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=CACHE_DIR,
    )

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    local_filename = Path(filename).name
    target = Path(model_dir) / local_filename

    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(model_path)

    print(f"Downloaded {repo_id}/{filename} -> {target}")


def download_external_model(url: str, filename: str, model_dir: str):
    """Download a model from an external URL (e.g. CivitAI) using aria2c."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cached_path = Path(CACHE_DIR) / filename

    if not cached_path.exists():
        print(f"Downloading {filename} from {url}...")
        subprocess.run(
            [
                "aria2c",
                "--console-log-level=error",
                "-x", "16",
                "-s", "16",
                "-o", filename,
                "-d", CACHE_DIR,
                url,
            ],
            check=True,
        )
        print(f"Downloaded {filename}")
    else:
        print(f"Using cached {filename}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    target = Path(model_dir) / filename

    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(cached_path)

    print(f"Linked {filename} -> {target}")


def download_models_from_config(models_config: list[dict]):
    """Download all models from a config list.

    Each entry is a dict with keys:
      - repo_id + filename (for HF models)
      - url + filename (for external models)
      - model_dir: target directory inside ComfyUI
    """
    for model in models_config:
        model_dir = model["model_dir"]
        if "repo_id" in model:
            hf_download(model["repo_id"], model["filename"], model_dir)
        elif "url" in model:
            download_external_model(model["url"], model["filename"], model_dir)
        else:
            print(f"Skipping unknown model config: {model}")


def symlink_cached_models(models_config: list[dict]):
    """Re-establish symlinks from cached volume to ComfyUI model dirs.

    Called at container startup (in @modal.enter) to ensure models
    on the volume are visible to ComfyUI.
    """
    for model in models_config:
        model_dir = model["model_dir"]
        filename = Path(model["filename"]).name

        if "repo_id" in model:
            # HF models are stored in the hub cache structure
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(
                model["repo_id"], model["filename"], cache_dir=CACHE_DIR
            )
            if cached and Path(cached).exists():
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                target = Path(model_dir) / filename
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(cached)
        elif "url" in model:
            cached_path = Path(CACHE_DIR) / model["filename"]
            if cached_path.exists():
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                target = Path(model_dir) / filename
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(cached_path)
