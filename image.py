"""Shared base image builder for ComfyUI on Modal."""

import modal

from config import MODELS_VOLUME_NAME, CACHE_DIR


def get_volumes():
    """Create or reference the persistent volumes."""
    models_vol = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)
    return (models_vol,)


def build_base_image() -> modal.Image:
    """Build the shared base image with ComfyUI and comfy-cli installed.

    This image is used by both interactive mode and API endpoints.
    Workflow-specific images extend this with their custom nodes and models.
    """
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "git-lfs", "libgl1-mesa-dev", "libglib2.0-0", "aria2")
        .uv_pip_install(
            "comfy-cli",
            "huggingface-hub",
            "fastapi[standard]==0.115.4",
        )
        .run_commands(
            "comfy --skip-prompt install --fast-deps --nvidia",
            "git lfs install",
        )
        .env({
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        })
        .add_local_python_source("config", "image")
    )
    return image


def build_workflow_image(
    base_image: modal.Image,
    workflow_path: str,
    models_config: list[dict] | None = None,
) -> modal.Image:
    """Extend the base image with workflow-specific custom nodes and models.

    Args:
        base_image: The shared base ComfyUI image.
        workflow_path: Local path to the workflow_api.json file.
        models_config: Optional list of model dicts to download during build.
    """
    from pathlib import Path
    from models import download_models_from_config

    (models_vol,) = get_volumes()

    # Add the workflow file and install its dependencies
    image = (
        base_image
        .add_local_file(workflow_path, "/root/workflow_api.json", copy=True)
        .run_commands(
            "comfy node install-deps --workflow=/root/workflow_api.json --fast-deps"
        )
    )

    # Download models if config provided
    if models_config:
        image = image.run_function(
            download_models_from_config,
            kwargs={"models_config": models_config},
            volumes={CACHE_DIR: models_vol},
        )

    return image
