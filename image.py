"""Shared base image builder for ComfyUI on Modal."""

import modal

from config import MODELS_VOLUME_NAME, CACHE_DIR, CUSTOM_NODES_FILE

hf_secret = modal.Secret.from_name("api-keys")


def get_volumes():
    """Create or reference the persistent volumes."""
    models_vol = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)
    return (models_vol,)


def _read_custom_nodes() -> list[str]:
    """Read custom node IDs from custom_nodes.txt, skipping comments and blanks."""
    if not CUSTOM_NODES_FILE.exists():
        return []
    ids = []
    for line in CUSTOM_NODES_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def build_base_image() -> modal.Image:
    """Build the shared base image with ComfyUI and comfy-cli installed.

    This image is used by both interactive mode and API endpoints.
    Custom nodes listed in custom_nodes.txt are baked in at build time.
    Workflow-specific images extend this with their custom nodes and models.
    """
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "git-lfs", "libgl1-mesa-dev", "libglib2.0-0", "aria2")
        .uv_pip_install(
            "comfy-cli",
            "huggingface-hub",
            "fastapi[standard]==0.115.4",
            "requests",
        )
        .run_commands(
            "comfy --skip-prompt install --fast-deps --nvidia",
            "git lfs install",
        )
        .env({
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        })
    )

    # Install custom nodes from custom_nodes.txt into the image
    node_ids = _read_custom_nodes()
    if node_ids:
        install_cmds = [f"comfy node install {nid}" for nid in node_ids]
        image = image.run_commands(*install_cmds)

    image = image.add_local_python_source("config", "image")

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
            "comfy node install-deps --workflow=/root/workflow_api.json"
        )
    )

    # Download models if config provided
    if models_config:
        image = image.run_function(
            download_models_from_config,
            kwargs={"models_config": models_config},
            volumes={CACHE_DIR: models_vol},
            secrets=[hf_secret],
        )

    return image
