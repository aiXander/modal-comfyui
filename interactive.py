"""Interactive ComfyUI UI on Modal.

Usage:
    modal serve interactive.py      # Dev mode — temporary URL
    modal deploy interactive.py     # Persistent URL

Opens the full ComfyUI web interface. Models persist to volumes.
Custom nodes are baked into the image via custom_nodes.txt.

After authoring a workflow:
1. Click the gear icon in ComfyUI
2. Select "Save (API Format)"
3. Save the file to workflows/ in this project
4. Run: python deploy_workflow.py workflows/your_workflow_api.json
"""

from __future__ import annotations

import subprocess

import modal

from config import (
    CACHE_DIR,
    COMFY_PORT,
    DEFAULT_GPU,
    INTERACTIVE_SCALEDOWN,
    MODELS_VOLUME_NAME,
)
from image import build_base_image, hf_secret

base_image = build_base_image()

models_vol = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

app = modal.App(name="comfyui-interactive", image=base_image)


@app.function(
    max_containers=1,
    gpu=DEFAULT_GPU,
    volumes={
        CACHE_DIR: models_vol,
    },
    secrets=[hf_secret],
    scaledown_window=INTERACTIVE_SCALEDOWN,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(COMFY_PORT, startup_timeout=120)
def ui():
    """Launch the ComfyUI web server for interactive authoring."""
    subprocess.Popen(
        f"comfy launch -- --listen 0.0.0.0 --port {COMFY_PORT}",
        shell=True,
    )
