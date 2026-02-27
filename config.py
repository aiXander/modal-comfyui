"""Central configuration for Modal ComfyUI deployment."""

import os

# Modal volume names
MODELS_VOLUME_NAME = os.environ.get("COMFY_MODELS_VOLUME", "comfy-models-cache")

# GPU defaults
DEFAULT_GPU = os.environ.get("COMFY_DEFAULT_GPU", "L40S")
DEFAULT_GPU_COUNT = int(os.environ.get("COMFY_GPU_COUNT", "1"))

# Paths inside the container
COMFYUI_ROOT = "/root/comfy/ComfyUI"
MODELS_DIR = f"{COMFYUI_ROOT}/models"
CACHE_DIR = "/cache"

# Scaledown windows (seconds)
INTERACTIVE_SCALEDOWN = 300  # 5 min for interactive work
API_SCALEDOWN = 300  # 5 min for API endpoints

# ComfyUI server port
COMFY_PORT = 8000

# Workflow execution timeout (seconds)
WORKFLOW_TIMEOUT = 1200

# Common model directory mappings (subfolder name -> full path)
MODEL_DIR_MAP = {
    "checkpoints": f"{MODELS_DIR}/checkpoints",
    "loras": f"{MODELS_DIR}/loras",
    "vae": f"{MODELS_DIR}/vae",
    "controlnet": f"{MODELS_DIR}/controlnet",
    "clip": f"{MODELS_DIR}/clip",
    "clip_vision": f"{MODELS_DIR}/clip_vision",
    "upscale_models": f"{MODELS_DIR}/upscale_models",
    "embeddings": f"{MODELS_DIR}/embeddings",
    "unet": f"{MODELS_DIR}/unet",
    "diffusion_models": f"{MODELS_DIR}/diffusion_models",
}

# Node types that represent user-configurable inputs
INPUT_NODE_TYPES = {
    "CLIPTextEncode": {"field": "text", "type": "str", "param_name": "prompt"},
    "KSampler": {
        "fields": {
            "seed": {"type": "int", "default": -1},
            "steps": {"type": "int", "default": 20},
            "cfg": {"type": "float", "default": 7.0},
            "denoise": {"type": "float", "default": 1.0},
        }
    },
    "KSamplerAdvanced": {
        "fields": {
            "noise_seed": {"type": "int", "default": -1},
            "steps": {"type": "int", "default": 20},
            "cfg": {"type": "float", "default": 7.0},
        }
    },
    "LoadImage": {"field": "image", "type": "str", "param_name": "input_image"},
    "EmptyLatentImage": {
        "fields": {
            "width": {"type": "int", "default": 1024},
            "height": {"type": "int", "default": 1024},
        }
    },
}

# Output node types (to find generated files)
OUTPUT_NODE_TYPES = ["SaveImage", "SaveAnimatedWEBP", "SaveVideo", "VHS_VideoCombine"]
