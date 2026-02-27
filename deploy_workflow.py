#!/usr/bin/env python3
"""Automated ComfyUI workflow deployment tool.

Analyzes a workflow_api.json file, detects dependencies and input
parameters, generates a Modal endpoint file, and optionally deploys it.

Usage:
    python deploy_workflow.py workflows/my_workflow_api.json
    python deploy_workflow.py workflows/my_workflow_api.json --name my-workflow --gpu L40S
    python deploy_workflow.py workflows/my_workflow_api.json --deploy
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from config import (
    INPUT_NODE_TYPES,
    OUTPUT_NODE_TYPES,
    MODEL_DIR_MAP,
    DEFAULT_GPU,
    COMFYUI_ROOT,
    CUSTOM_NODES_FILE,
)


def normalize_models_config(config, comfyui_root: str = COMFYUI_ROOT) -> list[dict]:
    """Normalize models config to a canonical list format.

    Supports two formats:
    - Old (list): [{repo_id, filename, model_dir}, ...]
    - New (dict): {"rel/path/model.safetensors": "https://...", ...}

    The new dict format also supports git clone commands as values
    (e.g. "git clone https://huggingface.co/org/repo").
    """
    if isinstance(config, list):
        return config

    if not isinstance(config, dict):
        raise ValueError(f"Unsupported models config type: {type(config)}")

    normalized = []
    for rel_path, source in config.items():
        target_dir = f"{comfyui_root}/{str(Path(rel_path).parent)}"
        filename = Path(rel_path).name
        # Use the full relative path (flattened) as a unique cache key
        cache_key = rel_path.replace("/", "__")

        if isinstance(source, str) and source.strip().startswith("git clone "):
            repo_url = source.strip().removeprefix("git clone ").strip()
            normalized.append({
                "model_dir": target_dir,
                "filename": filename,
                "cache_key": cache_key,
                "git_clone": repo_url,
            })
        else:
            normalized.append({
                "model_dir": target_dir,
                "filename": filename,
                "cache_key": cache_key,
                "url": source,
            })

    return normalized


def analyze_workflow(workflow_path: str) -> dict:
    """Analyze a workflow JSON to extract inputs, outputs, and model references."""
    with open(workflow_path) as f:
        workflow = json.load(f)

    inputs = []
    outputs = []
    models_referenced = []
    seen_params = set()

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        node_inputs = node.get("inputs", {})

        # Detect input parameters
        if class_type in INPUT_NODE_TYPES:
            spec = INPUT_NODE_TYPES[class_type]

            if "field" in spec:
                # Single-field input node (e.g. CLIPTextEncode)
                field = spec["field"]
                if field in node_inputs and not isinstance(node_inputs[field], list):
                    param_name = _unique_param(spec["param_name"], node_id, seen_params)
                    inputs.append({
                        "node_id": node_id,
                        "field": field,
                        "param_name": param_name,
                        "type": spec["type"],
                        "default": node_inputs[field],
                        "class_type": class_type,
                    })
            elif "fields" in spec:
                # Multi-field input node (e.g. KSampler)
                for field, field_spec in spec["fields"].items():
                    if field in node_inputs and not isinstance(node_inputs[field], list):
                        param_name = _unique_param(field, node_id, seen_params)
                        inputs.append({
                            "node_id": node_id,
                            "field": field,
                            "param_name": param_name,
                            "type": field_spec["type"],
                            "default": field_spec.get("default", node_inputs[field]),
                            "class_type": class_type,
                        })

        # Detect output nodes
        if class_type in OUTPUT_NODE_TYPES:
            prefix = node_inputs.get("filename_prefix", "ComfyUI")
            outputs.append({
                "node_id": node_id,
                "class_type": class_type,
                "filename_prefix": prefix,
            })

        # Detect model references
        for field_name, value in node_inputs.items():
            if isinstance(value, str) and _looks_like_model_path(value):
                model_dir = _guess_model_dir(class_type, field_name)
                models_referenced.append({
                    "filename": value,
                    "model_dir": model_dir,
                    "node_id": node_id,
                    "class_type": class_type,
                })

    return {
        "inputs": inputs,
        "outputs": outputs,
        "models": models_referenced,
    }


def _unique_param(base_name: str, node_id: str, seen: set) -> str:
    """Generate a unique parameter name, appending node_id if needed."""
    if base_name not in seen:
        seen.add(base_name)
        return base_name
    unique = f"{base_name}_{node_id}"
    seen.add(unique)
    return unique


def _looks_like_model_path(value: str) -> bool:
    """Heuristic: does this string look like a model filename?"""
    model_extensions = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf")
    return any(value.lower().endswith(ext) for ext in model_extensions)


def _guess_model_dir(class_type: str, field_name: str) -> str:
    """Guess the ComfyUI model subdirectory from the node type and field."""
    type_lower = class_type.lower()
    field_lower = field_name.lower()

    mapping = {
        "checkpoint": "checkpoints",
        "lora": "loras",
        "vae": "vae",
        "controlnet": "controlnet",
        "clip": "clip",
        "upscale": "upscale_models",
        "embedding": "embeddings",
        "unet": "unet",
        "diffusion": "diffusion_models",
    }

    for keyword, subdir in mapping.items():
        if keyword in type_lower or keyword in field_lower:
            return MODEL_DIR_MAP.get(subdir, f"{COMFYUI_ROOT}/models/{subdir}")

    return MODEL_DIR_MAP["checkpoints"]


def generate_endpoint_code(
    workflow_name: str,
    workflow_path: str,
    analysis: dict,
    gpu: str,
    models_config: list[dict] | None = None,
    output_dir: str = "endpoints",
) -> str:
    """Generate the Modal endpoint Python file content."""
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", workflow_name).lower()
    app_name = f"comfy-{safe_name}"

    # Read custom nodes to embed in generated code
    custom_node_ids = []
    if CUSTOM_NODES_FILE.exists():
        for line in CUSTOM_NODES_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                custom_node_ids.append(line)

    # Build the input model fields
    input_fields = []
    input_injections = []
    for inp in analysis["inputs"]:
        py_type = inp["type"]
        default = repr(inp["default"])
        input_fields.append(f'    {inp["param_name"]}: {py_type} = {default}')
        input_injections.append(
            f'        workflow["{inp["node_id"]}"]["inputs"]["{inp["field"]}"] = item.{inp["param_name"]}'
        )

    # Determine the output node for file prefix assignment
    output_node = analysis["outputs"][0] if analysis["outputs"] else None

    # Build models config string
    models_str = "None"
    if models_config:
        models_str = repr(models_config)

    # Generate the file
    code = f'''"""Auto-generated Modal endpoint for workflow: {workflow_name}

Generated by deploy_workflow.py. Review and customize before deploying.

Deploy (API):     modal deploy endpoints/{safe_name}.py
Interactive UI:   modal serve endpoints/{safe_name}.py
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal
import modal.experimental
from pydantic import BaseModel

# --- Configuration ---

WORKFLOW_NAME = "{workflow_name}"
GPU = "{gpu}"
CACHE_DIR = "/cache"
COMFY_PORT = 8000
WORKFLOW_TIMEOUT = 1200

MODELS_CONFIG = {models_str}

# --- Volumes ---

vol = modal.Volume.from_name("comfy-models-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("api-keys")

# --- Image ---

base_image = (
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
    .env({{
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }})
'''

    # Add custom node install commands to generated image
    if custom_node_ids:
        install_lines = "\n".join(f'        "comfy node install {nid}",' for nid in custom_node_ids)
        code += f'''    .run_commands(
{install_lines}
    )
'''

    code += ''')

'''

    # Add model download function if we have models
    if models_config:
        code += '''
def _download_models():
    """Download required models to the cache volume."""
    import subprocess
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    failed = []
    for model in MODELS_CONFIG:
        model_dir = model["model_dir"]
        filename = model["filename"]
        cache_key = model.get("cache_key", filename)

        try:
            if "repo_id" in model:
                cached = hf_hub_download(
                    repo_id=model["repo_id"],
                    filename=filename,
                    cache_dir=CACHE_DIR,
                )
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                target = Path(model_dir) / Path(filename).name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(cached)
                print(f"OK: {model['repo_id']}/{filename}")
            elif "git_clone" in model:
                clone_dir = Path(CACHE_DIR) / cache_key
                if not clone_dir.exists():
                    subprocess.run(
                        ["git", "clone", model["git_clone"], str(clone_dir)],
                        check=True,
                    )
                target = Path(model_dir) / filename
                Path(target.parent).mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    if target.is_symlink():
                        target.unlink()
                    else:
                        import shutil
                        shutil.rmtree(target) if target.is_dir() else target.unlink()
                target.symlink_to(clone_dir)
                print(f"OK: cloned {model['git_clone']}")
            elif "url" in model:
                cached_path = Path(CACHE_DIR) / cache_key
                if not cached_path.exists():
                    subprocess.run(
                        ["aria2c", "--console-log-level=error", "-x", "16", "-s", "16",
                         "-o", cache_key, "-d", CACHE_DIR, model["url"]],
                        check=True,
                    )
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                target = Path(model_dir) / filename
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(cached_path)
                print(f"OK: {filename}")
        except Exception as e:
            source = model.get("repo_id") or model.get("git_clone") or model.get("url", "?")
            print(f"FAILED: {filename} from {source} -- {e}")
            failed.append(filename)

    if failed:
        print(f"\\n{'='*60}")
        print(f"WARNING: {len(failed)}/{len(MODELS_CONFIG)} model downloads failed:")
        for f in failed:
            print(f"  - {f}")
        print(f"{'='*60}\\n")


def _symlink_models():
    """Re-establish symlinks from volume cache to ComfyUI model dirs.

    Called after snapshot restore since volume mounts are fresh.
    """
    from pathlib import Path

    if MODELS_CONFIG is None:
        return
    for model in MODELS_CONFIG:
        model_dir = model["model_dir"]
        filename = model["filename"]
        cache_key = model.get("cache_key", filename)
        if "repo_id" in model:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(
                model["repo_id"], model["filename"], cache_dir=CACHE_DIR
            )
            if cached and Path(cached).exists():
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                target = Path(model_dir) / Path(filename).name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(cached)
        elif "git_clone" in model:
            clone_dir = Path(CACHE_DIR) / cache_key
            if clone_dir.exists():
                target = Path(model_dir) / filename
                Path(target.parent).mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    if target.is_symlink():
                        target.unlink()
                    else:
                        import shutil
                        shutil.rmtree(target) if target.is_dir() else target.unlink()
                target.symlink_to(clone_dir)
        elif "url" in model:
            cached_path = Path(CACHE_DIR) / cache_key
            if cached_path.exists():
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                target = Path(model_dir) / filename
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(cached_path)


image = base_image.run_function(_download_models, volumes={CACHE_DIR: vol}, secrets=[hf_secret])
'''
    else:
        code += '''
def _symlink_models():
    """No models configured — nothing to symlink."""
    pass


image = base_image
'''

    # Compute the relative path from the output dir to the workflow file
    try:
        rel_workflow = str(Path(workflow_path).relative_to(Path.cwd()))
    except ValueError:
        rel_workflow = workflow_path

    # Add workflow file and install deps
    code += f'''
WORKFLOW_LOCAL_PATH = str(Path(__file__).parent.parent / "{rel_workflow}")

image = (
    image
    .add_local_file(WORKFLOW_LOCAL_PATH, "/root/workflow_api.json", copy=True)
    .run_commands(
        "comfy node install-deps --workflow=/root/workflow_api.json"
    )
)

# --- Input Schema ---

class WorkflowInput(BaseModel):
{chr(10).join(input_fields) if input_fields else "    prompt: str = 'a beautiful landscape'"}


# --- App ---

app = modal.App(name="{app_name}", image=image)


@app.cls(
    gpu=GPU,
    volumes={{CACHE_DIR: vol}},
    secrets=[hf_secret],
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={{"enable_gpu_snapshot": True}},
)
@modal.concurrent(max_inputs=5)
class Endpoint:
    port: int = COMFY_PORT

    @modal.enter(snap=True)
    def launch_comfy_background(self):
        """Start ComfyUI server — captured in GPU snapshot for fast restores."""
        cmd = f"comfy launch --background -- --port {{self.port}}"
        subprocess.run(cmd, shell=True, check=True)

    @modal.enter()
    def restore_symlinks(self):
        """Re-establish model symlinks after snapshot restore (volume mounts are fresh)."""
        _symlink_models()

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        """Run a workflow and return the output bytes."""
        self._check_health()

        cmd = f"comfy run --workflow {{workflow_path}} --wait --timeout {{WORKFLOW_TIMEOUT}} --verbose"
        subprocess.run(cmd, shell=True, check=True)

        output_dir = "/root/comfy/ComfyUI/output"
        workflow = json.loads(Path(workflow_path).read_text())

        # Find output files by matching the filename prefix
        file_prefix = None
        for node in workflow.values():
            if node.get("class_type") in {OUTPUT_NODE_TYPES!r}:
                file_prefix = node.get("inputs", {{}}).get("filename_prefix")
                break

        if file_prefix:
            for f in sorted(Path(output_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.name.startswith(file_prefix):
                    return f.read_bytes()

        # Fallback: return the most recently modified file
        files = sorted(Path(output_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0].read_bytes()
        raise FileNotFoundError("No output files generated")

    @modal.fastapi_endpoint(method="POST")
    def run(self, item: WorkflowInput):
        """API endpoint. Accepts workflow inputs, returns generated output."""
        from fastapi import Response

        workflow = json.loads(Path("/root/workflow_api.json").read_text())

        # Inject user inputs into workflow
{chr(10).join(input_injections) if input_injections else "        pass  # No auto-detected inputs — customize manually"}

        # Unique prefix for this request
        client_id = uuid.uuid4().hex
'''

    if output_node:
        code += f'''        workflow["{output_node["node_id"]}"]["inputs"]["filename_prefix"] = client_id
'''
    else:
        code += '''        # No output node detected — you may need to set filename_prefix manually
'''

    code += '''
        # Write temp workflow and execute
        tmp_workflow = f"/tmp/{client_id}.json"
        with open(tmp_workflow, "w") as f:
            json.dump(workflow, f)

        img_bytes = self.infer.local(tmp_workflow)

        # Clean up
        Path(tmp_workflow).unlink(missing_ok=True)

        # Determine media type from output
        media_type = "image/png"
        if img_bytes[:4] == b"\\xff\\xd8\\xff\\xe0":
            media_type = "image/jpeg"
        elif img_bytes[:4] == b"RIFF":
            media_type = "image/webp"

        return Response(img_bytes, media_type=media_type)

    def _check_health(self):
        """Verify ComfyUI server is responsive."""
        import socket
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/system_stats")
            urllib.request.urlopen(req, timeout=5)
        except (socket.timeout, urllib.error.URLError) as e:
            print(f"Health check failed: {e}")
            modal.experimental.stop_fetching_inputs()
            raise Exception("ComfyUI server unhealthy, stopping container")


# --- Interactive UI (modal serve only) ---

@app.function(
    gpu=GPU,
    volumes={{CACHE_DIR: vol}},
    secrets=[hf_secret],
    scaledown_window=300,
    max_containers=1,
    enable_memory_snapshot=True,
    experimental_options={{"enable_gpu_snapshot": True}},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(COMFY_PORT, startup_timeout=120)
def ui():
    """Full ComfyUI browser UI for interactive workflow editing.

    Use: modal serve endpoints/{safe_name}.py
    """
    _symlink_models()
    subprocess.Popen(
        f"comfy launch -- --listen 0.0.0.0 --port {{COMFY_PORT}}",
        shell=True,
    )
'''

    return code


def main():
    parser = argparse.ArgumentParser(
        description="Deploy a ComfyUI workflow as a Modal API endpoint"
    )
    parser.add_argument(
        "workflow",
        help="Path to workflow_api.json file",
    )
    parser.add_argument(
        "--name",
        help="Endpoint name (default: derived from filename)",
    )
    parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help=f"GPU type (default: {DEFAULT_GPU})",
    )
    parser.add_argument(
        "--models-config",
        help="Path to a JSON file with model download configs",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy immediately after generating",
    )
    parser.add_argument(
        "--output-dir",
        default="endpoints",
        help="Directory for generated endpoint files (default: endpoints/)",
    )

    args = parser.parse_args()

    workflow_path = Path(args.workflow).resolve()
    if not workflow_path.exists():
        print(f"Error: {workflow_path} not found")
        sys.exit(1)

    # Derive name from filename
    name = args.name or workflow_path.stem.replace("_api", "").replace("_", "-")
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()

    print(f"Analyzing workflow: {workflow_path}")
    analysis = analyze_workflow(str(workflow_path))

    print(f"\nDetected {len(analysis['inputs'])} input parameter(s):")
    for inp in analysis["inputs"]:
        print(f"  - {inp['param_name']} ({inp['type']}): node {inp['node_id']} ({inp['class_type']}.{inp['field']})")
        print(f"    default: {inp['default']!r}")

    print(f"\nDetected {len(analysis['outputs'])} output node(s):")
    for out in analysis["outputs"]:
        print(f"  - node {out['node_id']} ({out['class_type']})")

    print(f"\nDetected {len(analysis['models'])} model reference(s):")
    for model in analysis["models"]:
        print(f"  - {model['filename']} -> {model['model_dir']}")

    # Load models config if provided
    models_config = None
    if args.models_config:
        with open(args.models_config) as f:
            raw_config = json.load(f)
        models_config = normalize_models_config(raw_config)

    # Generate the endpoint file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{safe_name}.py"

    code = generate_endpoint_code(
        workflow_name=name,
        workflow_path=str(workflow_path),
        analysis=analysis,
        gpu=args.gpu,
        models_config=models_config,
        output_dir=args.output_dir,
    )

    output_file.write_text(code)
    print(f"\nGenerated endpoint: {output_file}")

    # Also save the analysis
    config_file = output_dir / f"{safe_name}_config.json"
    config_file.write_text(json.dumps({
        "workflow": str(workflow_path),
        "name": name,
        "gpu": args.gpu,
        "analysis": analysis,
    }, indent=2))
    print(f"Saved config: {config_file}")

    if args.deploy:
        print(f"\nDeploying {output_file}...")
        subprocess.run(["modal", "deploy", str(output_file)], check=True)
    else:
        print(f"\nTo deploy:  modal deploy {output_file}")
        print(f"To serve:   modal serve {output_file}")


if __name__ == "__main__":
    main()
