# Modal ComfyUI — Automated Workflow Deployment

# TODO:
- create proper textureflow_models.json file and use good urls for all models
- 


Deploy ComfyUI workflows as fast-booting Modal API endpoints with automatic dependency detection, GPU snapshots, and model caching.

## Architecture

- **Per-workflow API endpoints** — each workflow gets its own Modal app with independent scaling, GPU type, and model requirements
- **Shared base image** — ComfyUI + comfy-cli + FastAPI, extended per-workflow with custom nodes and models
- **Interactive mode** — full ComfyUI web UI with persistent volumes for authoring and debugging workflows
- **Headless API mode** — subprocess-based execution with GPU snapshots for sub-3s cold starts

## Project Structure

```
modal-comfyui/
├── config.py              # Central config (volumes, GPU defaults, paths, input node mappings)
├── image.py               # Shared base Modal image builder
├── models.py              # Model download utilities (HF Hub + aria2c)
├── interactive.py         # Interactive ComfyUI UI server
├── deploy_workflow.py     # CLI: analyze workflow → generate endpoint
├── custom_nodes.txt       # Custom node IDs to bake into the image
├── workflows/             # Drop your workflow_api.json files here
│   ├── flux_schnell_api.json      # Sample Flux Schnell txt2img workflow
│   └── flux_schnell_models.json   # Model download config for Flux Schnell
├── endpoints/             # Auto-generated per-workflow endpoint files
│   └── flux_schnell.py            # Generated Flux Schnell endpoint
└── requirements.txt       # Local Python dependencies
```

## Prerequisites

- Python 3.11+
- [Modal](https://modal.com) account and CLI (`pip install modal && modal setup`)
- `pip install -r requirements.txt`

## Workflow

### 1. Author interactively

Launch the full ComfyUI web UI on a temporary URL:

```bash
modal serve interactive.py
```

- Design your workflow visually, test custom nodes, download models
- Models persist to Modal volumes across container restarts
- Custom nodes tested here should be added to `custom_nodes.txt` for permanent inclusion (see below)
- When done, click the gear icon → **Save (API Format)** → save to `workflows/`

### 2. Generate an endpoint

Analyze the workflow and generate a deployable Modal endpoint:

```bash
python deploy_workflow.py workflows/your_workflow_api.json \
  --models-config workflows/your_models.json \
  --name my-workflow \
  --gpu L40S
```

The tool will:
- Auto-detect input parameters (prompts, dimensions, seeds, cfg values)
- Identify output nodes (SaveImage, SaveVideo, etc.)
- Find referenced model files
- Generate a self-contained endpoint file in `endpoints/`

Example output:
```
Detected 3 input parameter(s):
  - width (int): node 5 (EmptyLatentImage.width)
  - height (int): node 5 (EmptyLatentImage.height)
  - prompt (str): node 6 (CLIPTextEncode.text)

Detected 1 output node(s):
  - node 9 (SaveImage)

Generated endpoint: endpoints/my_workflow.py
```

### 3. Review and customize

The generated file in `endpoints/` is a readable, editable Python file. Review the auto-detected input schema, model downloads, and workflow injection logic before deploying.

### 4. Deploy

```bash
# Dev mode (temporary URL)
modal serve endpoints/my_workflow.py

# Production (persistent URL)
modal deploy endpoints/my_workflow.py
```

### 5. Call the API

```bash
curl -X POST https://your-org--comfy-my-workflow-endpoint-run.modal.run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat sitting on a rainbow", "width": 1024, "height": 1024}' \
  --output result.png
```

## Flux Schnell Example

A complete working example is included out of the box:

```bash
# Generate the endpoint
python deploy_workflow.py workflows/flux_schnell_api.json \
  --models-config workflows/flux_schnell_models.json \
  --name flux-schnell

# Deploy it
modal deploy endpoints/flux_schnell.py

# Test it
curl -X POST <endpoint_url> \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful mountain landscape at sunset"}' \
  --output test.png
```

## Custom Nodes

Custom nodes are **baked into the Docker image** at build time, not persisted to volumes. This is necessary because custom nodes install Python packages into `site-packages/` — a volume only persists the mount point, so imports break on restart.

Add node registry IDs to `custom_nodes.txt` (one per line):

```txt
# custom_nodes.txt
comfyui-impact-pack
comfyui-essentials
```

Find node IDs at the [ComfyUI Registry](https://registry.comfy.org/). The workflow:

1. Test the node in interactive mode (`modal serve interactive.py`)
2. Add its registry ID to `custom_nodes.txt`
3. Rebuild — both interactive mode and generated endpoints will include it

Models stay on a Volume (they're large, have no Python deps, and symlink cleanly).

## Model Configuration

Create a JSON file listing models to download during image build:

```json
[
  {
    "repo_id": "Comfy-Org/flux1-schnell",
    "filename": "flux1-schnell-fp8.safetensors",
    "model_dir": "/root/comfy/ComfyUI/models/diffusion_models"
  },
  {
    "url": "https://civitai.com/api/download/models/12345",
    "filename": "my_lora.safetensors",
    "model_dir": "/root/comfy/ComfyUI/models/loras"
  }
]
```

- HuggingFace models: use `repo_id` + `filename`
- External URLs (CivitAI, etc.): use `url` + `filename`
- All models are cached in a Modal Volume and symlinked into ComfyUI on boot

## Auto-Detected Input Types

The deploy tool recognizes these ComfyUI node types as configurable API inputs:

| Node Type | Detected Fields | API Parameter Type |
|---|---|---|
| `CLIPTextEncode` | `text` | `str` (prompt) |
| `KSampler` | `seed`, `steps`, `cfg`, `denoise` | `int` / `float` |
| `KSamplerAdvanced` | `noise_seed`, `steps`, `cfg` | `int` / `float` |
| `EmptyLatentImage` | `width`, `height` | `int` |
| `LoadImage` | `image` | `str` (path) |

## Performance Optimizations

| Optimization | How | Impact |
|---|---|---|
| GPU snapshots | `enable_memory_snapshot` + `enable_gpu_snapshot` | Sub-3s cold starts instead of 15s+ |
| Fast dependency resolution | `uv_pip_install` + `--fast-deps` | Image builds in a fraction of the time |
| Fast model downloads | `HF_XET_HIGH_PERFORMANCE=1` | Maximizes HF Hub bandwidth |
| Volume-based model caching | Download once to Modal Volume, symlink on boot | Zero re-download across containers |
| Snapshot stability | `TORCHINDUCTOR_COMPILE_THREADS=1` | Prevents async compilation breaking CUDA checkpoint |
| Image-baked custom nodes | `custom_nodes.txt` installed at build time | Deterministic, zero boot overhead, GPU-snapshot-compatible |
| Headless execution | `comfy launch --background` + `comfy run` | Proven subprocess pattern from Modal's official examples |

## CLI Reference

```
python deploy_workflow.py <workflow.json> [options]

Options:
  --name NAME              Endpoint name (default: derived from filename)
  --gpu GPU                GPU type (default: L40S)
  --models-config FILE     JSON file with model download configs
  --output-dir DIR         Output directory (default: endpoints/)
  --deploy                 Deploy immediately after generating
```
