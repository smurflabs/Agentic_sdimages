# SD 1.5 API

Stable Diffusion 1.5 behind FastAPI, running on a 6 GB GTX 1660 Ti in fp32. GTX 16xx cards produce black or NaN images in fp16.
Server: `http://10.0.0.6:8000`. It listens on all interfaces (`0.0.0.0`). Interactive docs: `/docs`.
**Full API spec: [API.md](API.md).** This README covers operations.

## Setup

Requires Python 3.13 and an NVIDIA GPU (6 GB+) with a CUDA 12.6-capable driver.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1   # first start downloads SD 1.5 (~4 GB) and writes api_key.txt
```

## Running as a service

The `SD15-API` scheduled task starts it at boot (`start-server.cmd`, logs in `server.log`).
To run it by hand, stop the task first and use `.\run.ps1`.

## API key
Every endpoint except `/docs` and `/openapi.json` requires a key, sent as `X-API-Key: <key>` or `Authorization: Bearer <key>`.
The key is in `api_key.txt`, which the server generated on first start. The `SD_API_KEY` env var overrides it.
To rotate it, delete `api_key.txt` and restart the task (`Stop-ScheduledTask SD15-API; Start-ScheduledTask SD15-API`).
A new key is generated on startup.

## Endpoints

| Method | Path | |
|---|---|---|
| POST | `/generate` | Blocks until done and returns the first image as PNG (`X-Seed`, `X-Elapsed` headers) |
| POST | `/generate/json` | Blocks until done and returns every image as base64 PNG, plus `seeds` and `infos` |
| POST | `/jobs` | Queues the job and returns at once with `id`. Poll `GET /jobs/{id}` for progress |
| GET | `/jobs/{id}/images/{n}` | Image *n* of a finished job |
| GET | `/jobs/{id}/preview` | Rough live preview of the image being denoised |
| DELETE | `/jobs/{id}` | Cancels a queued or running job |
| POST | `/preprocess` | `{preprocessor, image, params}` → control map PNG |
| GET | `/options` | Lists models, VAEs, LoRAs, embeddings, ControlNets, preprocessors and schedulers |
| GET | `/health` | GPU, active model, queue length and VRAM |

Images go in as base64 strings (a `data:image/...;base64,` prefix is fine). PNGs come back with an
A1111-style `parameters` text chunk, so "PNG Info" tools can read the settings back out.

## Request fields

The mode follows from the inputs: `init_image` gives **img2img**, and `init_image` + `mask_image` gives **inpaint**.

| Field | Default | Notes |
|---|---|---|
| `prompt`, `negative_prompt` | | A1111 syntax: `(word:1.3)`, `(word)`, `[word]`, `\(`, `BREAK`. No 77-token limit |
| `model` | `sd15`, or `sd15-inpaint` when inpainting | Also accepts a `.safetensors` file stem in `models/` or an HF repo id |
| `vae` | `default` | `ft-mse`, `ft-ema`, a file in `vae/`, or an HF repo id |
| `scheduler` | `dpmpp_2m` | See `/options`. Includes `euler_a`, `dpmpp_2m_karras`, `dpmpp_sde_karras`, `unipc`, `lcm`, `tcd` and more |
| `width`, `height` | 512, or the init image size (longest side capped at 768) | Rounded down to a multiple of 8 |
| `steps` | 25 | img2img actually runs `steps × strength` steps |
| `guidance_scale` | 7.5 | |
| `guidance_rescale`, `eta` | 0 | |
| `clip_skip` | 1 | A1111 numbering. 2 is common for anime models |
| `seed` | random | Image *i* uses `seed + i` |
| `num_images` | 1 | 1–16, generated one after another |
| `loras` | `[]` | `[{"name": "file-stem-or-hf-id", "weight": 0.8}]` |
| `freeu` | false | FreeU with the recommended SD 1.5 values |
| `init_image`, `strength` | –, 0.75 | img2img / inpaint |
| `mask_image` | | White = repaint. Related fields: `invert_mask`, `mask_blur` (4) |
| `inpaint_padding` | null | Set it to inpaint only the masked area, plus this many pixels of padding, at full resolution |
| `controlnets` | `[]` | See below. More than one entry gives multi-ControlNet |
| `guess_mode` | false | ControlNet guess mode |
| `hires` | null | `{"scale": 2, "upscaler": "lanczos" \| "latent", "steps": 15, "denoise": 0.5}`. txt2img only |

### ControlNet

```json
"controlnets": [
  {"model": "canny", "image": "<b64>", "preprocessor": "canny", "scale": 1.0, "start": 0, "end": 1},
  {"model": "depth", "image": "<b64>", "preprocessor": "depth", "scale": 0.7}
]
```
- **`model`:** `canny depth openpose lineart lineart_anime softedge scribble mlsd normalbae seg inpaint tile shuffle ip2p` (lllyasviel v1.1), a file or folder in `controlnets/`, or an HF repo id. Each one downloads on first use (~700 MB).
- **`preprocessor`:** leave it out if `image` is already a control map. The options are `canny depth depth_zoe openpose openpose_full lineart lineart_coarse lineart_anime softedge_hed softedge_pidi scribble_hed scribble_pidi mlsd normalbae shuffle inpaint`. `preprocessor_params` passes extra arguments, such as `{"low_threshold": 50}`.
- **`image`:** defaults to `init_image` when left out, which suits `tile` and `inpaint`.
- **Not built in:** there's no `seg` preprocessor, so supply your own segmentation map.

## Adding your own files

Drop files into these folders and refer to them by file name without the extension. No restart is needed.

| Folder | Contents |
|---|---|
| `models/` | SD 1.5 checkpoints (`.safetensors`), including fine-tunes and inpainting checkpoints |
| `loras/` | LoRAs (`.safetensors`) |
| `embeddings/` | Textual inversions (`.safetensors`/`.pt`/`.bin`). Use the file name as the token in the prompt |
| `vae/` | VAEs (`.safetensors`) |
| `controlnets/` | ControlNets (`.safetensors`, or diffusers folders) |

Only SD 1.5-family files work. SDXL, SD2 and Flux files will fail to load.

## Environment variables

| Var | Default | |
|---|---|---|
| `SD_MODEL` | `sd15` | Checkpoint loaded at startup and used by default |
| `SD_LOW_VRAM` | 0 | 1 = model CPU offload. Slower, but leaves more VRAM free |
| `SD_RAM_CACHE` | 3 | Number of checkpoints kept in system RAM for fast switching |
| `SD_SAFETY_CHECKER` | 0 | 1 = NSFW filter (~1.2 GB of VRAM) |
| `SD_DTYPE` | fp32 | Leave this as fp32 on GTX 16xx cards |

## Examples (PowerShell)

```powershell
$h = @{"X-API-Key" = (Get-Content C:\Users\Administrator\sd15-api\api_key.txt).Trim()}

# txt2img + hires fix
$b = @{prompt="a (cozy:1.2) cabin, snowy forest"; seed=1; hires=@{scale=2; denoise=0.45}} | ConvertTo-Json
Invoke-WebRequest http://10.0.0.6:8000/generate -Method Post -Headers $h -ContentType application/json -Body $b -OutFile out.png

# fast 6-step generation with the LCM LoRA
$b = @{prompt="a corgi"; loras=@(@{name="latent-consistency/lcm-lora-sdv1-5"}); scheduler="lcm"; steps=6; guidance_scale=1.5} | ConvertTo-Json -Depth 4

# ControlNet from a photo
$img = [Convert]::ToBase64String([IO.File]::ReadAllBytes("photo.png"))
$b = @{prompt="oil painting"; controlnets=@(@{model="canny"; image=$img; preprocessor="canny"})} | ConvertTo-Json -Depth 4
```
