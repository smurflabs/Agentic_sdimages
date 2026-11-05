<div align="center">

<img src="docs/logo.png" alt="Agentic" width="288">

**Agentic SD Images: a Stable Diffusion 1.5 server with LoRAs, ControlNet, inpainting and hires fix, behind a simple HTTP API. It powers Agentic's Images tab.**

[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB)
![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900)
![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688)

[Setup](#setup) · [Endpoints](#endpoints) · [Request fields](#request-fields) · [Full API spec](API.md) · [Use with Agentic](#use-with-agentic) · [Agentic server](https://github.com/smurflabs/Agentic)

</div>

---

## What is it?

Agentic SD Images runs Stable Diffusion 1.5 on your own NVIDIA GPU and serves it over HTTP with [FastAPI](https://fastapi.tiangolo.com). The [Agentic server](https://github.com/smurflabs/Agentic) uses it for its **Images** tab and its picture tool in Chat, and any HTTP client can call it directly.

It runs comfortably on a 6 GB card. It was built on a GTX 1660 Ti in fp32, because GTX 16xx cards produce black or NaN images in fp16.

- **Every mode:** text to image, image to image and inpainting, chosen from the inputs you send.
- **The A1111 prompt syntax:** weights like `(word:1.3)`, `[word]` and `BREAK`, with no 77-token limit.
- **Your own files:** checkpoints, LoRAs, textual inversions, VAEs and ControlNets go in folders and are used by file name, with no restart needed.
- **ControlNet:** 14 v1.1 models and 16 built-in preprocessors. Several ControlNets can be combined.
- **Hires fix**, FreeU, clip skip, many schedulers (including LCM and TCD), and up to 16 pictures per request.
- **A job queue:** jobs run in the background with progress and a live preview, and can be cancelled.
- **Readable results:** PNGs carry an A1111-style `parameters` chunk, so "PNG Info" tools can read the settings back.
- **API-key auth.** The key is generated on the first start.

**[API.md](API.md)** is the complete API specification. This README covers setup and operations.

## Setup

You need Python 3.13 and an NVIDIA GPU with 6 GB or more and a driver that supports CUDA 12.6.

```powershell
git clone https://github.com/smurflabs/Agentic_sdimages.git
cd Agentic_sdimages
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1      # the first start downloads SD 1.5 (about 4 GB) and writes api_key.txt
```

The server listens on every interface (`0.0.0.0`) on port 8000. Interactive docs are at `http://<host>:8000/docs`.

### Running as a service

`start-server.cmd` is the launcher for a scheduled task that runs as SYSTEM at boot (for example one named `SD15-API`). It writes its output to `server.log`.

SYSTEM has no user profile, so first edit the paths in `start-server.cmd`:
- `HF_HOME`: the Hugging Face model cache of the account that downloaded the model.
- `PYTHONPATH`: only needed for a split Python install.

To run the server by hand, stop the task first and use `.\run.ps1`.

### API key

Every endpoint except `/docs` and `/openapi.json` needs the key, sent as `X-API-Key: <key>` or `Authorization: Bearer <key>`.

- **Where it is:** in `api_key.txt`, which the server generates on its first start. The `SD_API_KEY` environment variable overrides it.
- **To rotate it:** delete `api_key.txt` and restart the server (`Stop-ScheduledTask SD15-API; Start-ScheduledTask SD15-API`). A new key is generated on startup.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/generate` | Waits until the job is done and returns the first picture as PNG (`X-Seed` and `X-Elapsed` headers) |
| POST | `/generate/json` | Waits until the job is done and returns every picture as base64 PNG, with `seeds` and `infos` |
| POST | `/jobs` | Queues the job and returns its `id` at once. Poll `GET /jobs/{id}` for progress |
| GET | `/jobs/{id}/images/{n}` | Picture *n* of a finished job |
| GET | `/jobs/{id}/preview` | A rough live preview of the picture being denoised |
| DELETE | `/jobs/{id}` | Cancels a queued or running job |
| POST | `/preprocess` | `{preprocessor, image, params}` → a control map PNG |
| GET | `/options` | Lists the models, VAEs, LoRAs, embeddings, ControlNets, preprocessors and schedulers |
| GET | `/health` | The GPU, the active model, the queue length and VRAM |

Images go in as base64 strings; a `data:image/...;base64,` prefix is fine.

## Request fields

The mode follows from the inputs:
- `init_image` alone gives **img2img**.
- `init_image` with `mask_image` gives **inpaint**.

| Field | Default | Notes |
|---|---|---|
| `prompt`, `negative_prompt` | | A1111 syntax: `(word:1.3)`, `(word)`, `[word]`, `\(`, `BREAK`. No 77-token limit |
| `model` | `sd15`, or `sd15-inpaint` when inpainting | Also accepts a `.safetensors` file name (without extension) in `models/`, or a Hugging Face repo id |
| `vae` | `default` | `ft-mse`, `ft-ema`, a file in `vae/`, or a Hugging Face repo id |
| `scheduler` | `dpmpp_2m` | See `/options`. Includes `euler_a`, `dpmpp_2m_karras`, `dpmpp_sde_karras`, `unipc`, `lcm`, `tcd` and more |
| `width`, `height` | 512, or the init image's size (longest side at most 768) | Rounded down to a multiple of 8 |
| `steps` | 25 | img2img actually runs `steps × strength` steps |
| `guidance_scale` | 7.5 | |
| `guidance_rescale`, `eta` | 0 | |
| `clip_skip` | 1 | A1111 numbering. 2 is common for anime models |
| `seed` | random | Picture *i* uses `seed + i` |
| `num_images` | 1 | 1–16, made one after another |
| `loras` | `[]` | `[{"name": "file-name-or-hf-id", "weight": 0.8}]` |
| `freeu` | false | FreeU with the recommended SD 1.5 values |
| `init_image`, `strength` | –, 0.75 | img2img and inpainting |
| `mask_image` | | White = repaint. See also `invert_mask` and `mask_blur` (4) |
| `inpaint_padding` | null | Set it to inpaint only the masked area, plus this many pixels around it, at full resolution |
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

- **`model`:**
  - one of `canny depth openpose lineart lineart_anime softedge scribble mlsd normalbae seg inpaint tile shuffle ip2p` (lllyasviel v1.1), each downloaded on first use (about 700 MB)
  - a file or folder in `controlnets/`
  - a Hugging Face repo id
- **`preprocessor`:** leave it out when `image` is already a control map.
  - The options are `canny depth depth_zoe openpose openpose_full lineart lineart_coarse lineart_anime softedge_hed softedge_pidi scribble_hed scribble_pidi mlsd normalbae shuffle inpaint`.
  - `preprocessor_params` passes extra settings, for example `{"low_threshold": 50}`.
- **`image`:** defaults to `init_image` when left out, which suits `tile` and `inpaint`.
- **Segmentation:** there is no `seg` preprocessor, so supply your own segmentation map.

## Your own files

Drop files into these folders and refer to them by file name without the extension. No restart is needed.

| Folder | Contents |
|---|---|
| `models/` | SD 1.5 checkpoints (`.safetensors`), including fine-tunes and inpainting checkpoints |
| `loras/` | LoRAs (`.safetensors`) |
| `embeddings/` | Textual inversions (`.safetensors`, `.pt`, `.bin`). Use the file name as the word in the prompt |
| `vae/` | VAEs (`.safetensors`) |
| `controlnets/` | ControlNets (`.safetensors`, or diffusers folders) |

Only SD 1.5-family files work. SDXL, SD2 and Flux files fail to load.

## Environment variables

| Variable | Default | What it does |
|---|---|---|
| `SD_API_KEY` | (contents of `api_key.txt`) | The API key to require |
| `SD_MODEL` | `sd15` | The checkpoint loaded at startup and used by default |
| `SD_LOW_VRAM` | 0 | 1 = model CPU offload: slower, but leaves more VRAM free |
| `SD_RAM_CACHE` | 3 | How many checkpoints are kept in system RAM for fast switching |
| `SD_SAFETY_CHECKER` | 0 | 1 = the NSFW filter (about 1.2 GB of VRAM) |
| `SD_DTYPE` | fp32 | Leave this at fp32 on GTX 16xx cards |

## Examples (PowerShell)

```powershell
$api = "http://localhost:8000"
$h = @{"X-API-Key" = (Get-Content .\api_key.txt).Trim()}

# txt2img with hires fix
$b = @{prompt="a (cozy:1.2) cabin, snowy forest"; seed=1; hires=@{scale=2; denoise=0.45}} | ConvertTo-Json
Invoke-WebRequest "$api/generate" -Method Post -Headers $h -ContentType application/json -Body $b -OutFile out.png

# a fast 6-step picture with the LCM LoRA
$b = @{prompt="a corgi"; loras=@(@{name="latent-consistency/lcm-lora-sdv1-5"}); scheduler="lcm"; steps=6; guidance_scale=1.5} | ConvertTo-Json -Depth 4

# ControlNet from a photo
$img = [Convert]::ToBase64String([IO.File]::ReadAllBytes("photo.png"))
$b = @{prompt="oil painting"; controlnets=@(@{model="canny"; image=$img; preprocessor="canny"})} | ConvertTo-Json -Depth 4
```

## Use with Agentic

1. Run this server on the GPU machine.
2. In Agentic's `appsettings.json`, set `"StableDiffusion": { "BaseUrl": "http://<gpu-host>:8000" }`.
3. In the Agentic web app, open **Images → Settings** and paste the key from `api_key.txt`. It is kept in Agentic's encrypted vault and never reaches a browser. You can also set the environment variable `AGENTIC_StableDiffusion__ApiKey`.

The **Images** tab then offers everything this server does: text to image, image to image, inpainting with a mask painter, LoRAs, ControlNet and hires fix. It also keeps a gallery for each user. Agentic sends jobs one at a time and queues the rest, and it can charge User accounts per picture and per GPU minute.

## License

Agentic SD Images is free software under the [GNU General Public License v3.0](LICENSE).

The Stable Diffusion 1.5 weights, and any models you add, come with their own licenses. SD 1.5 uses the CreativeML Open RAIL-M license.
