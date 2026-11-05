# SD 1.5 Image Generation API: Specification

This is an HTTP/JSON API for Stable Diffusion 1.5 image generation. It covers text-to-image, image-to-image,
inpainting, ControlNet, LoRAs, textual-inversion embeddings, custom checkpoints and VAEs, hi-res fix,
and asynchronous jobs with progress, previews and cancellation.

| | |
|---|---|
| Base URL | `http://<host>:8000` (the server listens on all interfaces) |
| Protocol | HTTP/1.1, plain HTTP (no TLS) |
| Request bodies | `application/json` (UTF-8) |
| Interactive docs | `GET /docs` (Swagger UI) and `GET /openapi.json`. These need no key |

---

## 1. Authentication

Every endpoint except `/docs` and `/openapi.json` requires the API key. Send it in **either** of these headers:

```
X-API-Key: <key>
Authorization: Bearer <key>
```

A missing or wrong key returns `401` with a `WWW-Authenticate: Bearer` header:

```json
{"detail": "invalid or missing API key"}
```

The server administrator issues the key.

> Traffic is unencrypted HTTP, so the key and images cross the network in plain text.
> Use the API only on a trusted network.

---

## 2. Conventions

### 2.1 Images in requests
Request fields that take an image (`init_image`, `mask_image`, `controlnets[].image`, `/preprocess` `image`)
expect a **base64-encoded image file** (PNG, JPEG, WebP, and so on). The `data:image/png;base64,` prefix is optional.
EXIF orientation is applied. Transparency is dropped: images are converted to RGB, and masks to grayscale.

### 2.2 Images in responses
Responses return images as **PNG**, either as raw bytes or base64 in JSON. Each generated PNG carries a `tEXt` chunk named
`parameters` with A1111-format generation settings (see [§7](#7-infotext-format)), so tools that read
A1111 "PNG Info" can recover the settings.

### 2.3 Processing model
- The server has a single GPU worker. **Jobs run one at a time in FIFO order.** Every generation
  request, sync or async, goes through the same queue.
- If the client disconnects, a synchronous request keeps running until it finishes.
- Job records live in memory. The last **200** finished jobs are kept. **Everything is lost when the server restarts.**

### 2.4 Errors
Every error response has a JSON body with a `detail` field.

| Status | Meaning |
|---|---|
| `400` | Invalid input that passed schema validation: unknown model, scheduler, LoRA or preprocessor, bad preprocessor params, an undecodable image, an invalid mode combination, or a model that failed to load or download |
| `401` | Missing or invalid API key |
| `404` | Unknown job id, image index not available yet, or no preview yet |
| `409` | (sync endpoints) The job was cancelled while you waited |
| `422` | Schema validation failed: wrong type, out-of-range value, missing `prompt`. Here `detail` is a list of `{loc, msg, type}` |
| `500` | Unexpected server error. `detail` is `"<ExceptionType>: <message>"` |
| `507` | The GPU ran out of memory. Lower the resolution or the hi-res scale |

---

## 3. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | [`/generate`](#31-post-generate) | Generate and block; returns the first image as a PNG |
| `POST` | [`/generate/json`](#32-post-generatejson) | Generate and block; returns all images as base64 |
| `POST` | [`/jobs`](#33-post-jobs) | Queue a generation and return immediately |
| `GET` | [`/jobs`](#34-get-jobs) | List known jobs, newest first |
| `GET` | [`/jobs/{id}`](#35-get-jobsid) | Job status and progress |
| `GET` | [`/jobs/{id}/images/{n}`](#36-get-jobsidimagesn) | Download image `n` of a finished job |
| `GET` | [`/jobs/{id}/preview`](#37-get-jobsidpreview) | Rough live preview of the image being generated |
| `DELETE` | [`/jobs/{id}`](#38-delete-jobsid) | Cancel a job |
| `POST` | [`/preprocess`](#39-post-preprocess) | Run a ControlNet preprocessor and return the control map |
| `GET` | [`/options`](#310-get-options) | Available models, LoRAs, schedulers, etc. |
| `GET` | [`/health`](#311-get-health) | Server and GPU status |

### 3.1 `POST /generate`
**Body:** a [GenerateRequest](#4-generaterequest).
**Response 200:** `image/png`, the **first** image only, even if `num_images > 1`.

| Response header | Value |
|---|---|
| `X-Seed` | Seed of the returned image |
| `X-Elapsed` | Generation time in seconds (excludes time spent queued) |

### 3.2 `POST /generate/json`
**Body:** a [GenerateRequest](#4-generaterequest).
**Response 200:**
```json
{
  "seeds": [1234, 1235],
  "infos": ["<infotext for image 0>", "<infotext for image 1>"],
  "elapsed_s": 27.41,
  "images": ["<base64 PNG>", "<base64 PNG>"]
}
```

### 3.3 `POST /jobs`
**Body:** a [GenerateRequest](#4-generaterequest).
**Response 202:** a [JobStatus](#5-jobstatus) with `status: "queued"`. Poll `GET /jobs/{id}` until `status`
is `done`, `failed` or `cancelled`. Polling every 0.5–1 s is reasonable.

### 3.4 `GET /jobs`
**Response 200:** an array of [JobStatus](#5-jobstatus), newest first (at most ~200 entries).

### 3.5 `GET /jobs/{id}`
**Response 200:** a [JobStatus](#5-jobstatus). **404** if the id is unknown or has expired.

### 3.6 `GET /jobs/{id}/images/{n}`
**Response 200:** `image/png` for image `n` (0-based), with its infotext embedded.
**404** if the job is unknown or image `n` doesn't exist yet. Images become available only when the job is `done`.

### 3.7 `GET /jobs/{id}/preview`
**Response 200:** `image/png`, a low-quality approximation of the latest denoising step. The server decodes it with a
cheap linear mapping, not the VAE. Its size is the generation size (width/8×8 by height/8×8).
**404** until the first step has completed. The preview stays available after the job finishes.

### 3.8 `DELETE /jobs/{id}`
Requests cancellation. **Response 200:** the current [JobStatus](#5-jobstatus).
- A queued job is skipped when it reaches the front of the queue (`status` → `cancelled`).
- A running job stops within one step (`status` → `cancelled`). No images are returned, including any already
  completed in a multi-image job.
- A finished job is not affected.

Cancellation is asynchronous: poll the job until `status` is `cancelled`.

### 3.9 `POST /preprocess`
Runs a ControlNet preprocessor so you can inspect, edit or cache the control map.

**Body:**
| Field | Type | Required | Description |
|---|---|---|---|
| `preprocessor` | string | yes | A name from [§6.3](#63-preprocessors), except `inpaint` |
| `image` | string | yes | Base64 image |
| `params` | object | no | Preprocessor parameters ([§6.3](#63-preprocessors)) |

**Response 200:** `image/png`, the control map. Its shorter side is `image_resolution` (512 by default). **400** for an unknown preprocessor.
This endpoint shares the GPU worker's lock, so it waits for any running generation to finish.
The first use of a preprocessor downloads its weights (tens to hundreds of MB).

### 3.10 `GET /options`
**Response 200:**
```json
{
  "models": ["sd15", "sd15-inpaint", "<files in models/>"],
  "loaded_models": ["sd15"],
  "active_model": "sd15",
  "vaes": ["default", "ft-mse", "ft-ema", "<files in vae/>"],
  "loras": ["<files in loras/>"],
  "embeddings": ["<files in embeddings/>"],
  "controlnets": ["canny", "depth", "...", "<files in controlnets/>"],
  "preprocessors": ["canny", "depth", "...", "inpaint"],
  "schedulers": ["dpmpp_2m", "euler_a", "..."],
  "folders": {"models": "C:\\...\\models", "...": "..."}
}
```
The local folder listings are read fresh on every call. Files added to the server's folders show up without a restart.

### 3.11 `GET /health`
**Response 200:**
```json
{
  "device": "NVIDIA GeForce GTX 1660 Ti",
  "dtype": "torch.float32",
  "low_vram": false,
  "active_model": "sd15",
  "queue": 0,
  "vram_allocated_mb": 4086,
  "vram_reserved_mb": 4214,
  "vram_free_mb": 514
}
```
`queue` is the number of jobs waiting, not counting the one running.

---

## 4. GenerateRequest

The same body is used by `/generate`, `/generate/json` and `/jobs`. Only `prompt` is required.

### 4.1 Mode selection
The inputs present decide the mode. There is no `mode` field.

| `init_image` | `mask_image` | Mode |
|---|---|---|
| – | – | **txt2img** |
| ✓ | – | **img2img** |
| ✓ | ✓ | **inpaint** |
| – | ✓ | `400`: `mask_image requires init_image` |

Adding `controlnets` works in any of the three modes. `hires` works only in txt2img.

### 4.2 Fields

#### Prompting
| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `prompt` | string | **required** | | Positive prompt, in [A1111 syntax](#43-prompt-syntax) |
| `negative_prompt` | string | `""` | | Negative prompt, same syntax |
| `clip_skip` | int | `1` | 1–12 | CLIP layers to skip, in A1111 numbering: `1` = none, `2` = penultimate layer (common for anime models) |

#### Model selection
| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string \| null | `sd15`, or `sd15-inpaint` in inpaint mode | A checkpoint from [§6.1](#61-models) |
| `vae` | string \| null | `default` (the checkpoint's own) | A VAE from [§6.1](#61-models) |
| `scheduler` | string | `dpmpp_2m` | A sampler from [§6.2](#62-schedulers) |
| `loras` | [LoraSpec](#loraspec)[] | `[]` | LoRAs to apply. Any LoRA loaded earlier but left out of the list is disabled for this request |

#### Sampling
| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `width` | int \| null | see [§4.4](#44-output-size) | 64–2048 | Output width; rounded **down** to a multiple of 8 |
| `height` | int \| null | see [§4.4](#44-output-size) | 64–2048 | Output height; rounded down to a multiple of 8 |
| `steps` | int | `25` | 1–150 | Denoising steps. In img2img/inpaint about `steps × strength` steps actually run |
| `guidance_scale` | float | `7.5` | 0–30 | CFG scale. Values ≤ 1 turn off classifier-free guidance, and the negative prompt is then ignored |
| `guidance_rescale` | float | `0.0` | 0–1 | CFG rescale (reduces over-exposure at high CFG). Applied in plain txt2img only |
| `eta` | float | `0.0` | 0–1 | DDIM eta. Only DDIM-family schedulers use it |
| `seed` | int \| null | random | any int | Image *i* uses `(seed + i) mod 2³²` |
| `num_images` | int | `1` | 1–16 | Images are generated **one after another** (not batched), each with its own seed |
| `freeu` | bool | `false` | | FreeU with the recommended SD 1.5 values (s1 0.9, s2 0.2, b1 1.5, b2 1.6) |

#### img2img / inpaint
| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `init_image` | base64 \| null | `null` | | Source image; resized to the output size |
| `strength` | float | `0.75` | >0–1 | Denoising strength. `1.0` ignores the source pixels completely (except outside the mask when inpainting) |
| `mask_image` | base64 \| null | `null` | | Inpaint mask, **white = repaint**, black = keep. Grayscale values blend |
| `invert_mask` | bool | `false` | | Swap the white and black areas of the mask |
| `mask_blur` | int | `4` | 0–64 | Gaussian blur radius (px) applied to the mask, for softer seams |
| `inpaint_padding` | int \| null | `null` | 0–512 | When set, the server crops to the mask's bounding box plus this padding and inpaints that crop at full resolution, then pastes it back (A1111's "only masked"). Gives better detail for small masks |

#### ControlNet
| Field | Type | Default | Description |
|---|---|---|---|
| `controlnets` | [ControlNetSpec](#controlnetspec)[] | `[]` | One entry for a single ControlNet, several for multi-ControlNet |
| `guess_mode` | bool | `false` | ControlNet guess mode, which lets the ControlNet guide the image even with an empty prompt |

#### Hi-res fix
| Field | Type | Default | Description |
|---|---|---|---|
| `hires` | [HiresFix](#hiresfix) \| null | `null` | Two-pass upscale. txt2img only (otherwise `400`) |

#### LoraSpec
| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `name` | string | **required** | | A LoRA from [§6.1](#61-models) |
| `weight` | float | `1.0` | −4–4 | Strength. Negative values invert the effect |

#### ControlNetSpec
| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `model` | string | **required** | | A ControlNet from [§6.1](#61-models) |
| `image` | base64 \| null | `init_image` | | Control image. Leaving it out falls back to `init_image`, which is useful for `tile`. It's an error (`400`) when neither is given |
| `preprocessor` | string \| null | `null` | | A [preprocessor](#63-preprocessors) to run on `image` first. Leave out or use `"none"` if `image` is already a control map |
| `preprocessor_params` | object | `{}` | | Preprocessor parameters ([§6.3](#63-preprocessors)) |
| `scale` | float | `1.0` | 0–2 | Conditioning strength |
| `start` | float | `0.0` | 0–1 | Fraction of the steps at which this ControlNet starts applying |
| `end` | float | `1.0` | 0–1 | Fraction of the steps at which it stops |

The server resizes control images to the output size (and to the upscaled size in the hi-res pass).

#### HiresFix
| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `scale` | float | `2.0` | 1–4 | Upscale factor. The final size is `round8(width × scale) × round8(height × scale)` |
| `upscaler` | `"lanczos"` \| `"latent"` | `"lanczos"` | | `lanczos` decodes the first pass and resizes the pixels; `latent` resizes the latents (bilinear) and usually needs `denoise` ≥ 0.55 |
| `steps` | int \| null | `steps` | 1–150 | Steps that actually run in the second pass |
| `denoise` | float | `0.5` | >0–1 | Second-pass denoising strength |

### 4.3 Prompt syntax
The server follows AUTOMATIC1111 conventions:

| Syntax | Effect |
|---|---|
| `(text)` | Weight × 1.1 (nesting multiplies: `((text))` = × 1.21) |
| `(text:1.4)` | Weight set to 1.4 |
| `[text]` | Weight ÷ 1.1 |
| `\(` `\)` `\[` `\]` | Literal brackets |
| `BREAK` | Starts a new 75-token chunk |
| *Any length* | Prompts longer than 75 tokens are encoded in 75-token chunks and concatenated, so nothing gets cut off |
| `<embedding-name>` | Putting an embedding's name (from `/options` → `embeddings`) in a prompt loads it automatically |

LoRAs go in the `loras` field. The `<lora:name:1>` prompt tag is **not** parsed.

### 4.4 Output size
1. If `width` and `height` are both given, the server uses them, rounded down to multiples of 8 (minimum 64).
2. Otherwise, without an `init_image`, it fills in 512 for the missing dimension.
3. With an `init_image`: if one dimension is given, the other follows the image's aspect ratio. If neither is given,
   it uses the image's size, scaled down so the longer side is at most 768.

SD 1.5 was trained at 512×512. Sizes well above 768 per side tend to produce duplicated subjects, so use
`hires` to get large images instead.

### 4.5 Validation and runtime errors (400)
- `mask_image requires init_image`
- `hires fix only applies to txt2img`
- `unknown scheduler '…'`
- `unknown models|vae|loras|controlnets entry '…'. Available: [...]`
- `'…' is an inpainting checkpoint; use it with init_image + mask_image`
- `could not decode base64 image: …`
- `controlnet '…' needs an image (or init_image)`
- `the 'inpaint' preprocessor needs init_image and mask_image`
- `unknown preprocessor '…'`
- `embedding '…' failed to load: …`
- A Hugging Face repo id that doesn't exist or failed to download

---

## 5. JobStatus

```json
{
  "id": "b34354b9561e",
  "status": "running",
  "queue_position": null,
  "progress": {"image": 1, "of": 2, "phase": "base", "step": 12, "total": 25},
  "seeds": [],
  "infos": [],
  "images": [],
  "error": null,
  "elapsed_s": 6.3
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | 12-character hex job id |
| `status` | string | `queued` → `running` → `done` \| `failed` \| `cancelled` |
| `queue_position` | int \| null | 1-based position while queued, `null` otherwise |
| `progress.image` | int | 1-based index of the image being generated |
| `progress.of` | int | `num_images` |
| `progress.phase` | string | `""` before the first step, then `base`, or `hires` during the hi-res pass |
| `progress.step` / `progress.total` | int | Step within the current phase. `total` is the number of steps that actually run, which differs from `steps` in img2img and hi-res |
| `seeds` | int[] | Filled in when `done` |
| `infos` | string[] | [Infotext](#7-infotext-format) per image, filled in when `done` |
| `images` | string[] | Relative URLs (`/jobs/{id}/images/{n}`) once `done`. They also need the API key |
| `error` | string \| null | Message when `failed` |
| `elapsed_s` | float \| null | Seconds since the job started running (final value once finished). `null` while queued |

---

## 6. Enumerations

### 6.1 Models
Wherever a name is accepted, it resolves in this order: **built-in shorthand → file or folder in the server's local
folder (name without extension) → Hugging Face repo id** (`org/name`). The server downloads a repo id on first use and caches it.
Only SD 1.5-family weights are compatible.

| Kind | Built-ins | Local folder (formats) |
|---|---|---|
| `model` | `sd15` (stable-diffusion-v1-5/stable-diffusion-v1-5), `sd15-inpaint` (stable-diffusion-v1-5/stable-diffusion-inpainting) | `models/` (`.safetensors` single-file checkpoints) |
| `vae` | `default`, `ft-mse` (stabilityai/sd-vae-ft-mse), `ft-ema` | `vae/` (`.safetensors`, or diffusers folders) |
| `loras[].name` | – | `loras/` (`.safetensors`), or a repo id such as `latent-consistency/lcm-lora-sdv1-5` |
| embeddings | – | `embeddings/` (`.safetensors`, `.pt`, `.bin`). Triggered by name in the prompt |
| `controlnets[].model` | see below | `controlnets/` (`.safetensors`, or diffusers folders) |

**Built-in ControlNets** (lllyasviel ControlNet v1.1; ~700 MB download each on first use):

| Name | Use with preprocessor |
|---|---|
| `canny` | `canny` |
| `depth` | `depth`, `depth_zoe` |
| `openpose` | `openpose`, `openpose_full` |
| `lineart` | `lineart`, `lineart_coarse` |
| `lineart_anime` | `lineart_anime` |
| `softedge` | `softedge_hed`, `softedge_pidi` |
| `scribble` | `scribble_hed`, `scribble_pidi`, or a hand-drawn white-on-black scribble |
| `mlsd` | `mlsd` |
| `normalbae` | `normalbae` |
| `seg` | none. Supply an ADE20K-colour segmentation map |
| `inpaint` | `inpaint` (needs `init_image` + `mask_image`) |
| `tile` | none. Uses the source image itself (leave out `image` to use `init_image`) |
| `shuffle` | `shuffle` |
| `ip2p` | none. Uses the source image itself, and the prompt is an instruction ("make it winter") |

Model switching: the active checkpoint lives on the GPU, and the server keeps up to 3 checkpoints in system RAM. A request
for a different checkpoint swaps it in, which takes seconds from RAM or longer from disk. Alternating between
txt2img (`sd15`) and inpainting (`sd15-inpaint`) therefore costs a swap each time.

### 6.2 Schedulers
| Name | A1111 equivalent |
|---|---|
| `dpmpp_2m` (default) | DPM++ 2M |
| `dpmpp_2m_karras` | DPM++ 2M Karras |
| `dpmpp_2m_sde` / `dpmpp_2m_sde_karras` | DPM++ 2M SDE (Karras) |
| `dpmpp_sde` / `dpmpp_sde_karras` | DPM++ SDE (Karras), ~2× slower per step |
| `dpmpp_2s` / `dpmpp_2s_karras` | DPM++ 2S (singlestep) |
| `euler` / `euler_karras` | Euler |
| `euler_a` | Euler a |
| `heun` / `heun_karras` | Heun, ~2× slower per step |
| `dpm2` / `dpm2_karras` | DPM2, ~2× slower per step |
| `dpm2_a` / `dpm2_a_karras` | DPM2 a, ~2× slower per step |
| `lms` / `lms_karras` | LMS |
| `ddim` | DDIM (uses `eta`) |
| `pndm` | PLMS |
| `unipc` | UniPC |
| `deis` | DEIS |
| `lcm` | LCM. Use with an LCM LoRA, 4–8 steps, `guidance_scale` 1–2 |
| `tcd` | TCD. Use with a TCD LoRA, 4–8 steps |

### 6.3 Preprocessors
Preprocessors run on the CPU. Every preprocessor accepts `detect_resolution` (default 512) and `image_resolution` (default 512).

| Name | Extra `preprocessor_params` (defaults) |
|---|---|
| `canny` | `low_threshold` (100), `high_threshold` (200) |
| `depth` | MiDaS: `a` (6.283), `bg_th` (0.1) |
| `depth_zoe` | ZoeDepth: `gamma_corrected` (false) |
| `openpose` | `include_body` (true), `include_hand` (false), `include_face` (false) |
| `openpose_full` | Same as `openpose`, with hands and face on |
| `lineart` | `coarse` (false) |
| `lineart_coarse` | Same as `lineart` with `coarse: true` |
| `lineart_anime` | – |
| `softedge_hed` | `safe` (false) |
| `softedge_pidi` | `safe` (false), `apply_filter` (false) |
| `scribble_hed` | HED with `scribble: true` |
| `scribble_pidi` | PiDiNet with `scribble: true` |
| `mlsd` | `thr_v` (0.1), `thr_d` (0.1) |
| `normalbae` | – |
| `shuffle` | `h`, `w`, `f` (random by default) |
| `inpaint` | Generate requests only (not `/preprocess`): builds the `inpaint` ControlNet condition from `init_image` + `mask_image`; `image` is ignored |

Parameters are passed straight to the preprocessor. Most preprocessors **silently ignore unknown names**, so check
the spelling. A preprocessor that does reject a parameter returns `400`.

---

## 7. Infotext format

The `parameters` PNG chunk, and the `infos[]` entries, use A1111's layout:

```
<prompt>
Negative prompt: <negative_prompt>
Steps: 25, Sampler: dpmpp_2m, CFG scale: 7.5, Seed: 1234, Size: 512x512, Model: sd15[, VAE: …][, Clip skip: 2][, Denoising strength: 0.75][, Lora: name:0.8 …][, ControlNet: canny:1.0 …][, Hires upscale: 2.0, Hires upscaler: lanczos, Hires steps: 15, Hires denoise: 0.5][, FreeU: on]
```
`Size` is the final size, after any hi-res upscale. `Sampler` uses this API's scheduler names.

---

## 8. Performance (GTX 1660 Ti 6 GB, fp32)

| Operation (512×512, 25 steps unless noted) | Typical time |
|---|---|
| txt2img, `dpmpp_2m` / `euler_a` / `ddim` / `unipc` | 11–14 s |
| txt2img, 2nd-order schedulers (`heun`, `dpmpp_sde`, `dpm2`) | ~20 s |
| txt2img with LCM LoRA, 6 steps | ~7 s |
| img2img, strength 0.6 | ~9 s |
| Inpaint, including the checkpoint swap | 20–45 s |
| ControlNet, single / two units | ~27 s / ~49 s |
| Hi-res fix 1.5× (768×768 output) | 60–75 s |
| First use of a new ControlNet, preprocessor or repo | Add download and load time |

Peak VRAM is ~4.9 GB for txt2img, so very large sizes or a big hi-res scale can return `507`.

---

## 9. Examples

Set these first:
```bash
API=http://localhost:8000
KEY=<your api key>
```

### 9.1 txt2img → PNG (curl)
```bash
curl -s -X POST "$API/generate" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a (cozy:1.2) log cabin in a snowy forest, golden hour", "negative_prompt": "blurry, lowres", "seed": 42}' \
  -o cabin.png -D - | grep -i '^x-'
```

### 9.2 Fast generation with an LCM LoRA
```json
{"prompt": "a corgi in a field of flowers",
 "loras": [{"name": "latent-consistency/lcm-lora-sdv1-5", "weight": 1.0}],
 "scheduler": "lcm", "steps": 6, "guidance_scale": 1.5}
```

### 9.3 Hi-res fix
```json
{"prompt": "a lighthouse on a cliff at sunset", "seed": 7,
 "hires": {"scale": 2.0, "upscaler": "lanczos", "steps": 15, "denoise": 0.45}}
```

### 9.4 img2img
```json
{"prompt": "the same scene in winter, snow", "init_image": "<base64>", "strength": 0.6}
```

### 9.5 Inpaint (only the masked region, at full resolution)
```json
{"prompt": "a vase of sunflowers", "init_image": "<base64>", "mask_image": "<base64 white-on-black>",
 "strength": 1.0, "mask_blur": 8, "inpaint_padding": 32}
```

### 9.6 Multi-ControlNet (pose + depth from one photo)
```json
{"prompt": "an astronaut dancing on the moon",
 "controlnets": [
   {"model": "openpose", "image": "<base64 photo>", "preprocessor": "openpose_full", "scale": 1.0},
   {"model": "depth",    "image": "<base64 photo>", "preprocessor": "depth", "scale": 0.6, "end": 0.7}
 ]}
```

### 9.7 Async job with progress, preview and cancel (Python)
```python
import base64, time, requests

API, KEY = "http://localhost:8000", "<your api key>"
H = {"X-API-Key": KEY}

job = requests.post(f"{API}/jobs", headers=H, json={"prompt": "a castle on a hill", "num_images": 2}).json()
while True:
    s = requests.get(f"{API}/jobs/{job['id']}", headers=H).json()
    p = s["progress"]
    print(s["status"], f"image {p['image']}/{p['of']} {p['phase']} {p['step']}/{p['total']}")
    if s["status"] in ("done", "failed", "cancelled"):
        break
    # optional: requests.get(f"{API}/jobs/{job['id']}/preview", headers=H).content -> PNG
    # optional: requests.delete(f"{API}/jobs/{job['id']}", headers=H) to cancel
    time.sleep(1)

if s["status"] == "done":
    for i, url in enumerate(s["images"]):
        open(f"castle_{i}.png", "wb").write(requests.get(API + url, headers=H).content)
else:
    print("error:", s["error"])
```

### 9.8 Preprocess only
```bash
IMG=$(base64 -w0 photo.jpg)
curl -s -X POST "$API/preprocess" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"preprocessor\": \"canny\", \"image\": \"$IMG\", \"params\": {\"low_threshold\": 50}}" -o edges.png
```

### 9.9 PowerShell
```powershell
$h = @{ "X-API-Key" = "<your api key>" }
$b = @{ prompt = "a red apple on a wooden table"; seed = 1 } | ConvertTo-Json -Depth 5
Invoke-WebRequest http://localhost:8000/generate -Method Post -Headers $h -ContentType application/json -Body $b -OutFile apple.png
```
