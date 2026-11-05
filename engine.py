"""Model management and generation for SD 1.5 on a 6 GB GTX 1660 Ti.

Memory strategy (fp32, since GTX 16xx cards produce NaN/black images in fp16):
- The active checkpoint (UNet + text encoder + VAE, ~4.1 GB) lives on the GPU.
- Other checkpoints (e.g. the inpainting model) wait in system RAM and are swapped in on demand.
- ControlNets (~1.4 GB each) stay in RAM and are streamed to the GPU block by block during
  each step (group offloading), so they never compete with the UNet for VRAM.
"""

import base64
import gc
import inspect
import io
import math
import os
import random
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, Field

import preprocess
from prompts import encode_prompt_pair
from schedulers import SCHEDULERS, make_scheduler

ROOT = Path(__file__).parent
DIRS = {k: ROOT / k for k in ("models", "loras", "embeddings", "vae", "controlnets")}
for d in DIRS.values():
    d.mkdir(exist_ok=True)

DEVICE = "cuda"
DTYPE = torch.float16 if os.getenv("SD_DTYPE", "fp32") == "fp16" else torch.float32
LOW_VRAM = os.getenv("SD_LOW_VRAM", "0") == "1"
SAFETY_CHECKER = os.getenv("SD_SAFETY_CHECKER", "0") == "1"
DEFAULT_MODEL = os.getenv("SD_MODEL", "sd15")
RAM_CACHE = int(os.getenv("SD_RAM_CACHE", "3"))  # checkpoints kept in memory

BUILTIN_MODELS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd15-inpaint": "stable-diffusion-v1-5/stable-diffusion-inpainting",
}
BUILTIN_VAES = {"ft-mse": "stabilityai/sd-vae-ft-mse", "ft-ema": "stabilityai/sd-vae-ft-ema"}
BUILTIN_CONTROLNETS = {
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
    "normalbae": "lllyasviel/control_v11p_sd15_normalbae",
    "seg": "lllyasviel/control_v11p_sd15_seg",
    "inpaint": "lllyasviel/control_v11p_sd15_inpaint",
    "tile": "lllyasviel/control_v11f1e_sd15_tile",
    "shuffle": "lllyasviel/control_v11e_sd15_shuffle",
    "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
}
HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")

PIPELINES = {
    ("txt2img", False): StableDiffusionPipeline,
    ("img2img", False): StableDiffusionImg2ImgPipeline,
    ("inpaint", False): StableDiffusionInpaintPipeline,
    ("txt2img", True): StableDiffusionControlNetPipeline,
    ("img2img", True): StableDiffusionControlNetImg2ImgPipeline,
    ("inpaint", True): StableDiffusionControlNetInpaintPipeline,
}

# Linear approximation of SD 1.5 latents -> RGB, for cheap progress previews
_LATENT_RGB = torch.tensor(
    [[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350], [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]]
)


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- request schema


class LoraSpec(BaseModel):
    name: str = Field(description="File stem in loras/ or a Hugging Face repo id")
    weight: float = Field(1.0, ge=-4, le=4)


class ControlNetSpec(BaseModel):
    model: str = Field(description="Shorthand (canny, depth, openpose, ...), file/folder in controlnets/, or HF repo id")
    image: str | None = Field(None, description="Base64 image; defaults to init_image")
    preprocessor: str | None = Field(None, description="Preprocessor to run on image first; omit if image is already a control map")
    preprocessor_params: dict = {}
    scale: float = Field(1.0, ge=0, le=2)
    start: float = Field(0.0, ge=0, le=1)
    end: float = Field(1.0, ge=0, le=1)


class HiresFix(BaseModel):
    scale: float = Field(2.0, ge=1, le=4)
    upscaler: Literal["lanczos", "latent"] = "lanczos"
    steps: int | None = Field(None, ge=1, le=150, description="Second-pass steps; defaults to steps")
    denoise: float = Field(0.5, gt=0, le=1)


class GenerateRequest(BaseModel):
    prompt: str = Field(description="A1111 syntax: (word:1.3), (word), [word], BREAK; no 77-token limit")
    negative_prompt: str = ""
    model: str | None = Field(None, description="sd15, sd15-inpaint, file stem in models/, or HF repo id")
    vae: str | None = Field(None, description="default, ft-mse, ft-ema, file stem in vae/, or HF repo id")
    scheduler: str = "dpmpp_2m"
    width: int | None = Field(None, ge=64, le=2048)
    height: int | None = Field(None, ge=64, le=2048)
    steps: int = Field(25, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=0, le=30)
    guidance_rescale: float = Field(0.0, ge=0, le=1)
    eta: float = Field(0.0, ge=0, le=1)
    clip_skip: int = Field(1, ge=1, le=12, description="A1111 numbering: 1 = none, 2 = penultimate layer")
    seed: int | None = None
    num_images: int = Field(1, ge=1, le=16, description="Generated one at a time with seeds seed, seed+1, ...")
    loras: list[LoraSpec] = []
    freeu: bool = False
    init_image: str | None = Field(None, description="Base64 image; enables img2img (or inpaint with mask_image)")
    strength: float = Field(0.75, gt=0, le=1)
    mask_image: str | None = Field(None, description="Base64 mask, white = repaint")
    invert_mask: bool = False
    mask_blur: int = Field(4, ge=0, le=64)
    inpaint_padding: int | None = Field(None, ge=0, le=512, description="Inpaint only the masked area plus this padding, at full resolution")
    controlnets: list[ControlNetSpec] = []
    guess_mode: bool = False
    hires: HiresFix | None = None


# ---------------------------------------------------------------- helpers


def decode_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        img.load()
    except Exception as e:
        raise ValueError(f"could not decode base64 image: {e}") from None
    return ImageOps.exif_transpose(img)


def latent_preview(latents: torch.Tensor) -> Image.Image:
    rgb = torch.einsum("chw,cr->hwr", latents[0], _LATENT_RGB)
    img = Image.fromarray(((rgb + 1) / 2).clamp(0, 1).mul(255).byte().numpy())
    return img.resize((img.width * 8, img.height * 8), Image.BILINEAR)


def round8(x: float) -> int:
    return max(64, int(x) // 8 * 8)


def target_size(req: GenerateRequest, init: Image.Image | None) -> tuple[int, int]:
    if req.width and req.height:
        return round8(req.width), round8(req.height)
    if init is None:
        return round8(req.width or 512), round8(req.height or 512)
    w, h = init.size
    if req.width:
        return round8(req.width), round8(req.width * h / w)
    if req.height:
        return round8(req.height * w / h), round8(req.height)
    s = min(1.0, 768 / max(w, h))  # keep large uploads in a size SD 1.5 (and 6 GB) can handle
    return round8(w * s), round8(h * s)


def local_entries(kind: str, exts: tuple[str, ...], dirs: bool = False) -> dict[str, Path]:
    out = {}
    for p in sorted(DIRS[kind].iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
        elif dirs and p.is_dir() and (p / "config.json").exists():
            out[p.name] = p
    return out


def resolve(name: str, kind: str, builtins: dict, exts=(".safetensors",), dirs=False) -> Path | str:
    """Builtin shorthand -> repo id; local file/folder -> Path; otherwise a Hugging Face repo id."""
    if name in builtins:
        return builtins[name]
    local = local_entries(kind, exts, dirs)
    if name in local:
        return local[name]
    if HF_ID.match(name):
        return name
    raise ValueError(f"unknown {kind} entry '{name}'. Available: {sorted([*builtins, *local])}")


def inpaint_condition(init: Image.Image, mask: Image.Image) -> torch.Tensor:
    """Control image for control_v11p_sd15_inpaint: masked pixels set to -1."""
    img = torch.from_numpy(np.array(init.convert("RGB"))).float() / 255
    img[torch.from_numpy(np.array(mask.convert("L"))) > 127] = -1.0
    return img.permute(2, 0, 1).unsqueeze(0)


def resize_control(img, w: int, h: int):
    if isinstance(img, torch.Tensor):
        return F.interpolate(img, size=(h, w), mode="nearest")
    return img.resize((w, h), Image.LANCZOS)


def infotext(req: GenerateRequest, mode: str, model: str, w: int, h: int, seed: int) -> str:
    p = [f"Steps: {req.steps}", f"Sampler: {req.scheduler}", f"CFG scale: {req.guidance_scale}",
         f"Seed: {seed}", f"Size: {w}x{h}", f"Model: {model}"]
    if req.vae:
        p.append(f"VAE: {req.vae}")
    if req.clip_skip > 1:
        p.append(f"Clip skip: {req.clip_skip}")
    if mode != "txt2img":
        p.append(f"Denoising strength: {req.strength}")
    if req.loras:
        p.append("Lora: " + " ".join(f"{l.name}:{l.weight}" for l in req.loras))
    if req.controlnets:
        p.append("ControlNet: " + " ".join(f"{c.model}:{c.scale}" for c in req.controlnets))
    if req.hires:
        p += [f"Hires upscale: {req.hires.scale}", f"Hires upscaler: {req.hires.upscaler}",
              f"Hires steps: {req.hires.steps or req.steps}", f"Hires denoise: {req.hires.denoise}"]
    if req.freeu:
        p.append("FreeU: on")
    return f"{req.prompt}\nNegative prompt: {req.negative_prompt}\n" + ", ".join(p)


# ---------------------------------------------------------------- engine


class Bundle:
    """One checkpoint's components plus the LoRAs/embeddings loaded into them."""

    def __init__(self, name: str, pipe: StableDiffusionPipeline):
        self.name = name
        self.pipe = pipe
        self.is_inpaint = pipe.unet.config.in_channels == 9
        self.adapters: set[str] = set()
        self.embeddings: set[str] = set()


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.bundles: OrderedDict[str, Bundle] = OrderedDict()
        self.active: Bundle | None = None
        self.vaes: dict[str, AutoencoderKL] = {}
        self.controlnets: OrderedDict[str, ControlNetModel] = OrderedDict()
        self.safety = (None, None)
        if SAFETY_CHECKER:
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            from transformers import CLIPImageProcessor

            base = BUILTIN_MODELS["sd15"]
            checker = StableDiffusionSafetyChecker.from_pretrained(base, subfolder="safety_checker", torch_dtype=DTYPE)
            self.safety = (checker if LOW_VRAM else checker.to(DEVICE),
                           CLIPImageProcessor.from_pretrained(base, subfolder="feature_extractor"))

    def warmup(self):
        with self.lock:
            b = self._bundle(DEFAULT_MODEL)
            self._place(b, b.pipe.vae)

    # ---- loading

    def _bundle(self, name: str) -> Bundle:
        if name in self.bundles:
            self.bundles.move_to_end(name)
            return self.bundles[name]
        src = resolve(name, "models", BUILTIN_MODELS)
        kw = dict(torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False)
        if isinstance(src, Path):
            pipe = StableDiffusionPipeline.from_single_file(str(src), **kw)
        else:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(src, use_safetensors=True, **kw)
            except (OSError, ValueError):
                pipe = StableDiffusionPipeline.from_pretrained(src, **kw)
        pipe.vae.enable_slicing()
        b = self.bundles[name] = Bundle(name, pipe)
        while len(self.bundles) > RAM_CACHE:
            old = next((k for k, v in self.bundles.items() if v is not self.active and k != name), None)
            if old is None:
                break
            del self.bundles[old]
            gc.collect()
        return b

    def _vae(self, name: str | None, b: Bundle) -> AutoencoderKL:
        if not name or name == "default":
            return b.pipe.vae
        if name not in self.vaes:
            src = resolve(name, "vae", BUILTIN_VAES, dirs=True)
            if isinstance(src, Path) and src.is_file():
                vae = AutoencoderKL.from_single_file(str(src), torch_dtype=DTYPE)
            else:
                vae = AutoencoderKL.from_pretrained(str(src), torch_dtype=DTYPE)
            vae.enable_slicing()
            self.vaes[name] = vae
        return self.vaes[name]

    def _controlnet(self, name: str) -> ControlNetModel:
        if name in self.controlnets:
            self.controlnets.move_to_end(name)
            return self.controlnets[name]
        src = resolve(name, "controlnets", BUILTIN_CONTROLNETS, dirs=True)
        if isinstance(src, Path) and src.is_file():
            cn = ControlNetModel.from_single_file(str(src), torch_dtype=DTYPE)
        else:
            try:  # fp16 weights are half the download; upcast to fp32 on load
                cn = ControlNetModel.from_pretrained(str(src), torch_dtype=DTYPE, variant="fp16", use_safetensors=True)
            except (OSError, ValueError):
                cn = ControlNetModel.from_pretrained(str(src), torch_dtype=DTYPE)
        if not LOW_VRAM:
            # leaf level keeps only a few layers' weights on the GPU at a time
            cn.enable_group_offload(onload_device=torch.device(DEVICE), offload_type="leaf_level", use_stream=True)
        self.controlnets[name] = cn
        while len(self.controlnets) > 6:
            self.controlnets.popitem(last=False)
            gc.collect()
        return cn

    # ---- placement

    def _place(self, b: Bundle, vae: AutoencoderKL):
        if LOW_VRAM:
            return  # enable_model_cpu_offload() moves modules per call
        if self.active is not b:
            if self.active is not None:
                for m in (self.active.pipe.unet, self.active.pipe.text_encoder, self.active.pipe.vae):
                    m.to("cpu")
            self.active = b
            gc.collect()
            torch.cuda.empty_cache()
        b.pipe.unet.to(DEVICE)  # no-ops when already there
        b.pipe.text_encoder.to(DEVICE)
        for v in [*self.vaes.values(), b.pipe.vae]:
            if v is not vae:
                v.to("cpu")
        vae.to(DEVICE)

    def _load_embeddings(self, b: Bundle, text: str):
        for name, path in local_entries("embeddings", (".safetensors", ".pt", ".bin")).items():
            if name in b.embeddings or name not in text:
                continue
            try:
                b.pipe.load_textual_inversion(str(path), token=name)
            except Exception as e:
                raise ValueError(f"embedding '{name}' failed to load: {e}") from e
            b.embeddings.add(name)

    def _set_loras(self, b: Bundle, loras: list[LoraSpec]):
        names, weights = [], []
        for lora in loras:
            adapter = re.sub(r"\W", "_", lora.name)
            if adapter not in b.adapters:
                src = resolve(lora.name, "loras", {})
                if isinstance(src, Path):
                    b.pipe.load_lora_weights(str(src.parent), weight_name=src.name, adapter_name=adapter)
                else:
                    b.pipe.load_lora_weights(src, adapter_name=adapter)
                b.adapters.add(adapter)
                if not LOW_VRAM:
                    b.pipe.unet.to(DEVICE)
                    b.pipe.text_encoder.to(DEVICE)
            names.append(adapter)
            weights.append(lora.weight)
        if names:
            b.pipe.enable_lora()
            b.pipe.set_adapters(names, adapter_weights=weights)
        elif b.adapters:
            b.pipe.disable_lora()

    def _pipe(self, mode: str, b: Bundle, vae, controlnet, scheduler: str):
        if b.is_inpaint and mode != "inpaint":
            raise ValueError(f"'{b.name}' is an inpainting checkpoint; use it with init_image + mask_image")
        cls = PIPELINES[(mode, controlnet is not None)]
        components = dict(
            vae=vae, text_encoder=b.pipe.text_encoder, tokenizer=b.pipe.tokenizer, unet=b.pipe.unet,
            scheduler=make_scheduler(scheduler, b.pipe.scheduler.config),
            safety_checker=self.safety[0], feature_extractor=self.safety[1], requires_safety_checker=False,
        )
        if controlnet is not None:
            components["controlnet"] = controlnet
        pipe = cls(**components)
        pipe.set_progress_bar_config(disable=True)
        if LOW_VRAM:
            pipe.enable_model_cpu_offload()
        return pipe

    # ---- generation

    @staticmethod
    def _callback(job, phase: str):
        def cb(pipe, step, timestep, kwargs):
            job.phase, job.step, job.total = phase, step + 1, pipe.num_timesteps
            job.latents = kwargs["latents"][:1].detach().float().cpu()
            if job.cancel.is_set():
                pipe._interrupt = True
            return kwargs

        return cb

    @staticmethod
    def _call(pipe, kwargs: dict, job):
        params = inspect.signature(pipe.__call__).parameters
        out = pipe(**{k: v for k, v in kwargs.items() if k in params and v is not None})
        if job.cancel.is_set():
            raise Cancelled()
        return out.images

    def _control_inputs(self, req: GenerateRequest, init, mask, w: int, h: int):
        if not req.controlnets:
            return None, None
        models, images = [], []
        for spec in req.controlnets:
            if spec.preprocessor == "inpaint":
                if init is None or mask is None:
                    raise ValueError("the 'inpaint' preprocessor needs init_image and mask_image")
                images.append(inpaint_condition(init, mask))
            else:
                src = decode_image(spec.image) if spec.image else init
                if src is None:
                    raise ValueError(f"controlnet '{spec.model}' needs an image (or init_image)")
                if spec.preprocessor and spec.preprocessor != "none":
                    src = preprocess.run(spec.preprocessor, src, spec.preprocessor_params)
                images.append(src.convert("RGB").resize((w, h), Image.LANCZOS))
            models.append(self._controlnet(spec.model))
        cn = models[0] if len(models) == 1 else MultiControlNetModel(models)
        return cn, images

    def generate(self, req: GenerateRequest, job):
        with self.lock, torch.inference_mode():
            init = decode_image(req.init_image).convert("RGB") if req.init_image else None
            if req.mask_image and init is None:
                raise ValueError("mask_image requires init_image")
            mode = "inpaint" if req.mask_image else "img2img" if init else "txt2img"
            if req.hires and mode != "txt2img":
                raise ValueError("hires fix only applies to txt2img")
            if req.scheduler not in SCHEDULERS:
                raise ValueError(f"unknown scheduler '{req.scheduler}'. Available: {sorted(SCHEDULERS)}")

            w, h = target_size(req, init)
            mask = None
            if init is not None:
                init = init.resize((w, h), Image.LANCZOS)
            if req.mask_image:
                mask = decode_image(req.mask_image).convert("L").resize((w, h), Image.LANCZOS)
                if req.invert_mask:
                    mask = ImageOps.invert(mask)
                if req.mask_blur:
                    mask = mask.filter(ImageFilter.GaussianBlur(req.mask_blur))

            model = req.model or ("sd15-inpaint" if mode == "inpaint" else DEFAULT_MODEL)
            b = self._bundle(model)
            vae = self._vae(req.vae, b)
            self._place(b, vae)
            self._load_embeddings(b, f"{req.prompt} {req.negative_prompt}")
            self._set_loras(b, req.loras)
            if req.freeu:
                b.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)  # recommended SD 1.5 values
            else:
                b.pipe.disable_freeu()
            controlnet, control_images = self._control_inputs(req, init, mask, w, h)
            if controlnet is not None:
                b.pipe.unet.set_attention_slice("auto")  # less activation memory, a bit slower
            else:
                b.pipe.unet.set_attn_processor(AttnProcessor2_0())  # fast SDPA default

            final = (round8(w * req.hires.scale), round8(h * req.hires.scale)) if req.hires else (w, h)
            if max(final) > 768:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            pos, neg = encode_prompt_pair(b.pipe, req.prompt, req.negative_prompt, req.clip_skip, DEVICE)
            if controlnet is not None and not LOW_VRAM:
                # Text encoder isn't needed while denoising; free its 0.5 GB for ControlNet activations.
                # _place() moves it back for the next job.
                b.pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()
            seed0 = req.seed if req.seed is not None else random.randrange(2**32)
            images, seeds, infos = [], [], []
            for i in range(req.num_images):
                job.image_index = i
                seed = (seed0 + i) % 2**32
                images.append(self._one(req, mode, b, vae, controlnet, control_images, init, mask, w, h, pos, neg, seed, job))
                seeds.append(seed)
                infos.append(infotext(req, mode, model, *final, seed))
            return images, seeds, infos

    def _one(self, req, mode, b, vae, controlnet, control_images, init, mask, w, h, pos, neg, seed, job):
        kw = dict(
            prompt_embeds=pos, negative_prompt_embeds=neg, num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale, guidance_rescale=req.guidance_rescale, eta=req.eta,
            generator=torch.Generator("cpu").manual_seed(seed), width=w, height=h,
            strength=req.strength, padding_mask_crop=req.inpaint_padding,
            callback_on_step_end=self._callback(job, "base"),
            output_type="latent" if req.hires and req.hires.upscaler == "latent" else "pil",
        )
        if mode != "txt2img":
            kw["image"] = init
        if mode == "inpaint":
            kw["mask_image"] = mask
        if controlnet is not None:
            single = len(req.controlnets) == 1
            pick = (lambda xs: xs[0]) if single else list
            kw["image" if mode == "txt2img" else "control_image"] = pick(control_images)
            kw["controlnet_conditioning_scale"] = pick([c.scale for c in req.controlnets])
            kw["control_guidance_start"] = pick([c.start for c in req.controlnets])
            kw["control_guidance_end"] = pick([c.end for c in req.controlnets])
            kw["guess_mode"] = req.guess_mode

        out = self._call(self._pipe(mode, b, vae, controlnet, req.scheduler), kw, job)
        if not req.hires:
            return out[0]

        # Hires fix: upscale the first pass, then refine it with img2img at the larger size
        hr = req.hires
        tw, th = round8(w * hr.scale), round8(h * hr.scale)
        if hr.upscaler == "latent":
            src = F.interpolate(out, size=(th // 8, tw // 8), mode="bilinear")
        else:
            src = out[0].resize((tw, th), Image.LANCZOS)
        kw2 = dict(
            kw, image=src, strength=hr.denoise, output_type="pil",
            num_inference_steps=math.ceil((hr.steps or req.steps) / hr.denoise),
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=self._callback(job, "hires"),
        )
        del kw2["width"], kw2["height"]
        if controlnet is not None:
            resized = [resize_control(c, tw, th) for c in control_images]
            kw2["control_image"] = resized[0] if len(resized) == 1 else resized
        return self._call(self._pipe("img2img", b, vae, controlnet, req.scheduler), kw2, job)[0]

    # ---- misc

    def preprocess(self, name: str, image: str, params: dict) -> Image.Image:
        with self.lock:
            return preprocess.run(name, decode_image(image), params)

    def options(self) -> dict:
        return {
            "models": [*BUILTIN_MODELS, *local_entries("models", (".safetensors",))],
            "loaded_models": list(self.bundles),
            "active_model": self.active.name if self.active else None,
            "vaes": ["default", *BUILTIN_VAES, *local_entries("vae", (".safetensors",), dirs=True)],
            "loras": list(local_entries("loras", (".safetensors",))),
            "embeddings": list(local_entries("embeddings", (".safetensors", ".pt", ".bin"))),
            "controlnets": [*BUILTIN_CONTROLNETS, *local_entries("controlnets", (".safetensors",), dirs=True)],
            "preprocessors": [*preprocess.PREPROCESSORS, "inpaint"],
            "schedulers": list(SCHEDULERS),
            "folders": {k: str(v) for k, v in DIRS.items()},
        }
