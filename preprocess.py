"""ControlNet preprocessors (controlnet_aux). They run on the CPU so the GPU keeps its VRAM for the UNet."""

import controlnet_aux
from PIL import Image

ANNOTATORS = "lllyasviel/Annotators"

# name: (controlnet_aux class, loads weights from ANNOTATORS, default call kwargs)
PREPROCESSORS = {
    "canny": ("CannyDetector", False, {}),
    "depth": ("MidasDetector", True, {}),
    "depth_zoe": ("ZoeDetector", True, {}),
    "openpose": ("OpenposeDetector", True, {}),
    "openpose_full": ("OpenposeDetector", True, {"include_hand": True, "include_face": True}),
    "lineart": ("LineartDetector", True, {}),
    "lineart_coarse": ("LineartDetector", True, {"coarse": True}),
    "lineart_anime": ("LineartAnimeDetector", True, {}),
    "softedge_hed": ("HEDdetector", True, {}),
    "softedge_pidi": ("PidiNetDetector", True, {}),
    "scribble_hed": ("HEDdetector", True, {"scribble": True}),
    "scribble_pidi": ("PidiNetDetector", True, {"scribble": True}),
    "mlsd": ("MLSDdetector", True, {}),
    "normalbae": ("NormalBaeDetector", True, {}),
    "shuffle": ("ContentShuffleDetector", False, {}),
}

_detectors: dict = {}


def run(name: str, image: Image.Image, params: dict | None = None) -> Image.Image:
    if name not in PREPROCESSORS:
        raise ValueError(f"unknown preprocessor '{name}'. Available: {sorted(PREPROCESSORS) + ['inpaint']}")
    cls_name, has_weights, defaults = PREPROCESSORS[name]
    if cls_name not in _detectors:
        cls = getattr(controlnet_aux, cls_name)
        _detectors[cls_name] = cls.from_pretrained(ANNOTATORS) if has_weights else cls()
    kwargs = {"detect_resolution": 512, "image_resolution": 512, **defaults, **(params or {})}
    try:
        return _detectors[cls_name](image.convert("RGB"), **kwargs).convert("RGB")
    except TypeError as e:  # unknown/badly typed preprocessor_params
        raise ValueError(f"invalid params for preprocessor '{name}': {e}") from e
