"""Sampler/scheduler presets, named after their A1111 equivalents."""

from diffusers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    TCDScheduler,
    UniPCMultistepScheduler,
)

KARRAS = {"use_karras_sigmas": True}
SDE = {"algorithm_type": "sde-dpmsolver++"}

SCHEDULERS = {
    "dpmpp_2m": (DPMSolverMultistepScheduler, {}),
    "dpmpp_2m_karras": (DPMSolverMultistepScheduler, KARRAS),
    "dpmpp_2m_sde": (DPMSolverMultistepScheduler, SDE),
    "dpmpp_2m_sde_karras": (DPMSolverMultistepScheduler, {**SDE, **KARRAS}),
    "dpmpp_sde": (DPMSolverSDEScheduler, {}),
    "dpmpp_sde_karras": (DPMSolverSDEScheduler, KARRAS),
    "dpmpp_2s": (DPMSolverSinglestepScheduler, {}),
    "dpmpp_2s_karras": (DPMSolverSinglestepScheduler, KARRAS),
    "euler": (EulerDiscreteScheduler, {}),
    "euler_karras": (EulerDiscreteScheduler, KARRAS),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "heun": (HeunDiscreteScheduler, {}),
    "heun_karras": (HeunDiscreteScheduler, KARRAS),
    "dpm2": (KDPM2DiscreteScheduler, {}),
    "dpm2_karras": (KDPM2DiscreteScheduler, KARRAS),
    "dpm2_a": (KDPM2AncestralDiscreteScheduler, {}),
    "dpm2_a_karras": (KDPM2AncestralDiscreteScheduler, KARRAS),
    "lms": (LMSDiscreteScheduler, {}),
    "lms_karras": (LMSDiscreteScheduler, KARRAS),
    "ddim": (DDIMScheduler, {}),
    "pndm": (PNDMScheduler, {}),
    "unipc": (UniPCMultistepScheduler, {}),
    "deis": (DEISMultistepScheduler, {}),
    # for use with LCM / TCD LoRAs (4-8 steps, guidance_scale ~1-2)
    "lcm": (LCMScheduler, {}),
    "tcd": (TCDScheduler, {}),
}


def make_scheduler(name: str, base_config):
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler '{name}'. Available: {sorted(SCHEDULERS)}")
    cls, kwargs = SCHEDULERS[name]
    return cls.from_config(base_config, **kwargs)
