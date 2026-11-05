"""Stable Diffusion 1.5 API (FastAPI + diffusers), tuned for a 6 GB GTX 1660 Ti.

All generation goes through one queue/worker (the GPU fits one job at a time).
  POST   /generate            -> first image as PNG (blocks until done)
  POST   /generate/json       -> all images as base64 PNG + seeds + infotext (blocks)
  POST   /jobs                -> {"id"}; poll GET /jobs/{id} for progress
  GET    /jobs/{id}/images/{n}, /jobs/{id}/preview, DELETE /jobs/{id} (cancel)
  POST   /preprocess          -> run a ControlNet preprocessor, returns PNG
  GET    /options, /health
All endpoints require the API key (X-API-Key header or Authorization: Bearer).
See API.md for the full spec and README.md for operations.
"""

import base64
import io
import logging
import os
import queue
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

import torch
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import Response
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel

import engine
from engine import Cancelled, GenerateRequest

log = logging.getLogger("uvicorn.error")
MAX_JOBS = 200  # finished jobs kept for polling/downloading
KEY_FILE = engine.ROOT / "api_key.txt"


def load_api_key() -> str:
    """SD_API_KEY env var, else api_key.txt, else generate one into api_key.txt."""
    if key := os.getenv("SD_API_KEY", "").strip():
        return key
    if KEY_FILE.exists() and (key := KEY_FILE.read_text().strip()):
        return key
    key = secrets.token_urlsafe(32)
    KEY_FILE.write_text(key + "\n")
    return key


API_KEY = load_api_key()


def require_api_key(
    header_key: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False)),
    bearer: HTTPAuthorizationCredentials | None = Security(HTTPBearer(auto_error=False)),
):
    supplied = header_key or (bearer.credentials if bearer else None)
    if not supplied or not secrets.compare_digest(supplied.encode(), API_KEY.encode()):
        raise HTTPException(401, "invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})


class Job:
    def __init__(self, req: GenerateRequest):
        self.id = uuid.uuid4().hex[:12]
        self.req = req
        self.status = "queued"
        self.created = time.time()
        self.started = self.finished = None
        self.image_index = self.step = self.total = 0
        self.phase = ""
        self.latents = None
        self.images, self.seeds, self.infos = [], [], []
        self.error, self.code = None, 500
        self.cancel = threading.Event()
        self.done = threading.Event()


eng: engine.Engine | None = None
jobs: OrderedDict[str, Job] = OrderedDict()
work: queue.Queue[Job] = queue.Queue()


def worker():
    while True:
        job = work.get()
        if job.cancel.is_set():
            job.status = "cancelled"
            job.done.set()
            continue
        job.status, job.started = "running", time.time()
        try:
            job.images, job.seeds, job.infos = eng.generate(job.req, job)
            job.status = "done"
        except Cancelled:
            job.status = "cancelled"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            job.status, job.code = "failed", 507
            job.error = "CUDA out of memory: lower resolution/hires scale or set SD_LOW_VRAM=1"
        except (ValueError, OSError) as e:
            job.status, job.code, job.error = "failed", 400, str(e)
        except Exception as e:
            log.exception("job %s failed", job.id)
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            # Drop cached blocks: on Windows, allocations that spilled to shared system memory
            # (driver sysmem fallback) would otherwise be reused and slow every later job.
            peak = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            job.finished = time.time()
            log.info("job %s %s in %.1fs, peak VRAM %.0f MB", job.id, job.status, job.finished - job.started, peak)
            job.done.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global eng
    eng = engine.Engine()
    eng.warmup()
    threading.Thread(target=worker, daemon=True, name="sd-worker").start()
    yield


# Every route requires the key; /docs and /openapi.json stay open (they expose only the schema).
app = FastAPI(title="SD 1.5 API", lifespan=lifespan, dependencies=[Depends(require_api_key)])


def to_png(img, info: str | None = None) -> bytes:
    meta = PngInfo()
    if info:
        meta.add_text("parameters", info)  # A1111-compatible, readable by "PNG Info" tools
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()


def submit(req: GenerateRequest) -> Job:
    job = Job(req)
    jobs[job.id] = job
    while len(jobs) > MAX_JOBS:
        oldest = next((k for k, j in jobs.items() if j.done.is_set()), None)
        if oldest is None:
            break
        del jobs[oldest]
    work.put(job)
    return job


def wait(req: GenerateRequest) -> Job:
    job = submit(req)
    job.done.wait()
    if job.status == "cancelled":
        raise HTTPException(409, "job was cancelled")
    if job.status == "failed":
        raise HTTPException(job.code, job.error)
    return job


def get_job(job_id: str) -> Job:
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    return jobs[job_id]


def job_status(job: Job) -> dict:
    queued = list(work.queue)
    return {
        "id": job.id,
        "status": job.status,
        "queue_position": queued.index(job) + 1 if job in queued else None,
        "progress": {
            "image": job.image_index + 1, "of": job.req.num_images,
            "phase": job.phase, "step": job.step, "total": job.total,
        },
        "seeds": job.seeds,
        "infos": job.infos,
        "images": [f"/jobs/{job.id}/images/{i}" for i in range(len(job.images))],
        "error": job.error,
        "elapsed_s": round((job.finished or time.time()) - job.started, 2) if job.started else None,
    }


# Sync endpoints run in FastAPI's threadpool, so waiting doesn't block the event loop.
@app.post("/generate", response_class=Response, responses={200: {"content": {"image/png": {}}}})
def generate_png(req: GenerateRequest):
    """Blocks until done; returns the first image as PNG. Seed/time in X-Seed / X-Elapsed headers."""
    job = wait(req)
    return Response(to_png(job.images[0], job.infos[0]), media_type="image/png",
                    headers={"X-Seed": str(job.seeds[0]), "X-Elapsed": f"{job.finished - job.started:.2f}"})


@app.post("/generate/json")
def generate_json(req: GenerateRequest):
    """Blocks until done; returns all images as base64 PNGs."""
    job = wait(req)
    return {
        "seeds": job.seeds,
        "infos": job.infos,
        "elapsed_s": round(job.finished - job.started, 2),
        "images": [base64.b64encode(to_png(img, info)).decode() for img, info in zip(job.images, job.infos)],
    }


@app.post("/jobs", status_code=202)
def create_job(req: GenerateRequest):
    """Queue a job and return immediately; poll GET /jobs/{id}."""
    return job_status(submit(req))


@app.get("/jobs")
def list_jobs():
    return [job_status(j) for j in reversed(jobs.values())]


@app.get("/jobs/{job_id}")
def read_job(job_id: str):
    return job_status(get_job(job_id))


@app.get("/jobs/{job_id}/images/{n}", response_class=Response, responses={200: {"content": {"image/png": {}}}})
def job_image(job_id: str, n: int):
    job = get_job(job_id)
    if not 0 <= n < len(job.images):
        raise HTTPException(404, "no such image (yet)")
    return Response(to_png(job.images[n], job.infos[n]), media_type="image/png")


@app.get("/jobs/{job_id}/preview", response_class=Response, responses={200: {"content": {"image/png": {}}}})
def job_preview(job_id: str):
    """Low-quality preview of the image currently being denoised."""
    job = get_job(job_id)
    if job.latents is None:
        raise HTTPException(404, "no preview yet")
    return Response(to_png(engine.latent_preview(job.latents)), media_type="image/png")


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    job = get_job(job_id)
    if not job.done.is_set():
        job.cancel.set()
    return job_status(job)


class PreprocessRequest(BaseModel):
    preprocessor: str
    image: str
    params: dict = {}


@app.post("/preprocess", response_class=Response, responses={200: {"content": {"image/png": {}}}})
def run_preprocessor(req: PreprocessRequest):
    """Run a ControlNet preprocessor so you can inspect/edit the control map."""
    try:
        img = eng.preprocess(req.preprocessor, req.image, req.params)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    return Response(to_png(img), media_type="image/png")


@app.get("/options")
def options():
    return eng.options()


@app.get("/health")
def health():
    return {
        "device": torch.cuda.get_device_name(0),
        "dtype": str(engine.DTYPE),
        "low_vram": engine.LOW_VRAM,
        "active_model": eng.active.name if eng.active else None,
        "queue": work.qsize(),
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / 2**20),
        "vram_reserved_mb": round(torch.cuda.memory_reserved() / 2**20),
        "vram_free_mb": round(torch.cuda.mem_get_info()[0] / 2**20),
    }
