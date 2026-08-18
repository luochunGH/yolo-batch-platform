import hmac
import json
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path

import redis
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from service import db


API_KEY = os.getenv("API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MAX_UPLOAD_GB = int(os.getenv("MAX_UPLOAD_GB", "20"))
QUEUE_KEY = "yolo:jobs"
MODEL_NAMES = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI(title="YOLO Batch Platform", version="1.0.0")
queue = redis.from_url(REDIS_URL, decode_responses=True)


def available_models() -> list[str]:
    model_dir = db.DATA_DIR / "models"
    return [name for name in MODEL_NAMES if (model_dir / name).is_file()]


def trained_models() -> list[dict[str, object]]:
    models = []
    for job in db.list_jobs():
        artifact = Path(str(job.get("artifact_path") or ""))
        if job.get("task_type") == "train" and job.get("status") == "completed" and artifact.is_file():
            models.append({
                "id": job["id"],
                "name": job["name"],
                "filename": artifact.name,
                "created_at": job["created_at"],
                "size": artifact.stat().st_size,
            })
    return models


def selected_trained_model(model_id: str | None) -> dict[str, object] | None:
    if not model_id:
        return None
    for model in trained_models():
        if model["id"] == model_id:
            return model
    return None


@app.on_event("startup")
def startup() -> None:
    db.initialize()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY or not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid API key")


def job_payload(job: dict[str, object]) -> dict[str, object]:
    total = int(job["total"])
    completed = int(job["completed"])
    failed = int(job["failed"])
    job["progress"] = round((completed + failed) * 100 / total, 2) if total else 0
    return job


def validate_archive_names(package: zipfile.ZipFile) -> None:
    for item in package.infolist():
        if Path(item.filename).is_absolute() or ".." in Path(item.filename).parts:
            raise ValueError("archive contains an unsafe path")


def validate_training_archive(package: zipfile.ZipFile) -> tuple[int, int]:
    validate_archive_names(package)
    names = {Path(item.filename).name.lower() for item in package.infolist()}
    if not ({"data.yaml", "data.yml"} & names):
        raise ValueError("training archive must contain data.yaml or data.yml")
    images = [item for item in package.infolist() if Path(item.filename).suffix.lower() in IMAGE_SUFFIXES]
    labels = [item for item in package.infolist() if Path(item.filename).suffix.lower() == ".txt"]
    if not images:
        raise ValueError("training archive contains no supported images")
    if not labels:
        raise ValueError("training archive contains no YOLO label txt files")
    return len(images), len(labels)


def normalize_imgsz(value: str) -> str:
    size = value.strip().lower().replace("×", "x")
    if size == "original":
        return size
    if size.isdigit():
        numeric = int(size)
        if 320 <= numeric <= 4096:
            return str(numeric)
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if match:
        width, height = (int(part) for part in match.groups())
        if 320 <= width <= 4096 and 320 <= height <= 4096:
            return f"{width}x{height}"
    raise HTTPException(status_code=422, detail="imgsz must be original, 320-4096, or WIDTHxHEIGHT")


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/dashboard", dependencies=[Depends(require_api_key)])
def get_dashboard() -> dict[str, object]:
    usage = shutil.disk_usage(db.DATA_DIR)
    status_path = db.DATA_DIR / "worker-status.json"
    worker_status = status_path.read_text(encoding="utf-8") if status_path.exists() else "{}"
    return {
        **db.dashboard(),
        "disk": {"total": usage.total, "used": usage.used, "free": usage.free},
        "models": available_models(),
        "trained_models": trained_models(),
        "worker": json.loads(worker_status),
    }


@app.get("/api/v1/jobs", dependencies=[Depends(require_api_key)])
def get_jobs() -> list[dict[str, object]]:
    return [job_payload(job) for job in db.list_jobs()]


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict[str, object]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job_payload(job)


@app.post("/api/v1/jobs", dependencies=[Depends(require_api_key)])
async def create_job(
    archive: UploadFile = File(...),
    name: str = Form(default=""),
    task_type: str = Form(default="train"),
    model: str = Form(default="yolo11n.pt"),
    model_id: str | None = Form(default=None),
    imgsz: str = Form(default="640"),
    confidence: float = Form(default=0.25),
    epochs: int = Form(default=50),
    train_batch: int = Form(default=16),
) -> dict[str, object]:
    if not archive.filename or not archive.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=415, detail="upload a ZIP archive containing images")
    if task_type not in {"train", "evaluate", "inference"}:
        raise HTTPException(status_code=422, detail="invalid task type")
    try:
        normalized_imgsz = normalize_imgsz(imgsz)
    except AttributeError as exc:
        raise HTTPException(status_code=422, detail="invalid imgsz") from exc
    if not 0.01 <= confidence <= 0.99:
        raise HTTPException(status_code=422, detail="invalid confidence")
    if task_type == "train" and (model not in available_models() or model_id):
        raise HTTPException(status_code=422, detail="selected pretrained model is not installed")
    if task_type != "train" and not selected_trained_model(model_id):
        raise HTTPException(status_code=422, detail="select a completed trained model")
    if task_type == "train" and (not 1 <= epochs <= 1000 or not 1 <= train_batch <= 128):
        raise HTTPException(status_code=422, detail="invalid training parameters")

    job_id = uuid.uuid4().hex
    archive_path = db.DATA_DIR / "uploads" / f"{job_id}.zip"
    size = 0
    with archive_path.open("wb") as target:
        while chunk := await archive.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_GB * 1024**3:
                target.close()
                archive_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="archive exceeds upload limit")
            target.write(chunk)
    try:
        with zipfile.ZipFile(archive_path) as package:
            if task_type in {"train", "evaluate"}:
                image_count, _ = validate_training_archive(package)
            else:
                validate_archive_names(package)
                image_count = sum(Path(item.filename).suffix.lower() in IMAGE_SUFFIXES for item in package.infolist())
                if not image_count:
                    raise ValueError("archive contains no supported images")
    except (zipfile.BadZipFile, ValueError) as exc:
        archive_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = {
        "id": job_id,
        "name": name.strip() or Path(archive.filename).stem,
        "status": "queued",
        "model": model,
        "model_id": model_id,
        "imgsz": normalized_imgsz,
        "confidence": confidence,
        "task_type": task_type,
        "epochs": epochs if task_type == "train" else 0,
        "train_batch": train_batch if task_type == "train" else 0,
        "created_at": db.now(),
        "uploaded_path": str(archive_path),
        "dataset_path": None,
        "artifact_path": None,
    }
    db.create_job(job)
    queue.rpush(QUEUE_KEY, job_id)
    return job_payload({**job, "total": epochs if task_type == "train" else image_count, "completed": 0, "failed": 0})


@app.post("/api/v1/jobs/{job_id}/cancel", dependencies=[Depends(require_api_key)])
def cancel_job(job_id: str) -> dict[str, object]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] == "queued":
        db.update_job(job_id, status="cancelled", finished_at=db.now())
    elif job["status"] == "running":
        db.update_job(job_id, status="cancelling")
    else:
        raise HTTPException(status_code=409, detail="job is not cancellable")
    return job_payload(db.get_job(job_id) or job)


@app.delete("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def remove_job(job_id: str) -> dict[str, str]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] in {"queued", "running", "cancelling"}:
        raise HTTPException(status_code=409, detail="cancel the job before deletion")
    for path in (db.DATA_DIR / "uploads" / f"{job_id}.zip", db.DATA_DIR / "work" / job_id, db.DATA_DIR / "results" / job_id, db.DATA_DIR / "models" / job_id):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    db.delete_job(job_id)
    return {"status": "deleted"}


@app.get("/api/v1/jobs/{job_id}/download", dependencies=[Depends(require_api_key)])
def download_result(job_id: str) -> FileResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    path_value = job.get("artifact_path") or job.get("result_path")
    if not path_value:
        raise HTTPException(status_code=404, detail="result not found")
    path = Path(str(path_value))
    if not path.exists():
        raise HTTPException(status_code=404, detail="result has been cleaned")
    if job.get("task_type") == "train":
        filename, media_type = f"{job_id}-best.pt", "application/octet-stream"
    elif job.get("task_type") == "evaluate":
        filename, media_type = f"{job_id}-evaluation.json", "application/json"
    else:
        filename, media_type = f"{job_id}-inference.zip", "application/zip"
    return FileResponse(path, filename=filename, media_type=media_type)
