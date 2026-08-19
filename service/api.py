import csv
import hmac
import json
import os
import re
import secrets
import shutil
import uuid
import zipfile
from pathlib import Path

import redis
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from service import db


API_KEY = os.getenv("API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MAX_UPLOAD_GB = int(os.getenv("MAX_UPLOAD_GB", "20"))
MAX_ARCHIVE_FILES = int(os.getenv("MAX_ARCHIVE_FILES", "100000"))
MAX_EXTRACT_GB = int(os.getenv("MAX_EXTRACT_GB", "100"))
MAX_ARCHIVE_FILE_GB = int(os.getenv("MAX_ARCHIVE_FILE_GB", "10"))
MIN_DISK_FREE_GB = int(os.getenv("MIN_DISK_FREE_GB", "10"))
MAX_COMPRESSION_RATIO = int(os.getenv("MAX_COMPRESSION_RATIO", "100"))
TRAIN_QUEUE_KEY = "yolo:jobs:train"
NONTRAIN_QUEUE_KEY = "yolo:jobs:nontrain"
DOWNLOAD_TOKEN_PREFIX = "yolo:download:"
DOWNLOAD_TOKEN_TTL = int(os.getenv("DOWNLOAD_TOKEN_TTL_SECONDS", "300"))
MODEL_NAMES = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
NAME_SUFFIX_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI(title="Docker YOLO Web Console", version="1.0.0")
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


def unique_job_name(requested: str, fallback: str) -> str:
    base = requested.strip() or fallback
    if not db.name_exists(base):
        return base
    for _ in range(20):
        candidate = f"{base}-{''.join(secrets.choice(NAME_SUFFIX_ALPHABET) for _ in range(6))}"
        if not db.name_exists(candidate):
            return candidate
    raise HTTPException(status_code=500, detail="could not allocate a unique task name")


def validate_archive_names(package: zipfile.ZipFile) -> None:
    entries = package.infolist()
    if len(entries) > MAX_ARCHIVE_FILES:
        raise ValueError("archive contains too many files")
    total_size = sum(item.file_size for item in entries)
    if total_size > MAX_EXTRACT_GB * 1024**3:
        raise ValueError("archive expands beyond the allowed size")
    if any(item.file_size > MAX_ARCHIVE_FILE_GB * 1024**3 for item in entries):
        raise ValueError("archive contains an oversized file")
    for item in entries:
        if item.file_size and item.compress_size and item.file_size / item.compress_size > MAX_COMPRESSION_RATIO:
            raise ValueError("archive compression ratio is too high")
    if shutil.disk_usage(db.DATA_DIR).free < total_size + MIN_DISK_FREE_GB * 1024**3:
        raise ValueError("insufficient disk space to extract archive safely")
    for item in entries:
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


def legacy_inference_csv(archive_path: Path) -> Path:
    csv_path = archive_path.with_name("inference-results.csv")
    if csv_path.is_file():
        return csv_path
    with zipfile.ZipFile(archive_path) as package, csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        output = csv.writer(stream)
        output.writerow(["图片路径", "状态", "类别ID", "类别名称", "置信度", "x1", "y1", "x2", "y2", "说明"])
        with package.open("results.jsonl") as source:
            for line in source:
                result = json.loads(line)
                detections = result.get("detections", [])
                if not detections:
                    output.writerow([result.get("image", ""), "未检测到目标", "", "", "", "", "", "", "", ""])
                    continue
                for detection in detections:
                    x1, y1, x2, y2 = detection["box_xyxy"]
                    output.writerow([result.get("image", ""), "已识别", detection["class_id"], detection["class_name"], detection["confidence"], x1, y1, x2, y2, ""])
        summary = json.loads(package.read("summary.json"))
        for skipped in summary.get("skipped", []):
            output.writerow([skipped.get("image", ""), "已跳过", "", "", "", "", "", "", "", skipped.get("reason", "")])
    return csv_path


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


def queue_key_for(task_type: str) -> str:
    return TRAIN_QUEUE_KEY if task_type == "train" else NONTRAIN_QUEUE_KEY


def enqueue_job(job_id: str, task_type: str) -> None:
    try:
        queue.rpush(queue_key_for(task_type), job_id)
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail="Redis queue is unavailable; task was not created") from exc


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/dashboard", dependencies=[Depends(require_api_key)])
def get_dashboard() -> dict[str, object]:
    usage = shutil.disk_usage(db.DATA_DIR)
    status_paths = sorted(db.DATA_DIR.glob("worker-status*.json"))
    worker_statuses = []
    for path in status_paths:
        try:
            worker_statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    worker_status = dict(next((item for item in worker_statuses if item.get("state") not in {"idle", None}), worker_statuses[0] if worker_statuses else {}))
    worker_status["workers"] = worker_statuses
    return {
        **db.dashboard(),
        "disk": {"total": usage.total, "used": usage.used, "free": usage.free},
        "models": available_models(),
        "trained_models": trained_models(),
        "worker": worker_status,
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
        "name": unique_job_name(name, Path(archive.filename).stem),
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
    try:
        enqueue_job(job_id, task_type)
    except HTTPException:
        db.delete_job(job_id)
        archive_path.unlink(missing_ok=True)
        raise
    return job_payload({**job, "total": epochs if task_type == "train" else image_count, "completed": 0, "failed": 0})


@app.post("/api/v1/jobs/{source_job_id}/retry", dependencies=[Depends(require_api_key)])
def retry_job(
    source_job_id: str,
    name: str = Form(default=""),
    model: str = Form(default=""),
    imgsz: str = Form(default=""),
    epochs: int | None = Form(default=None),
    train_batch: int | None = Form(default=None),
) -> dict[str, object]:
    source = db.get_job(source_job_id)
    if not source:
        raise HTTPException(status_code=404, detail="source job not found")
    if source.get("task_type") != "train":
        raise HTTPException(status_code=422, detail="only training jobs can be retrained")
    source_archive = Path(str(source.get("uploaded_path") or ""))
    if not source_archive.is_file():
        raise HTTPException(status_code=404, detail="source ZIP is no longer available")
    selected_model = model.strip() or str(source.get("model") or "yolo11n.pt")
    if selected_model not in available_models():
        raise HTTPException(status_code=422, detail="selected pretrained model is not installed")
    normalized_imgsz = normalize_imgsz(imgsz.strip() or str(source.get("imgsz") or "640"))
    selected_epochs = int(epochs if epochs is not None else source.get("epochs") or 50)
    selected_batch = int(train_batch if train_batch is not None else source.get("train_batch") or 16)
    if not 1 <= selected_epochs <= 1000 or not 1 <= selected_batch <= 128:
        raise HTTPException(status_code=422, detail="invalid training parameters")

    job_id = uuid.uuid4().hex
    archive_path = db.DATA_DIR / "uploads" / f"{job_id}.zip"
    shutil.copy2(source_archive, archive_path)
    job = {
        "id": job_id,
        "name": unique_job_name(name, f"{source['name']}-重新训练"),
        "status": "queued",
        "model": selected_model,
        "model_id": None,
        "imgsz": normalized_imgsz,
        "confidence": float(source.get("confidence") or 0.25),
        "task_type": "train",
        "epochs": selected_epochs,
        "train_batch": selected_batch,
        "created_at": db.now(),
        "uploaded_path": str(archive_path),
        "dataset_path": None,
        "artifact_path": None,
    }
    db.create_job(job)
    try:
        enqueue_job(job_id, "train")
    except HTTPException:
        db.delete_job(job_id)
        archive_path.unlink(missing_ok=True)
        raise
    return job_payload({**job, "total": selected_epochs, "completed": 0, "failed": 0, "source_job_id": source_job_id})


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


@app.post("/api/v1/jobs/{job_id}/download-token", dependencies=[Depends(require_api_key)])
def create_download_token(job_id: str, format: str | None = Query(default=None)) -> dict[str, str | int]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    token = secrets.token_urlsafe(32)
    try:
        queue.setex(f"{DOWNLOAD_TOKEN_PREFIX}{token}", DOWNLOAD_TOKEN_TTL, json.dumps({"job_id": job_id, "format": format or ""}))
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail="download token service is unavailable") from exc
    return {"token": token, "expires_in": DOWNLOAD_TOKEN_TTL}


def consume_download_token(token: str | None, job_id: str, format: str | None) -> None:
    if not token:
        raise HTTPException(status_code=401, detail="download token is required")
    try:
        raw = queue.getdel(f"{DOWNLOAD_TOKEN_PREFIX}{token}")
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail="download token service is unavailable") from exc
    if not raw:
        raise HTTPException(status_code=401, detail="download token is invalid or expired")
    payload = json.loads(raw)
    if payload.get("job_id") != job_id or payload.get("format") != (format or ""):
        raise HTTPException(status_code=401, detail="download token does not match this file")


@app.get("/api/v1/jobs/{job_id}/download")
def download_result(job_id: str, format: str | None = Query(default=None), token: str | None = Query(default=None)) -> FileResponse:
    consume_download_token(token, job_id, format)
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    path_value = (job.get("model_package_path") or job.get("artifact_path")) if job.get("task_type") == "train" else (job.get("artifact_path") or job.get("result_path"))
    if not path_value:
        raise HTTPException(status_code=404, detail="result not found")
    path = Path(str(path_value))
    if not path.exists():
        raise HTTPException(status_code=404, detail="result has been cleaned")
    if job.get("task_type") == "train":
        filename, media_type = f"{job_id}-models.zip", "application/zip"
    elif job.get("task_type") == "evaluate":
        filename, media_type = f"{job_id}-evaluation.json", "application/json"
    else:
        if format == "csv":
            if path.suffix.lower() == ".zip":
                csv_path = path.with_name("inference-results.csv")
                path = csv_path if csv_path.is_file() else legacy_inference_csv(path)
            filename, media_type = f"{job_id}-inference.csv", "text/csv; charset=utf-8"
        else:
            if path.suffix.lower() != ".zip":
                zip_path = path.with_name("inference-images.zip")
                if not zip_path.is_file():
                    raise HTTPException(status_code=404, detail="annotated image ZIP not found")
                path = zip_path
            filename, media_type = f"{job_id}-inference-images.zip", "application/zip"
    return FileResponse(path, filename=filename, media_type=media_type)
