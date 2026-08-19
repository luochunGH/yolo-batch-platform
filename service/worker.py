import json
import os
import os.path
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import redis
import yaml
from PIL import Image
from ultralytics import YOLO

from service import db


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_KEY = "yolo:jobs"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/data/models/yolo11n.pt"))
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")
DEVICE = os.getenv("DEVICE", "0")
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
DEFAULT_CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", "2"))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def model_path_for(name: object) -> Path:
    requested = str(name or MODEL_NAME)
    candidate = MODEL_PATH.parent / requested
    return candidate if candidate.exists() else MODEL_PATH


def model_path_for_job(job: dict[str, object]) -> Path:
    model_id = job.get("model_id")
    if model_id:
        trained = db.get_job(str(model_id))
        artifact = Path(str((trained or {}).get("artifact_path") or ""))
        if trained and trained.get("task_type") == "train" and trained.get("status") == "completed" and artifact.is_file():
            return artifact
        raise FileNotFoundError("selected trained model is unavailable")
    path = model_path_for(job.get("model"))
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path.name}")
    return path


def safe_extract(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as package:
        for item in package.infolist():
            target = (destination / item.filename).resolve()
            try:
                if os.path.commonpath((str(target), str(destination.resolve()))) != str(destination.resolve()):
                    raise ValueError("unsafe archive path")
            except ValueError:
                raise ValueError("unsafe archive path")
        package.extractall(destination)


def gpu_status() -> dict[str, object]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip().split(",")

        def number(value: str, decimal: bool = False) -> int | float | None:
            value = value.strip()
            if value in {"", "[N/A]", "N/A"}:
                return None
            return round(float(value), 1) if decimal else int(float(value))

        return {
            "gpu_name": output[0].strip(),
            "gpu_utilization": number(output[1]),
            "gpu_memory_used": number(output[2]),
            "gpu_memory_total": number(output[3]),
            "gpu_temperature": number(output[4], decimal=True),
            "gpu_power": number(output[5], decimal=True),
        }
    except Exception:
        return {"gpu_utilization": None, "gpu_memory_used": None, "gpu_memory_total": None}


def write_worker_status(state: str, job_id: str | None = None, **extra: object) -> None:
    payload = {"state": state, "job_id": job_id, "updated_at": db.now(), **gpu_status(), **extra}
    db.write_json(db.DATA_DIR / "worker-status.json", payload)


def resolve_imgsz(value: object, dataset_root: Path | None = None) -> int | tuple[int, int]:
    size = str(value or DEFAULT_IMGSZ).strip().lower().replace("×", "x")
    if size == "original":
        if dataset_root is not None:
            image = next((path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES), None)
            if image is not None:
                with Image.open(image) as source:
                    width, height = source.size
                return (height, width)
        return DEFAULT_IMGSZ
    if size.isdigit():
        return int(size)
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if match:
        width, height = (int(part) for part in match.groups())
        return (height, width)
    raise ValueError(f"invalid imgsz: {value}")


def serialize_result(result: object, source: Path) -> dict[str, object]:
    names = result.names
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": round(float(box.conf.item()), 6),
                "box_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
            }
        )
    return {"image": str(source), "detections": detections}


def resolve_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if os.path.commonpath((str(resolved), str(root.resolve()))) != str(root.resolve()):
        raise ValueError("dataset path escapes archive")
    return resolved


def label_path_for(image: Path, dataset_root: Path) -> Path:
    relative = image.relative_to(dataset_root)
    parts = list(relative.parts)
    if "images" in parts:
        index = parts.index("images")
        return dataset_root.joinpath(*parts[:index], "labels", *parts[index + 1:]).with_suffix(".txt")
    return image.with_suffix(".txt")


def validate_label_file(label_path: Path, class_count: int) -> None:
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"invalid label format: {label_path.name}:{line_number}")
        class_id = int(values[0])
        coordinates = [float(value) for value in values[1:]]
        if not 0 <= class_id < class_count or any(value < 0 or value > 1 for value in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
            raise ValueError(f"invalid label values: {label_path.name}:{line_number}")


def prepare_training_dataset(work_dir: Path) -> tuple[Path, int, int]:
    yaml_files = sorted(list(work_dir.rglob("data.yaml")) + list(work_dir.rglob("data.yml")))
    if not yaml_files:
        raise ValueError("training archive must contain data.yaml or data.yml")
    config_path = yaml_files[0]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda value: int(value))]
    if not isinstance(names, list) or not names:
        raise ValueError("data.yaml must define a non-empty names list")
    config_root = config_path.parent
    configured_root = Path(str(config.get("path", ".")))
    if configured_root.is_absolute():
        raise ValueError("data.yaml path must be relative")
    dataset_root = resolve_inside(config_root / configured_root, work_dir)

    def dataset_split(value: object) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("data.yaml must define train and val paths")
        split = Path(value)
        if split.is_absolute():
            raise ValueError("train and val paths must be relative")
        return resolve_inside(dataset_root / split, work_dir)

    train_dir = dataset_split(config.get("train"))
    val_dir = dataset_split(config.get("val"))
    train_images = [path for path in train_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    val_images = [path for path in val_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not train_images:
        raise ValueError("training split contains no supported images")
    if not val_images:
        raise ValueError("validation split contains no supported images")
    image_paths = train_images + val_images
    for image_path in image_paths:
        label_path = label_path_for(image_path, dataset_root)
        if not label_path.exists():
            raise ValueError(f"label file is missing for {image_path.name}")
        validate_label_file(label_path, len(names))

    normalized = dict(config)
    normalized["path"] = str(dataset_root)
    normalized["train"] = str(train_dir.relative_to(dataset_root))
    normalized["val"] = str(val_dir.relative_to(dataset_root))
    normalized["names"] = names
    normalized_path = work_dir / "normalized-data.yaml"
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return normalized_path, len(image_paths), len(names)


def run_training(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job or job["status"] == "cancelled":
        return
    archive = Path(str(job["uploaded_path"]))
    work_dir = db.DATA_DIR / "work" / job_id
    result_dir = db.DATA_DIR / "results" / job_id
    model_dir = db.DATA_DIR / "models" / job_id
    try:
        epochs = int(job["epochs"] or 50)
        db.update_job(job_id, status="running", started_at=db.now(), completed=0, total=epochs, error=None)
        work_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_extract(archive, work_dir)
        data_path, image_count, class_count = prepare_training_dataset(work_dir)
        db.update_job(job_id, dataset_path=str(data_path), total=epochs)
        write_worker_status("training", job_id, image_count=image_count, class_count=class_count, epoch=0, epochs=epochs)
        base_model = model_path_for(job.get("model"))
        model = YOLO(str(base_model))

        def on_epoch_end(trainer: object) -> None:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            current = db.get_job(job_id)
            if not current or current["status"] in {"cancelling", "cancelled"}:
                setattr(trainer, "stop", True)
                return
            db.update_job(job_id, completed=min(epoch, epochs))
            write_worker_status("training", job_id, image_count=image_count, class_count=class_count, epoch=epoch, epochs=epochs)

        model.add_callback("on_fit_epoch_end", on_epoch_end)
        model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=resolve_imgsz(job["imgsz"], data_path.parent),
            batch=int(job["train_batch"] or 16),
            workers=DATALOADER_WORKERS,
            device=DEVICE,
            project=str(result_dir),
            name="run",
            exist_ok=True,
            plots=True,
            verbose=False,
        )
        current = db.get_job(job_id)
        if not current or current["status"] in {"cancelling", "cancelled"}:
            db.update_job(job_id, status="cancelled", finished_at=db.now())
            return
        run_dir = result_dir / "run"
        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"
        if not best_path.exists():
            raise FileNotFoundError("training completed without best.pt")
        artifact_path = model_dir / "best.pt"
        shutil.copy2(best_path, artifact_path)
        if last_path.exists():
            shutil.copy2(last_path, model_dir / "last.pt")
        db.update_job(job_id, status="completed", completed=epochs, finished_at=db.now(), result_path=str(run_dir), artifact_path=str(artifact_path))
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as exc:
        db.update_job(job_id, status="failed", error=str(exc)[:1000], finished_at=db.now())
    finally:
        write_worker_status("idle")


def run_evaluation(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job or job["status"] == "cancelled":
        return
    archive = Path(str(job["uploaded_path"]))
    work_dir = db.DATA_DIR / "work" / job_id
    result_dir = db.DATA_DIR / "results" / job_id
    try:
        db.update_job(job_id, status="running", started_at=db.now(), error=None)
        work_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        safe_extract(archive, work_dir)
        data_path, image_count, class_count = prepare_training_dataset(work_dir)
        db.update_job(job_id, dataset_path=str(data_path), total=image_count, completed=0)
        write_worker_status("evaluating", job_id, image_count=image_count, class_count=class_count)
        model = YOLO(str(model_path_for_job(job)))
        metrics = model.val(
            data=str(data_path),
            imgsz=resolve_imgsz(job["imgsz"], data_path.parent),
            device=DEVICE,
            project=str(result_dir),
            name="run",
            exist_ok=True,
            plots=True,
            verbose=False,
        )
        box = getattr(metrics, "box", None)

        def metric(name: str) -> float | None:
            value = getattr(box, name, None)
            return round(float(value), 6) if value is not None else None

        report = {
            "job_id": job_id,
            "model_id": job.get("model_id"),
            "images": image_count,
            "classes": class_count,
            "precision": metric("mp"),
            "recall": metric("mr"),
            "map50": metric("map50"),
            "map50_95": metric("map"),
            "completed_at": db.now(),
        }
        report_path = result_dir / "evaluation.json"
        db.write_json(report_path, report)
        db.update_job(job_id, status="completed", completed=image_count, finished_at=db.now(), result_path=str(report_path))
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as exc:
        db.update_job(job_id, status="failed", error=str(exc)[:1000], finished_at=db.now())
    finally:
        write_worker_status("idle")


def run_inference(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job or job["status"] == "cancelled":
        return
    archive = Path(str(job["uploaded_path"]))
    work_dir = db.DATA_DIR / "work" / job_id
    result_dir = db.DATA_DIR / "results" / job_id
    try:
        db.update_job(job_id, status="running", started_at=db.now(), error=None)
        work_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        safe_extract(archive, work_dir)
        images = sorted(path for path in work_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError("archive contains no supported images")
        db.update_job(job_id, total=len(images), completed=0, failed=0)
        model = YOLO(str(model_path_for_job(job)))
        result_path = result_dir / "results.jsonl"
        annotated_dir = result_dir / "images"
        batch_size = DEFAULT_BATCH_SIZE
        for offset in range(0, len(images), batch_size):
            current = db.get_job(job_id)
            if not current or current["status"] in {"cancelling", "cancelled"}:
                db.update_job(job_id, status="cancelled", finished_at=db.now())
                return
            batch = images[offset : offset + batch_size]
            results = model.predict(
                source=[str(image) for image in batch],
                imgsz=resolve_imgsz(job["imgsz"]),
                conf=float(job["confidence"] or DEFAULT_CONFIDENCE),
                device=DEVICE,
                half=True,
                verbose=False,
            )
            with result_path.open("a", encoding="utf-8") as output:
                for image, result in zip(batch, results):
                    output.write(json.dumps(serialize_result(result, image.relative_to(work_dir)), ensure_ascii=False) + "\n")
                    target = annotated_dir / image.relative_to(work_dir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(result.plot()[:, :, ::-1]).save(target)
            db.update_job(job_id, completed=min(offset + len(batch), len(images)))
            write_worker_status("inferring", job_id)
        summary_path = result_dir / "summary.json"
        db.write_json(summary_path, {"job_id": job_id, "total": len(images), "completed_at": db.now()})
        archive_result = result_dir / "inference-results.zip"
        with zipfile.ZipFile(archive_result, "w", zipfile.ZIP_DEFLATED) as package:
            for path in (annotated_dir / image.relative_to(work_dir) for image in images):
                package.write(path, path.relative_to(result_dir))
            package.write(result_path, result_path.name)
            package.write(summary_path, summary_path.name)
        db.update_job(job_id, status="completed", completed=len(images), finished_at=db.now(), result_path=str(archive_result))
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as exc:
        db.update_job(job_id, status="failed", error=str(exc)[:1000], finished_at=db.now())
    finally:
        write_worker_status("idle")


def run_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job or job["status"] == "cancelled":
        return
    if job.get("task_type") == "train":
        run_training(job_id)
    elif job.get("task_type") == "evaluate":
        run_evaluation(job_id)
    else:
        run_inference(job_id)


def main() -> None:
    db.initialize()
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    for job in db.list_jobs(limit=10000):
        if job["status"] in {"running", "cancelling"}:
            db.update_job(job["id"], status="queued", started_at=None, error="Worker 重启后自动重新排队")
            redis_client.rpush(QUEUE_KEY, job["id"])
    write_worker_status("idle")
    while True:
        item = redis_client.blpop(QUEUE_KEY, timeout=10)
        if item:
            run_job(item[1])
        else:
            write_worker_status("idle")


if __name__ == "__main__":
    main()
