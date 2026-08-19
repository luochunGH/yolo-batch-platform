import csv
import gc
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
WORKER_ID = os.getenv("WORKER_ID", "worker-1")
STATUS_PATH = db.DATA_DIR / ("worker-status.json" if WORKER_ID == "worker-1" else f"worker-status-{WORKER_ID}.json")
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
    db.write_json(STATUS_PATH, payload)


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


def write_inference_result_rows(output: csv.writer, result: object, source: Path) -> None:
    detections = serialize_result(result, source)["detections"]
    if not detections:
        output.writerow([str(source), "未检测到目标", "", "", "", "", "", "", "", ""])
        return
    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        output.writerow([
            str(source),
            "已识别",
            detection["class_id"],
            detection["class_name"],
            detection["confidence"],
            x1,
            y1,
            x2,
            y2,
            "",
        ])


def readable_images(images: list[Path], work_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    valid: list[Path] = []
    skipped: list[dict[str, str]] = []
    for image in images:
        try:
            with Image.open(image) as source:
                source.verify()
            with Image.open(image) as source:
                source.load()
            valid.append(image)
        except Exception as exc:
            skipped.append({"image": str(image.relative_to(work_dir)), "reason": str(exc)[:300]})
    return valid, skipped


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
        write_worker_status("training", job_id, phase="准备训练环境")
        work_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        write_worker_status("training", job_id, phase="解压数据集")
        safe_extract(archive, work_dir)
        write_worker_status("training", job_id, phase="校验标注数据")
        data_path, image_count, class_count = prepare_training_dataset(work_dir)
        db.update_job(job_id, dataset_path=str(data_path), total=epochs)
        write_worker_status("training", job_id, phase="加载基础模型", image_count=image_count, class_count=class_count, epoch=0, epochs=epochs)
        base_model = model_path_for(job.get("model"))
        model = YOLO(str(base_model))
        write_worker_status("training", job_id, phase="检查训练环境", image_count=image_count, class_count=class_count, epoch=0, epochs=epochs)

        def on_epoch_end(trainer: object) -> None:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            current = db.get_job(job_id)
            if not current or current["status"] in {"cancelling", "cancelled"}:
                setattr(trainer, "stop", True)
                return
            db.update_job(job_id, completed=min(epoch, epochs))
            write_worker_status("training", job_id, phase=f"训练 Epoch {epoch}/{epochs}", image_count=image_count, class_count=class_count, epoch=epoch, epochs=epochs)

        model.add_callback("on_fit_epoch_end", on_epoch_end)
        model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=resolve_imgsz(job["imgsz"], data_path.parent),
            batch=int(job["train_batch"] or 16),
            workers=DATALOADER_WORKERS,
            device=DEVICE,
            amp=False,
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
        write_worker_status("training", job_id, phase="保存模型", image_count=image_count, class_count=class_count, epoch=epochs, epochs=epochs)
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
        write_worker_status("evaluating", job_id, phase="准备评估环境", image_count=image_count, class_count=class_count)
        model = YOLO(str(model_path_for_job(job)))
        write_worker_status("evaluating", job_id, phase="加载模型", image_count=image_count, class_count=class_count)
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
        images, skipped = readable_images(images, work_dir)
        total_images = len(images) + len(skipped)
        if not images:
            raise ValueError(f"all {total_images} images are unreadable")
        db.update_job(job_id, total=total_images, completed=0, failed=len(skipped))
        if skipped:
            write_worker_status("inferring", job_id, phase=f"跳过损坏图片 {len(skipped)} 张")
        model = YOLO(str(model_path_for_job(job)))
        result_path = result_dir / "inference-results.csv"
        annotated_dir = result_dir / "images"
        archive_result = result_dir / "inference-images.zip"
        batch_size = DEFAULT_BATCH_SIZE
        with result_path.open("w", encoding="utf-8-sig", newline="") as stream:
            output = csv.writer(stream)
            output.writerow(["图片路径", "状态", "类别ID", "类别名称", "置信度", "x1", "y1", "x2", "y2", "说明"])
            for skipped_image in skipped:
                output.writerow([skipped_image["image"], "已跳过", "", "", "", "", "", "", "", skipped_image["reason"]])
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
                for image, result in zip(batch, results):
                    relative_image = image.relative_to(work_dir)
                    write_inference_result_rows(output, result, relative_image)
                    target = annotated_dir / relative_image
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(result.plot()[:, :, ::-1]).save(target)
                db.update_job(job_id, completed=min(offset + len(batch), len(images)), failed=len(skipped))
                write_worker_status("inferring", job_id, phase=f"推理图片 {min(offset + len(batch), len(images))}/{len(images)}")
        with zipfile.ZipFile(archive_result, "w", zipfile.ZIP_DEFLATED) as package:
            for image in images:
                target = annotated_dir / image.relative_to(work_dir)
                package.write(target, target.relative_to(result_dir))
        db.update_job(job_id, status="completed", completed=len(images), failed=len(skipped), finished_at=db.now(), result_path=str(result_path))
        db.update_job(job_id, result_path=str(archive_result))
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


def release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def active_worker_job_ids() -> set[str]:
    job_ids: set[str] = set()
    for path in db.DATA_DIR.glob("worker-status*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("state") in {"training", "evaluating", "inferring"} and status.get("job_id"):
            job_ids.add(str(status["job_id"]))
    return job_ids


def preserve_interrupted_run(job_id: str) -> None:
    run_dir = db.DATA_DIR / "results" / job_id / "run"
    if not run_dir.is_dir():
        return
    archived_run = run_dir.with_name(f"interrupted-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.move(str(run_dir), str(archived_run))


def recover_interrupted_jobs(redis_client: redis.Redis) -> None:
    active_jobs = active_worker_job_ids()
    for job in db.list_jobs(limit=10000):
        status = str(job.get("status"))
        job_id = str(job["id"])
        if status not in {"running", "cancelling"} or job_id in active_jobs:
            continue
        if not redis_client.set(f"yolo:recovery:{job_id}", "1", nx=True, ex=30):
            continue
        if status == "cancelling":
            db.update_job(job_id, status="cancelled", finished_at=db.now(), error="Worker 重启时任务正在取消")
            continue
        preserve_interrupted_run(job_id)
        db.update_job(
            job_id,
            status="queued",
            completed=0,
            failed=0,
            started_at=None,
            finished_at=None,
            error="Worker 意外重启，已从原始 ZIP 自动重新排队",
        )
        redis_client.rpush(QUEUE_KEY, job_id)


def main() -> None:
    db.initialize()
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_worker_status("idle")
    recover_interrupted_jobs(redis_client)
    while True:
        item = redis_client.blpop(QUEUE_KEY, timeout=10)
        if item:
            try:
                run_job(item[1])
            finally:
                release_gpu_memory()
        else:
            write_worker_status("idle")


if __name__ == "__main__":
    main()
