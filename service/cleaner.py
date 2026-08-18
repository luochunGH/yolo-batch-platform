import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from service import db


INTERVAL = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))
RESULT_RETENTION_DAYS = int(os.getenv("RESULT_RETENTION_DAYS", "30"))
FAILED_RETENTION_DAYS = int(os.getenv("FAILED_RETENTION_DAYS", "7"))
HIGH_WATERMARK = int(os.getenv("DISK_HIGH_WATERMARK", "80"))
AUTO_CLEANUP = os.getenv("AUTO_CLEANUP", "0").lower() in {"1", "true", "yes"}


def parsed(value: object) -> datetime:
    return datetime.fromisoformat(str(value)).astimezone(timezone.utc)


def remove_job_files(job_id: str) -> None:
    for path in (db.DATA_DIR / "uploads" / f"{job_id}.zip", db.DATA_DIR / "work" / job_id, db.DATA_DIR / "results" / job_id):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def cleanup() -> None:
    if not AUTO_CLEANUP:
        return
    usage = shutil.disk_usage(db.DATA_DIR)
    pressure = usage.used * 100 / usage.total >= HIGH_WATERMARK
    now = datetime.now(timezone.utc)
    for job in db.list_jobs(limit=10000):
        if job["status"] not in {"completed", "failed", "cancelled"} or job["cleaned_at"]:
            continue
        retention = FAILED_RETENTION_DAYS if job["status"] == "failed" else RESULT_RETENTION_DAYS
        finished = parsed(job["finished_at"] or job["created_at"])
        if pressure or finished < now - timedelta(days=retention):
            remove_job_files(str(job["id"]))
            db.update_job(str(job["id"]), cleaned_at=db.now(), result_path=None)


def main() -> None:
    db.initialize()
    while True:
        cleanup()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
