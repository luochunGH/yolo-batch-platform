import os
import shutil
import time

from service import db


INTERVAL = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))
def remove_job_files(job_id: str) -> None:
    for path in (db.DATA_DIR / "uploads" / f"{job_id}.zip", db.DATA_DIR / "work" / job_id, db.DATA_DIR / "results" / job_id):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def cleanup() -> None:
    # Data is retained until the user explicitly deletes the task.
    return


def main() -> None:
    db.initialize()
    while True:
        cleanup()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
