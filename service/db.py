import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "app.db"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_layout() -> None:
    for name in ("uploads", "work", "results", "models", "logs"):
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    ensure_layout()
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    try:
        yield db
        db.commit()
    finally:
        db.close()


def initialize() -> None:
    with connection() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              status TEXT NOT NULL,
              model TEXT NOT NULL,
              imgsz INTEGER NOT NULL,
              confidence REAL NOT NULL,
              total INTEGER NOT NULL DEFAULT 0,
              completed INTEGER NOT NULL DEFAULT 0,
              failed INTEGER NOT NULL DEFAULT 0,
              error TEXT,
              created_at TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT,
              cleaned_at TEXT,
              result_path TEXT,
              uploaded_path TEXT NOT NULL
            )
            """
        )
        migrations = {
            "task_type": "TEXT NOT NULL DEFAULT 'inference'",
            "epochs": "INTEGER NOT NULL DEFAULT 0",
            "train_batch": "INTEGER NOT NULL DEFAULT 0",
            "dataset_path": "TEXT",
            "artifact_path": "TEXT",
            "model_id": "TEXT",
        }
        existing = {row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
        for column, definition in migrations.items():
            if column not in existing:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")


def create_job(job: dict[str, object]) -> None:
    with connection() as db:
        db.execute(
            """
            INSERT INTO jobs (id,name,status,model,model_id,imgsz,confidence,created_at,uploaded_path,
                              task_type,epochs,train_batch,dataset_path,artifact_path)
            VALUES (:id,:name,:status,:model,:model_id,:imgsz,:confidence,:created_at,:uploaded_path,
                    :task_type,:epochs,:train_batch,:dataset_path,:artifact_path)
            """,
            job,
        )


def get_job(job_id: str) -> dict[str, object] | None:
    with connection() as db:
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 100) -> list[dict[str, object]]:
    with connection() as db:
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def update_job(job_id: str, **values: object) -> None:
    if not values:
        return
    columns = ", ".join(f"{key} = ?" for key in values)
    with connection() as db:
        db.execute(f"UPDATE jobs SET {columns} WHERE id = ?", (*values.values(), job_id))


def delete_job(job_id: str) -> None:
    with connection() as db:
        db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def dashboard() -> dict[str, object]:
    with connection() as db:
        rows = db.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
        total_images = db.execute("SELECT COALESCE(SUM(total), 0) FROM jobs").fetchone()[0]
        completed_images = db.execute("SELECT COALESCE(SUM(completed), 0) FROM jobs").fetchone()[0]
    counts = {row["status"]: row["count"] for row in rows}
    return {"counts": counts, "total_images": total_images, "completed_images": completed_images}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
