import threading
from datetime import datetime, timezone

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def create_job(job_id: str) -> None:
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "log_tail": [],
            "error": None,
        }


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> dict | None:
    with _lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def get_active_job() -> dict | None:
    with _lock:
        for job in _jobs.values():
            if job["status"] in ("queued", "running"):
                return dict(job)
    return None
