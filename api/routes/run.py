import uuid

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user
from job_store import create_job, get_active_job, get_job
from pipeline_runner import launch_pipeline

router = APIRouter()


@router.post("/")
def trigger_run(
    skip_fetch: bool = False,
    _: str = Depends(get_current_user),
):
    active = get_active_job()
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"A pipeline run is already {active['status']}. Wait for it to finish.",
        )

    job_id = str(uuid.uuid4())
    create_job(job_id)
    launch_pipeline(job_id, skip_fetch=skip_fetch)
    return {"job_id": job_id, "status": "queued"}


@router.get("/status/{job_id}")
def get_status(job_id: str, _: str = Depends(get_current_user)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/latest")
def get_latest(_: str = Depends(get_current_user)):
    """Returns the most recently created job (useful for page refresh)."""
    from job_store import _jobs
    if not _jobs:
        return None
    latest = max(_jobs.values(), key=lambda j: j["created_at"])
    return latest
