import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from job_store import update_job

PIPELINE_DIR = Path(os.getenv("PIPELINE_DIR", str(Path(__file__).parent.parent / "ihsg_forecast")))


def run_pipeline_background(job_id: str, skip_fetch: bool = False) -> None:
    cmd = ["python", "main.py"]
    if skip_fetch:
        cmd.append("--skip-fetch")

    update_job(job_id, status="running")
    log_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PIPELINE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            log_lines.append(line)
            update_job(job_id, log_tail=log_lines[-30:])

        proc.wait()
        if proc.returncode == 0:
            update_job(
                job_id,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                log_tail=log_lines[-30:],
            )
        else:
            update_job(
                job_id,
                status="failed",
                error=f"Process exited with code {proc.returncode}",
                log_tail=log_lines[-30:],
            )
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), log_tail=log_lines[-30:])


def launch_pipeline(job_id: str, skip_fetch: bool = False) -> None:
    t = threading.Thread(target=run_pipeline_background, args=(job_id, skip_fetch), daemon=True)
    t.start()
