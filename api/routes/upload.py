from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from auth import get_current_user
from pipeline_runner import PIPELINE_DIR
from validation import validate_file

router = APIRouter()

UPLOAD_TARGETS: dict[str, Path] = {
    "volume": PIPELINE_DIR / "data/raw/ihsg_volume.csv",
    "macro": PIPELINE_DIR / "data/macro/macro_shocks.csv",
    "ipo": PIPELINE_DIR / "data/ipo/ipo_calendar.csv",
    "scenarios": PIPELINE_DIR / "data/macro/scenarios.csv",
}


@router.post("/{file_type}")
async def upload_file(
    file_type: str,
    file: UploadFile = File(...),
    _: str = Depends(get_current_user),
):
    if file_type not in UPLOAD_TARGETS:
        raise HTTPException(status_code=400, detail=f"Unknown file type '{file_type}'. Use: {list(UPLOAD_TARGETS)}")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()

    errors = validate_file(file_type, content)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})

    dest = UPLOAD_TARGETS[file_type]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)

    return {
        "message": f"{file_type} uploaded successfully",
        "filename": file.filename,
        "size_bytes": len(content),
    }


@router.get("/status")
def upload_status(_: str = Depends(get_current_user)):
    """Returns which required files are present on disk."""
    return {
        file_type: path.exists()
        for file_type, path in UPLOAD_TARGETS.items()
    }
