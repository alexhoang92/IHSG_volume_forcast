import io
import zipfile
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from auth import get_current_user
from pipeline_runner import PIPELINE_DIR

router = APIRouter()

OUTPUT_DIR = PIPELINE_DIR / "outputs"

# Allowed category paths (prevent path traversal)
ALLOWED_CATEGORIES = {
    "charts",
    "csv",
    "csv/scenarios",
    "reports",
}


@router.get("/")
def list_results(_: str = Depends(get_current_user)):
    if not OUTPUT_DIR.exists():
        return {"charts": [], "csv": {"main": [], "scenarios": []}, "reports": []}

    def ls(path: Path, pattern: str) -> list[str]:
        if not path.exists():
            return []
        return sorted(f.name for f in path.glob(pattern) if f.is_file())

    return {
        "charts": ls(OUTPUT_DIR / "charts", "*.png"),
        "csv": {
            "main": ls(OUTPUT_DIR / "csv", "*.csv"),
            "scenarios": ls(OUTPUT_DIR / "csv/scenarios", "*.csv"),
        },
        "reports": ls(OUTPUT_DIR / "reports", "*.txt"),
        "sensitivity": ls(OUTPUT_DIR / "csv/scenarios", "*.txt"),
    }


@router.get("/chart/{filename}")
def get_chart(filename: str, _: str = Depends(get_current_user)):
    safe = Path(filename).name
    path = OUTPUT_DIR / "charts" / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Chart '{safe}' not found")
    return FileResponse(str(path), media_type="image/png")


@router.get("/download/{category:path}/{filename}")
def download_file(category: str, filename: str, _: str = Depends(get_current_user)):
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    safe = Path(filename).name
    path = OUTPUT_DIR / category / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File '{safe}' not found")
    return FileResponse(str(path), filename=safe)


@router.get("/download-all")
def download_all(_: str = Depends(get_current_user)):
    if not OUTPUT_DIR.exists():
        raise HTTPException(status_code=404, detail="No outputs available yet")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUTPUT_DIR.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(OUTPUT_DIR))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=ihsg_forecast_outputs.zip"},
    )


@router.get("/scenario-data")
def get_scenario_data(_: str = Depends(get_current_user)):
    path = OUTPUT_DIR / "csv/scenarios/forecast_summary_table.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No forecast results available. Run the pipeline first.")
    try:
        df = pd.read_csv(path, comment="#")
        df = df.dropna(how="all")
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading forecast data: {exc}")


@router.get("/backtest-data")
def get_backtest_data(_: str = Depends(get_current_user)):
    path = OUTPUT_DIR / "csv/backtest_results.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No backtest results available. Run the pipeline first.")
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading backtest data: {exc}")


@router.get("/backtest-summary")
def get_backtest_summary(_: str = Depends(get_current_user)):
    path = OUTPUT_DIR / "reports/backtest_summary.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No backtest summary available.")
    return {"text": path.read_text()}


@router.get("/sensitivity-text")
def get_sensitivity_text(_: str = Depends(get_current_user)):
    path = OUTPUT_DIR / "csv/scenarios/sensitivity_explanation.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No sensitivity explanation available.")
    return {"text": path.read_text()}


@router.get("/all-scenarios-data")
def get_all_scenarios(_: str = Depends(get_current_user)):
    path = OUTPUT_DIR / "csv/scenarios/forecast_all_scenarios.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No scenario data available.")
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
