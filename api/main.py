import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth import router as auth_router
from routes.results import router as results_router
from routes.run import router as run_router
from routes.upload import router as upload_router

app = FastAPI(title="IHSG Forecast API", version="1.0.0")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(upload_router, prefix="/upload", tags=["upload"])
app.include_router(run_router, prefix="/run", tags=["run"])
app.include_router(results_router, prefix="/results", tags=["results"])


@app.get("/health")
def health():
    return {"status": "ok"}
