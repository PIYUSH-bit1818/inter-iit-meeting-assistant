"""FastAPI entry point. Run from backend/: uvicorn app.main:app --reload"""

from fastapi import FastAPI

from . import __version__
from .config import get_settings

app = FastAPI(title="Meeting Assistant API", version=__version__)


@app.get("/health")
def health() -> dict:
    """Liveness plus a config readiness report (booleans only, no secret values)."""
    return {
        "status": "ok",
        "version": __version__,
        "config": get_settings().readiness(),
    }
