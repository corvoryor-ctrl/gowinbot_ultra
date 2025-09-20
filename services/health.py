# services/health.py
from __future__ import annotations
import time
from fastapi import APIRouter

router = APIRouter()

@router.get("/healthz")
def healthz():
    return {"ok": True, "ts": time.time()}

@router.get("/readiness")
def readiness():
    try:
        from core.bot_engine import bot_engine  # type: ignore
        ready = bool(getattr(bot_engine, "is_running", False))
    except Exception:
        ready = False
    return {"ready": ready}
