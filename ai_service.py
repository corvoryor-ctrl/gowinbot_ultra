# backend/ai_service.py
from fastapi import APIRouter, Body, Query
import logging

# ---------- resilient imports (ไม่ทำให้บริการทั้งตัวล้มถ้าโมดูลหนึ่งหาย) ----------
# ai_meta: ควบคุมสถานะ/รีเทรน/คาลิเบรต
try:
    import ai_meta  # ปกติอยู่ใน backend.ai_meta หรือ sys.path เดิม
except Exception as e:  # pragma: no cover
    ai_meta = None
    _AI_META_IMPORT_ERR = e

# trade_logger: ดึง metrics จากฐาน SQLite (เราแพตช์ไว้ให้ robust แล้ว)
try:
    from database import trade_logger
except Exception:  # pragma: no cover - เผื่อโครงสร้างแพ็กเกจต่างสภาพแวดล้อม
    try:
        from backend.database import trade_logger  # แบบ absolute อีกทาง
    except Exception:
        trade_logger = None

router = APIRouter()

# ---------- helpers ----------
def _sanitize_tf(x: int | str | None, default: int = 1) -> int:
    """
    รับค่ามาจาก payload/query แล้วบีบให้เหลือ 1 หรือ 5 (อนาคตถ้าใช้ 2-4 ก็จะตกที่ 1)
    """
    try:
        v = int(x if x is not None else default)
    except Exception:
        v = int(default)
    return 5 if v == 5 else 1

def _tf_from(payload: dict | None, default: int = 1) -> int:
    return _sanitize_tf((payload or {}).get("tf"), default)

def _metrics_from_db(tf: int):
    """
    พยายามอ่านเมตริกจาก DB (ถ้าใช้ได้); ถ้าไม่ได้จะคืน (None, None, None)
    """
    if trade_logger is None:
        logging.warning("[API] trade_logger not importable; metrics unavailable")
        return None, None, None
    try:
        acc, ece, mean_conf = trade_logger.compute_metrics(tf, last_n=200)
        logging.info(f"[API] metrics from DB tf={tf} acc={acc} ece={ece} mean_conf={mean_conf}")
        return acc, ece, mean_conf
    except Exception as e:  # ไม่ให้พัง
        logging.warning(f"[API] trade_logger metrics failed tf={tf}: {e}")
        return None, None, None

def _require_aimeta():
    if ai_meta is None:
        err = str(globals().get("_AI_META_IMPORT_ERR", "ai_meta not available"))
        return {"ok": False, "error": f"ai_meta module not importable: {err}"}
    return None

# ---------- STATUS ----------
@router.get("/ai/status")
def ai_status(tf: int = Query(1, ge=1, le=5)):
    tf = _sanitize_tf(tf, 1)
    logging.info(f"[API] GET /ai/status tf={tf}")
    if (err := _require_aimeta()) is not None:
        return err
    return ai_meta.get_status(tf)

# ---------- AUTO (เปิด/ปิด auto-retrain) ----------
@router.post("/ai/auto")
def ai_auto(payload: dict | None = Body(default=None)):
    if (err := _require_aimeta()) is not None:
        return err
    tf      = _tf_from(payload, 1)
    enabled = bool((payload or {}).get("enabled", True))
    every   = int((payload or {}).get("every_orders", 40) or 40)
    # กันค่า every ต่ำเกินไป (เผื่อผู้ใช้ผิดพลาด)
    if every < 10:
        logging.info(f"[API] bump every_orders from {every} -> 10")
        every = 10
    logging.info(f"[API] POST /ai/auto tf={tf} enabled={enabled} every={every}")
    return ai_meta.set_auto(tf, enabled, every)

# ---------- RETRAIN ----------
def _do_retrain(tf: int):
    logging.info(f"[API] ▶ retrain request tf={tf}")
    if (err := _require_aimeta()) is not None:
        return err
    acc, ece, _ = _metrics_from_db(tf)
    # ให้ ai_meta ตัดสินใจต่อ (รองรับ None ๆ)
    return ai_meta.retrain_and_calibrate(tf, accuracy=acc, ece=ece)

@router.post("/ai/retrain")
def ai_retrain_post(payload: dict | None = Body(default=None)):
    tf = _tf_from(payload, 1)
    logging.info(f"[API] POST /ai/retrain tf={tf}, payload={payload}")
    return _do_retrain(tf)

@router.get("/ai/retrain")
def ai_retrain_get(tf: int = Query(1, ge=1, le=5)):
    tf = _sanitize_tf(tf, 1)
    logging.info(f"[API] GET /ai/retrain tf={tf}")
    return _do_retrain(tf)

# ---------- CALIBRATE ----------
def _do_calibrate(tf: int):
    logging.info(f"[API] ▶ calibrate request tf={tf}")
    if (err := _require_aimeta()) is not None:
        return err
    acc, ece, _ = _metrics_from_db(tf)
    return ai_meta.retrain_and_calibrate(tf, accuracy=acc, ece=ece)

@router.post("/ai/calibrate")
def ai_calibrate_post(payload: dict | None = Body(default=None)):
    tf = _tf_from(payload, 1)
    logging.info(f"[API] POST /ai/calibrate tf={tf}, payload={payload}")
    return _do_calibrate(tf)

@router.get("/ai/calibrate")
def ai_calibrate_get(tf: int = Query(1, ge=1, le=5)):
    tf = _sanitize_tf(tf, 1)
    logging.info(f"[API] GET /ai/calibrate tf={tf}")
    return _do_calibrate(tf)
