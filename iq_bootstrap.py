# iq_bootstrap.py
import json, os, iqoptionapi.constants as C

def ensure_mapping(extra=None, path="actives.json"):
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:  # << เปลี่ยนตรงนี้
                data = json.load(f) or {}
        except Exception as e:
            print(f"[ensure_mapping] cannot load {path}: {e}")  # ไม่กลืนเงียบ ๆ แล้ว
    if extra: data.update(extra)
    if data: C.ACTIVES.update(data)
    return data
