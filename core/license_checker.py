# core/license_checker.py
from __future__ import annotations
import os, json, base64, hashlib
from datetime import datetime

try:
    import machineid
except Exception:
    machineid = None  # type: ignore

from nacl.signing import VerifyKey  # pip install pynacl
from nacl.exceptions import BadSignatureError

# -------------------------------------------------------------------
# ใส่ PUBLIC KEY (base64) ตรงนี้ได้เลย แล้ว ENV จะสามารถ override ได้
# -------------------------------------------------------------------
HARDCODED_PUBKEY_B64 = "Id1oiNxMzNMAdigAhWHiPFlKsFcBWVEGUmahknZLpvY="  # <-- วางค่า PUBLIC_BASE64 จาก Seller GUI

# เริ่มต้นด้วยค่าที่ฮาร์ดโค้ดไว้ก่อน แล้วค่อยเอา ENV มาทับ (ถ้ามี)
LICENSE_PUBKEY_B64 = os.getenv("LICENSE_PUBKEY_B64", HARDCODED_PUBKEY_B64).strip()

_verify_key = None  # lazy init เพื่อไม่ให้ล้มตอน import

def _get_verify_key() -> VerifyKey:
    """
    โหลด VerifyKey แบบ lazy + validate ความถูกต้องของ base64/ความยาว (ต้องได้ 32 ไบต์สำหรับ Ed25519)
    """
    global _verify_key, LICENSE_PUBKEY_B64
    if _verify_key is None:
        # เผื่อมีการตั้ง ENV หลังเริ่มโปรเซสไว้แล้ว
        LICENSE_PUBKEY_B64 = os.getenv("LICENSE_PUBKEY_B64", LICENSE_PUBKEY_B64).strip()
        if not LICENSE_PUBKEY_B64:
            raise RuntimeError(
                "Public key is missing. Set HARDCODED_PUBKEY_B64 in license_checker.py "
                "or provide env LICENSE_PUBKEY_B64."
            )
        try:
            raw = base64.b64decode(LICENSE_PUBKEY_B64)
        except Exception:
            raise RuntimeError("LICENSE_PUBKEY_B64 is not valid base64.")
        if len(raw) != 32:
            raise RuntimeError(f"LICENSE_PUBKEY_B64 must decode to 32 bytes (got {len(raw)}).")
        _verify_key = VerifyKey(raw)
    return _verify_key

def _current_hwid() -> str:
    if machineid:
        try:
            return machineid.id()
        except Exception:
            pass
    # fallback: hash จากชื่อเครื่อง
    return hashlib.sha256(os.environ.get("COMPUTERNAME","unknown").encode()).hexdigest()[:16]

def _parse(json_text: str) -> dict:
    data = json.loads(json_text)
    for k in ["email","hwid","expire_date","signature","issued_at","edition","seat"]:
        if k not in data:
            raise ValueError(f"missing field: {k}")
    return data

def _verify_signature(data: dict) -> None:
    """
    ข้อความที่เซ็น:
    email|hwid|expire_date|edition|seat|issued_at
    """
    msg = "|".join([
        str(data.get("email","")),
        str(data.get("hwid","")),
        str(data.get("expire_date","")),
        str(data.get("edition","standard")),
        str(data.get("seat","1")),
        str(data.get("issued_at","")),
    ]).encode("utf-8")
    sig_b = base64.b64decode(str(data.get("signature","")))
    vk = _get_verify_key()
    vk.verify(msg, sig_b)  # ถ้าผิดจะโยน BadSignatureError

def check_license(license_key_json: str) -> tuple[bool, str]:
    """
    ตรวจคีย์: (True/False, message)
    - HWID ต้องตรงเครื่อง
    - ลายเซ็น Ed25519 ต้องถูก
    - วันหมดอายุต้องยังไม่เกิน
    """
    try:
        obj = _parse(license_key_json)

        # 1) HWID binding
        if obj["hwid"] != _current_hwid():
            return False, "License is not valid for this machine (HWID mismatch)."

        # 2) Verify signature (Ed25519)
        _verify_signature(obj)

        # 3) Expiry
        expire = datetime.strptime(obj["expire_date"], "%Y-%m-%d")
        if datetime.now() > expire:
            return False, f"License expired on {obj['expire_date']}."

        email = str(obj.get("email",""))
        return True, f"License is valid for {email} until {obj['expire_date']}."
    except BadSignatureError:
        return False, "Invalid license signature."
    except Exception as e:
        return False, f"Invalid license: {e}"
