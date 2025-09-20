# core/cancel_token.py
from __future__ import annotations
import asyncio, time

class CancelToken:
    """สัญญาณหยุดแบบ async + ใช้แทน time.sleep ด้วยการรอที่ยกเลิกได้"""
    def __init__(self) -> None:
        self._evt = asyncio.Event()
        self._ts_created = time.monotonic()

    def set(self) -> None:
        self._evt.set()

    def is_set(self) -> bool:
        return self._evt.is_set()

    async def sleep(self, seconds: float) -> bool:
        """
        รอได้สูงสุด 'seconds' ถ้าถูก set ระหว่างรอ → ตัดทันที
        return True = ถูกปลุกเพราะ stop, False = ครบเวลา
        """
        try:
            await asyncio.wait_for(self._evt.wait(), timeout=max(0.0, seconds))
            return True
        except asyncio.TimeoutError:
            return False

    async def wait(self) -> None:
        await self._evt.wait()
