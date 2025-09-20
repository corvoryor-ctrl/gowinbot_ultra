# core/safe_task.py
# ตัวช่วยงาน loop ที่ "หยุดได้-กันล้ม-ล็อกซ้ำ"
from __future__ import annotations
import threading, time, logging

class SafeLoop:
    """
    ใช้กับ loop แบบ while True: ... โดยมี:
      • start()/stop() ที่ชัดเจน
      • กัน double-start
      • จับ exception แล้ว backoff
    """
    def __init__(self, name: str = "loop", target=None, backoff: float = 0.5, max_backoff: float = 4.0):
        self.name = name
        self._target = target
        self._th = None  # type: ignore
        self._stop = threading.Event()
        self._running = threading.Event()
        self._backoff = backoff
        self._max_backoff = max_backoff
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self):
        with self._lock:
            if self._th and self._th.is_alive():
                logging.info("[%s] already running", self.name)
                return False
            self._stop.clear()
            self._th = threading.Thread(target=self._run, name=self.name, daemon=True)
            self._th.start()
            return True

    def stop(self):
        with self._lock:
            self._stop.set()
        if self._th:
            self._th.join(timeout=5.0)
        self._running.clear()
        logging.info("[%s] stopped", self.name)

    def _run(self):
        self._running.set()
        backoff = self._backoff
        logging.info("[%s] started", self.name)
        while not self._stop.is_set():
            try:
                if self._target:
                    self._target()
                # ถ้าทำงานรอบหนึ่งเสร็จ ให้รีเซ็ต backoff
                backoff = self._backoff
            except Exception as e:
                logging.exception("[%s] crashed: %s", self.name, e)
                time.sleep(backoff)
                backoff = min(self._max_backoff, backoff * 2.0)
        self._running.clear()
