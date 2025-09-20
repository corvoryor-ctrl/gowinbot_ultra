# strategies/base_strategy.py
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, List
import pandas as pd

class BaseStrategy(ABC):
    """
    คลาสแม่สำหรับกลยุทธ์ทั้งหมด (Abstract Base Class)
    - คง interface เดิม (check_signal) และ helper เดิม (_get_sens01, _meta, get_signal_meta)
    - เพิ่ม default methods/fields ที่เอนจิ้นคาดหวัง เพื่อลดโอกาสพังและเขียนซ้ำ
    """

    def __init__(self, settings: dict | None = None):
        self.name: str = self.__class__.__name__ if self.__class__.__name__ != "BaseStrategy" else "Base Strategy"
        self.settings: Dict[str, Any] = settings or {}
        self.last_meta: Optional[Dict[str, Any]] = None  # ให้ทุกกลยุทธ์เข้าถึง meta ล่าสุดได้

        # ---- Common knobs (อ่านจาก settings ถ้ามี) ----
        self.tf: int = int(self.settings.get("tf", self.settings.get("duration", 1) or 1))
        self.required_bars: int = int(self.settings.get("required_bars", 205))
        self.multi_asset: bool = bool(self.settings.get("multi_asset", False))

        # micro-5s (ให้เอนจิ้นอ่านไปใช้ filter ได้ ถ้ากลยุทธ์เปิด)
        self.micro_5s_enable: bool = bool(self.settings.get("micro_5s_enable", False))
        self.micro_5s_window_sec: int = int(self.settings.get("micro_5s_window_sec", 15))

        # thresholds (ตั้ง None เพื่อคงพฤติกรรมเดิม: เอนจิ้นจะไม่บังคับใช้หาก None)
        self.conf_thr: Optional[float] = self._get_opt_float("conf_thr")
        self.gap_thr: Optional[float] = self._get_opt_float("gap_thr")

        # cache สำหรับความน่าจะเป็นขาขึ้น ที่เอนจิ้นอาจอ่าน (ถ้ามี)
        self.p_up: Optional[float] = None

    # ---------- Sensitivity helpers ----------
    def _get_sens01(self, default_100: int = 50) -> float:
        """
        คืนค่า sensitivity ในสเกล 0..1 แบบ 'ยอมรับ 0 จริง ๆ'
        ถ้าไม่มี key เลยค่อย fallback เป็น default_100
        """
        val = self.settings.get("sensitivity")
        return float(val) / 100.0 if val is not None else float(default_100) / 100.0

    def _clamp01(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _get_opt_float(self, key: str) -> Optional[float]:
        v = self.settings.get(key)
        try:
            return None if v is None else float(v)
        except Exception:
            return None

    # ---------- Meta helpers ----------
    def _meta(self, **kw):
        """เก็บ meta + log ถ้าเปิด verbose (ค่า default เปิดไว้)"""
        self.last_meta = kw
        verbose = self.settings.get("verbose", True)
        if verbose:
            try:
                logging.info(f"[{self.name}] META: {kw}")
            except Exception:
                pass

    def get_signal_meta(self):
        """BotEngine จะดึงไป log ต่อ/โชว์ใน UI ได้"""
        return self.last_meta

    # ---------- Core abstract ----------
    @abstractmethod
    def check_signal(self, candles: pd.DataFrame) -> str:
        """
        :return: 'call', 'put', หรือ 'none'
        """
        pass

    # ---------- Compatibility shims (ที่เอนจิ้นเรียกใช้แบบยืดหยุ่น) ----------
    def get_signal(self, candles: pd.DataFrame) -> str:
        """ดีฟอลต์: ใช้ check_signal()"""
        try:
            out = self.check_signal(candles)
            return str(out).lower() if isinstance(out, str) else "none"
        except Exception as e:
            logging.error(f"{self.name}.get_signal error: {e}")
            return "none"

    def get_signal_asset(self, asset: str, candles: pd.DataFrame) -> str:
        """บางกลยุทธ์อยากรู้ชื่อ asset; ดีฟอลต์ไม่ใช้"""
        return self.get_signal(candles)

    def decide(self, asset: str, candles: pd.DataFrame):
        return self.get_signal_asset(asset, candles)

    def evaluate(self, asset: str, candles: pd.DataFrame):
        return self.decide(asset, candles)

    def infer(self, asset: str, candles: pd.DataFrame):
        return self.evaluate(asset, candles)

    def predict(self, asset: str, candles: pd.DataFrame):
        return self.infer(asset, candles)

    def predict_proba(self, asset: str, candles: pd.DataFrame):
        """
        ดีฟอลต์: พยายามอนุมานจากทิศทาง + sensitivity เล็กน้อย
        - คงพฤติกรรมปลอดภัย: ถ้าไม่ชัด ให้ p_up ≈ 0.5 (ไม่ bias)
        """
        try:
            dir_ = self.get_signal_asset(asset, candles)
            s = self._get_sens01(50)  # 0..1
            # ไม่ bias ค่าดีฟอลต์ เพื่อคงพฤติกรรมเดิม (เอนจิ้นจะไม่บังคับ threshold ถ้า conf/gap ไม่มี)
            base_p = 0.5 + (s - 0.5) * 0.0
            if dir_ == "call":
                self.p_up = base_p
                return {"signal": "call", "p_up": base_p}
            elif dir_ == "put":
                self.p_up = 1.0 - base_p
                return {"signal": "put", "p_up": 1.0 - base_p}
        except Exception as e:
            logging.debug(f"{self.name}.predict_proba fallback error: {e}")

        self.p_up = None
        return {"signal": "none", "p_up": None}

    # ---------- Multi-asset scaffolding ----------
    def check_signals_pair(self, df_map: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        คืน {asset: 'call'|'put'} จากการประเมินรายตัว (ดีฟอลต์ไม่มี ranking)
        กลยุทธ์ลูกสามารถ override เพื่อทำ pair/arb/relative-strength ได้
        """
        out: Dict[str, str] = {}
        try:
            for a, df in (df_map or {}).items():
                sig = self.get_signal(df)
                if sig in ("call", "put"):
                    out[a] = sig
            return out
        except Exception as e:
            logging.debug(f"{self.name}.check_signals_pair error: {e}")
            return {}

    def check_signals_rank(self, df_map: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        คืนลิสต์เรียงลำดับ [{'asset': a, 'signal': 'call'|'put', 'score': float}]
        ดีฟอลต์ให้ score=0.0 (เพื่อความเข้ากัน) — กลยุทธ์ลูกควร override
        """
        picks: List[Dict[str, Any]] = []
        try:
            for a, df in (df_map or {}).items():
                sig = self.get_signal(df)
                if sig in ("call", "put"):
                    picks.append({"asset": a, "signal": sig, "score": 0.0})
            return picks
        except Exception as e:
            logging.debug(f"{self.name}.check_signals_rank error: {e}")
            return []

    # ---------- Settings updater ----------
    def set_params(self, **kwargs):
        """
        อัปเดต settings แล้ว sync ฟิลด์ที่เกี่ยวข้อง—สะดวกเวลาแก้ค่าจาก UI
        """
        self.settings.update(kwargs)
        self.tf = int(self.settings.get("tf", self.settings.get("duration", self.tf)))
        self.required_bars = int(self.settings.get("required_bars", self.required_bars))
        self.multi_asset = bool(self.settings.get("multi_asset", self.multi_asset))
        self.micro_5s_enable = bool(self.settings.get("micro_5s_enable", self.micro_5s_enable))
        self.micro_5s_window_sec = int(self.settings.get("micro_5s_window_sec", self.micro_5s_window_sec))
        self.conf_thr = self._get_opt_float("conf_thr")
        self.gap_thr = self._get_opt_float("gap_thr")
        return self
