# strategies/indicator_suite.py
import numpy as np
import pandas as pd

# ---------- helpers ----------
def _to_np(x: pd.Series) -> np.ndarray:
    return pd.Series(x, dtype=float).to_numpy()

# ---------- moving averages ----------
def sma(series, length: int):
    s = pd.Series(series, dtype=float)
    return s.rolling(int(length), min_periods=int(length)).mean().to_numpy()

def ema(series, length: int):
    s = pd.Series(series, dtype=float)
    return s.ewm(span=int(length), adjust=False).mean().to_numpy()

def ma(series, length: int = 50, type: str = "EMA"):
    return sma(series, length) if str(type).upper() == "SMA" else ema(series, length)

# ---------- oscillators / trend ----------
def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    c = pd.Series(close, dtype=float)
    ema_fast = c.ewm(span=int(fast), adjust=False).mean()
    ema_slow = c.ewm(span=int(slow), adjust=False).mean()
    line = ema_fast - ema_slow
    sig  = line.ewm(span=int(signal), adjust=False).mean()
    hist = line - sig
    return line.to_numpy(), sig.to_numpy(), hist.to_numpy()

def ichimoku(high, low, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    h = pd.Series(high, dtype=float); l = pd.Series(low, dtype=float)
    conv  = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2.0
    base  = (h.rolling(kijun).max()  + l.rolling(kijun).min())  / 2.0
    spanA = ((conv + base) / 2.0).to_numpy()
    spanB = ((h.rolling(senkou_b).max() + l.rolling(senkou_b).min()) / 2.0).to_numpy()
    return conv.to_numpy(), base.to_numpy(), spanA, spanB

def rsi(close, length: int = 14):
    c = pd.Series(close, dtype=float)
    diff = c.diff()
    up = diff.clip(lower=0.0).rolling(length).mean()
    dn = (-diff.clip(upper=0.0)).rolling(length).mean().replace(0.0, 1e-9)
    rs = up / dn
    return (100.0 - (100.0 / (1.0 + rs))).to_numpy()

def stoch(high, low, close, k: int = 14, d: int = 3, smooth: int = 3):
    h = pd.Series(high, dtype=float); l = pd.Series(low, dtype=float); c = pd.Series(close, dtype=float)
    ll = l.rolling(k).min(); hh = h.rolling(k).max()
    k_raw = 100.0 * (c - ll) / (hh - ll + 1e-9)
    k_s = k_raw.rolling(int(smooth)).mean()
    d_s = k_s.rolling(int(d)).mean()
    return k_s.to_numpy(), d_s.to_numpy()

# ---------- Bollinger ----------
def bollinger(close, length: int = 20, k: float = 2.0):
    c = pd.Series(close, dtype=float)
    mid = c.rolling(length).mean()
    sd  = c.rolling(length).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    bw = (upper - lower) / (mid.replace(0.0, 1e-9)).abs()
    return mid.to_numpy(), upper.to_numpy(), lower.to_numpy(), bw.to_numpy(), sd.to_numpy()

def bbands(close, length: int = 20, k: float = 2.0):
    """Alias ที่ ai_model_5m เรียกใช้—คืนค่า 3 ชุดเหมือนชื่อคลาสสิก."""
    mid, upper, lower, _, _ = bollinger(close, length=length, k=k)
    return mid, upper, lower

def bbwidth(close, length: int = 20, k: float = 2.0):
    """ความกว้างของแถบ Bollinger"""
    _, _, _, bw, _ = bollinger(close, length=length, k=k)
    return bw

# ---------- volatility / volume ----------
def atr(high, low, close, length: int = 14):
    h = pd.Series(high, dtype=float); l = pd.Series(low, dtype=float); c = pd.Series(close, dtype=float)
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean().to_numpy()

def vwap(high, low, close, volume):
    h = pd.Series(high, dtype=float); l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float); v = pd.Series(volume, dtype=float).fillna(0.0)
    tp = (h + l + c) / 3.0
    cum_v = v.cumsum().replace(0.0, np.nan)
    return ((tp * v).cumsum() / cum_v).to_numpy()

def obv(close, volume):
    c = pd.Series(close, dtype=float); v = pd.Series(volume, dtype=float).fillna(0.0)
    sign = np.sign(c.diff().fillna(0.0))
    return (sign * v).cumsum().to_numpy()

def volume_profile(close, volume, bins: int = 24):
    c = np.asarray(close, dtype=float); v = np.asarray(volume, dtype=float)
    if len(c) < 10:
        return {"poc": float(c[-1]) if len(c) else 0.0, "levels": []}
    lo, hi = np.nanmin(c), np.nanmax(c)
    if hi <= lo:
        return {"poc": float(c[-1]), "levels": []}
    edges = np.linspace(lo, hi, int(bins) + 1)
    idx = np.clip(np.digitize(c, edges) - 1, 0, len(edges) - 2)
    volbins = np.zeros(len(edges) - 1, dtype=float)
    for i, vol in zip(idx, v):
        volbins[i] += float(vol or 0.0)
    poc_i = int(np.argmax(volbins))
    poc_price = (edges[poc_i] + edges[poc_i + 1]) / 2.0
    return {"poc": float(poc_price), "levels": volbins.tolist()}
