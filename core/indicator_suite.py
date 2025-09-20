# strategies/indicator_suite.py
# คืนค่า pandas.Series เสมอ (เพื่อความเข้ากันได้กับโค้ดเดิมที่เรียก .to_numpy())
# พร้อมตั้ง index เป็นช่วง -N..-1 เพื่อรองรับการอ้าง s[-1] โดยไม่พัง
import numpy as np
import pandas as pd

# ---------- helpers ----------
def _as_series(x) -> pd.Series:
    return pd.Series(x, dtype=float).astype(float)

def _clamp_len(n: int) -> int:
    try:
        return max(1, int(n))
    except Exception:
        return 1

def _neg_index(s: pd.Series) -> pd.Series:
    # ตั้ง index ให้เป็นช่วง -N..-1 เพื่อให้สอดคล้องโค้ดที่อ้างด้วย s[-1]
    try:
        n = len(s)
        s.index = pd.RangeIndex(start=-n, stop=0, step=1)
    except Exception:
        pass
    return s

# ---------- moving averages ----------
def sma(series, length: int):
    n = _clamp_len(length)
    s = _as_series(series).rolling(n, min_periods=n).mean()
    return _neg_index(s)

def ema(series, length: int):
    n = _clamp_len(length)
    s = _as_series(series).ewm(span=n, adjust=False).mean()
    return _neg_index(s)

def ma(series, length: int = 50, type: str = "EMA"):
    return sma(series, length) if str(type).upper() == "SMA" else ema(series, length)

# ---------- oscillators / trend ----------
def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    f = _clamp_len(fast); s = _clamp_len(slow); g = _clamp_len(signal)
    c = _as_series(close)
    ema_fast = c.ewm(span=f, adjust=False).mean()
    ema_slow = c.ewm(span=s, adjust=False).mean()
    line = ema_fast - ema_slow
    sig  = line.ewm(span=g, adjust=False).mean()
    hist = line - sig
    return _neg_index(line), _neg_index(sig), _neg_index(hist)  # ทั้งสามเป็น Series

def ichimoku(high, low, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    t = _clamp_len(tenkan); k = _clamp_len(kijun); b = _clamp_len(senkou_b)
    h = _as_series(high); l = _as_series(low)
    conv  = (h.rolling(t).max() + l.rolling(t).min()) / 2.0
    base  = (h.rolling(k).max() + l.rolling(k).min()) / 2.0
    spanA = (conv + base) / 2.0
    spanB = (h.rolling(b).max() + l.rolling(b).min()) / 2.0
    return _neg_index(conv), _neg_index(base), _neg_index(spanA), _neg_index(spanB)

def rsi(close, length: int = 14):
    # Wilder RSI (EMA-style) เสถียรกว่า rolling mean
    n = _clamp_len(length)
    c = _as_series(close)
    diff = c.diff()
    up = diff.clip(lower=0.0)
    dn = (-diff.clip(upper=0.0))
    roll_up = up.ewm(alpha=1.0/n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1.0/n, adjust=False).mean().replace(0.0, 1e-9)
    rs = roll_up / roll_dn
    r = 100.0 - (100.0 / (1.0 + rs))
    r = r.clip(lower=0.0, upper=100.0)
    return _neg_index(r)

def stoch(high, low, close, k: int = 14, d: int = 3, smooth: int = 3):
    kk = _clamp_len(k); dd = _clamp_len(d); sm = _clamp_len(smooth)
    h = _as_series(high); l = _as_series(low); c = _as_series(close)
    ll = l.rolling(kk).min()
    hh = h.rolling(kk).max()
    k_raw = 100.0 * (c - ll) / (hh - ll + 1e-9)
    k_s = k_raw.rolling(sm, min_periods=sm).mean().clip(lower=0.0, upper=100.0)
    d_s = k_s.rolling(dd, min_periods=dd).mean().clip(lower=0.0, upper=100.0)
    return _neg_index(k_s), _neg_index(d_s)

# ---------- Bollinger ----------
def bollinger(close, length: int = 20, k: float = 2.0):
    n = _clamp_len(length)
    c = _as_series(close)
    mid = c.rolling(n, min_periods=n).mean()
    sd  = c.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + float(k) * sd
    lower = mid - float(k) * sd
    bw = (upper - lower) / mid.replace(0.0, 1e-9).abs()
    return _neg_index(mid), _neg_index(upper), _neg_index(lower), _neg_index(bw), _neg_index(sd)

def bbands(close, length: int = 20, k: float = 2.0):
    mid, upper, lower, _, _ = bollinger(close, length=length, k=k)
    return mid, upper, lower  # สามชุดเป็น Series

def bbwidth(close, length: int = 20, k: float = 2.0):
    _, _, _, bw, _ = bollinger(close, length=length, k=k)
    return bw

# ---------- volatility / volume ----------
def atr(high, low, close, length: int = 14):
    # Wilder ATR (EMA-style)
    n = _clamp_len(length)
    h = _as_series(high); l = _as_series(low); c = _as_series(close)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1.0/n, adjust=False).mean()
    return _neg_index(a)

def vwap(high, low, close, volume):
    h = _as_series(high); l = _as_series(low)
    c = _as_series(close); v = _as_series(volume).fillna(0.0)
    tp = (h + l + c) / 3.0
    cum_v = v.cumsum().replace(0.0, np.nan)
    out = (tp * v).cumsum() / cum_v
    return _neg_index(out)

def obv(close, volume):
    c = _as_series(close); v = _as_series(volume).fillna(0.0)
    sign = np.sign(c.diff().fillna(0.0))
    out = (sign * v).cumsum()
    return _neg_index(out)

def volume_profile(close, volume, bins: int = 24):
    c = _as_series(close).to_numpy()
    v = _as_series(volume).to_numpy()
    if c.size < 10:
        return {"poc": float(c[-1]) if c.size else 0.0, "levels": []}
    lo, hi = (np.nanmin(c), np.nanmax(c))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {"poc": float(c[-1]) if c.size else 0.0, "levels": []}
    edges = np.linspace(lo, hi, _clamp_len(bins) + 1)
    idx = np.clip(np.digitize(c, edges) - 1, 0, len(edges) - 2)
    volbins = np.zeros(len(edges) - 1, dtype=float)
    for i, vol in zip(idx, v):
        volbins[int(i)] += float(vol or 0.0)
    poc_i = int(np.argmax(volbins))
    poc_price = (edges[poc_i] + edges[poc_i + 1]) / 2.0
    return {"poc": float(poc_price), "levels": volbins.tolist()}
