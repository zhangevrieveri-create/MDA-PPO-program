"""表2 量价特征（纯 pandas 实现，无需 TA-Lib）：6 项交易数据 + 16 项技术指标 = 22 维。
输入单只股票按日期排序的 OHLCV 与换手率。MACD、动量不在表2中，保留计算但默认不进入模型。"""
import numpy as np
import pandas as pd


def _sma(x, n):
    return x.rolling(n, min_periods=n).mean()


def _ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def _wilder(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n):
    d = close.diff()
    up, dn = _wilder(d.clip(lower=0), n), _wilder((-d).clip(lower=0), n)
    return 100 * up / (up + dn + 1e-12)


def atr(high, low, close, n=14):
    pc = close.shift()
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)


def adx(high, low, close, n=14):
    up, dn = high.diff(), -low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(high, low, close, n) + 1e-12
    pdi, ndi = 100 * _wilder(pdm, n) / a, 100 * _wilder(ndm, n) / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-12)
    return _wilder(dx, n)


def kdj(high, low, close, n=9, m1=3, m2=3):
    lo, hi = low.rolling(n).min(), high.rolling(n).max()
    rsv = 100 * (close - lo) / (hi - lo + 1e-12)
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def cci(high, low, close, n=14):
    tp = (high + low + close) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-12)


def macd(close, fast=12, slow=26, sig=9):
    dif = _ema(close, fast) - _ema(close, slow)
    dea = _ema(dif, sig)
    return dif, dea, 2 * (dif - dea)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """返回原始指标（与表2一致，另加 MACD 与动量）。"""
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    out = pd.DataFrame(index=df.index)
    out['open'], out['high'], out['low'], out['close'], out['volume'] = o, h, l, c, v
    to = df['turnover'] if 'turnover' in df and df['turnover'].notna().any() else np.nan
    out['turnover'] = to
    for n in (5, 10, 30, 120):
        out[f'ma{n}'] = _sma(c, n)
    out['ema30'] = _ema(c, 30)
    mid, sd = _sma(c, 20), c.rolling(20).std()
    out['boll_up'], out['boll_dn'] = mid + 2 * sd, mid - 2 * sd
    out['atr'] = atr(h, l, c)
    out['adx'] = adx(h, l, c)
    out['kdj_k'], out['kdj_d'], out['kdj_j'] = kdj(h, l, c)
    for n in (6, 12, 24):
        out[f'rsi{n}'] = rsi(c, n)
    out['cci'] = cci(h, l, c)
    out['macd_dif'], out['macd_dea'], out['macd_hist'] = macd(c)
    out['mtm'] = c / c.shift(10) - 1               # 10日动量
    return out


def to_model_features(ind: pd.DataFrame) -> pd.DataFrame:
    """把原始指标转换为平稳、可跨股票比较的模型输入，列名与表2一致。"""
    c = ind['close']
    f = pd.DataFrame(index=ind.index)
    for k in ['open', 'high', 'low']:
        f[k] = ind[k] / c - 1                                  # 相对收盘价偏离
    f['close'] = np.log(c).diff()                              # 收盘价 -> 对数收益
    f['volume'] = np.log((ind['volume'] + 1) / (ind['volume'].rolling(20).mean() + 1))
    f['turnover'] = ind['turnover']
    for k in ['ma5', 'ma10', 'ma30', 'ma120', 'ema30', 'boll_up', 'boll_dn']:
        f[k] = ind[k] / c - 1
    f['atr'] = ind['atr'] / c
    f['adx'] = ind['adx'] / 100
    for k in ['kdj_k', 'kdj_d', 'kdj_j', 'rsi6', 'rsi12', 'rsi24']:
        f[k] = ind[k] / 100 - 0.5
    f['cci'] = ind['cci'] / 100
    for k in ['macd_dif', 'macd_dea', 'macd_hist']:
        f[k] = ind[k] / c
    f['mtm'] = ind['mtm']
    return f
