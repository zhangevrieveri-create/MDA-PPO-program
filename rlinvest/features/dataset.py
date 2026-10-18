"""构建强化学习数据集：22 维量价特征 ⊕ 3 维舆情特征 -> 状态张量 S ∈ R^{D×U×F}（论文 F=25）。

舆情特征（论文第二部分（二）第四步）：
  CNS_t  = λ·CNS_{t−1} + (1−λ)·NS_t                        式(3)
  CDIV_t = sqrt(λ·CDIV²_{t−1} + (1−λ)·DIV_t²)               式(4)
  当日没有新闻时 NS_t = DIV_t = 0，即沿用上一日状态并按 λ 衰减。
  米筐个股新闻情绪滞后 1 个交易日使用。

npz 内容：
  X        (D,U,F) 用训练集统计量标准化后的特征；未上市日期为 0
  close    (D,U)   前复权收盘价（停牌日沿用前收盘，未上市日期为 NaN）
  listed   (D,U)   当日已上市（未上市股票用掩码排除，不参与选股）
  buy_ok   (D,U)   当日收盘可买入（已上市、未停牌且未涨停）
  sell_ok  (D,U)   当日收盘可卖出（未停牌且未跌停）
  bench    (D,)    基准指数收盘价
  dates, codes, features, val_idx, test_idx, mu, sd
  论文只划分训练集与测试集：dataset.val_date 为 null 时 val_idx = test_idx（没有验证集）。
"""
import numpy as np
import pandas as pd

from ..data import rq_api
from ..utils import get_logger
from .indicators import compute_indicators, to_model_features

log = get_logger()
WARMUP = 130          # MA120 等长窗口指标的预热期
LIMIT_TOL = 1e-3      # 收盘价距涨跌停价的相对容差


def select_universe(cfg, price, news_daily, trading_dates):
    u, d = cfg['dataset']['universe'], cfg['dataset']
    codes_all = sorted(price['order_book_id'].unique())
    if u['method'] == 'all':                      # 论文：股票池全部股票
        return codes_all
    if u['method'] == 'codes':
        if len(u['codes']) == 0:
            raise ValueError('dataset.universe.method=codes 时需填写 codes 列表')
        return list(u['codes'])
    # top_news：只用测试期之前的新闻数排序，避免用到样本外信息
    td = trading_dates[(trading_dates >= d['start_date']) & (trading_dates <= d['end_date'])]
    p = price[(price['date'] >= d['start_date']) & (price['date'] <= d['end_date'])]
    cover = p[p['volume'] > 0].groupby('order_book_id')['date'].nunique() / len(td)
    ok = cover[cover >= u['min_coverage']].index
    nd = news_daily[news_daily['order_book_id'].isin(ok)]
    nd = nd[nd['date'] < pd.Timestamp(d.get('val_date') or d['test_date'])]
    cnt = nd.groupby('order_book_id')['news_cnt'].sum()
    if len(cnt) < u['n']:
        raise ValueError(f'满足覆盖率要求且有新闻的股票只有 {len(cnt)} 只，少于 n={u["n"]}')
    return list(cnt.sort_values(ascending=False).head(u['n']).index)


def cumulate(ns: np.ndarray, div: np.ndarray, lam: float):
    """式(3)(4)：按交易日递推累积净情绪与累积情绪分歧度。"""
    cns, cdiv = np.zeros(len(ns)), np.zeros(len(ns))
    c, q = 0.0, 0.0
    for i in range(len(ns)):
        c = lam * c + (1 - lam) * ns[i]
        q = np.sqrt(lam * q * q + (1 - lam) * div[i] ** 2)
        cns[i], cdiv[i] = c, q
    return cns, cdiv


def _stock_panel(code, price, news_daily, rq_senti, dates, cfg):
    p = price[price['order_book_id'] == code].set_index('date').sort_index().reindex(dates)
    has_row = p['close'].notna()
    listed = has_row.cummax()                                  # 首个有行情的交易日起视为已上市
    traded = has_row & (p['volume'].fillna(0) > 0)
    p[['open', 'high', 'low', 'close']] = p[['open', 'high', 'low', 'close']].ffill()
    for k in ['open', 'high', 'low']:                          # 停牌日 OHLC 取前收盘
        p[k] = p[k].where(traded, p['close'])
    p['volume'] = p['volume'].fillna(0)
    p['turnover'] = p['turnover'].where(traded, 0.0) if 'turnover' in p else 0.0

    f = to_model_features(compute_indicators(p))
    nd = news_daily[news_daily['order_book_id'] == code].set_index('date').reindex(dates)
    lam = cfg['sentiment']['decay']
    f['cns'], f['cdiv'] = cumulate(nd['ns'].fillna(0).to_numpy(), nd['div'].fillna(0).to_numpy(), lam)
    f['news_cnt'] = np.log1p(nd['news_cnt'].fillna(0))
    if rq_senti is not None:
        rs = rq_senti[rq_senti['order_book_id'] == code].set_index('date')['rq_senti']
        lag = cfg['dataset'].get('rq_senti_lag', 0)
        f['rq_senti'] = rs.reindex(dates).ffill().shift(lag).fillna(0).clip(-1, 1)
    else:
        f['rq_senti'] = 0.0
    f.loc[~listed.to_numpy()] = np.nan                         # 未上市日期置为缺失，标准化后填 0

    up = p['limit_up'] if 'limit_up' in p else np.inf
    dn = p['limit_down'] if 'limit_down' in p else 0.0
    buy_ok = traded & ~(p['close'] >= up * (1 - LIMIT_TOL))
    sell_ok = traded & ~(p['close'] <= dn * (1 + LIMIT_TOL))
    close = p['close'].where(listed)
    return f, close, listed, buy_ok, sell_ok


def build_dataset(cfg):
    P, d = cfg['paths'], cfg['dataset']
    td = rq_api.load_trading_dates(P['raw'])
    news_daily = pd.read_parquet(P['interim'] / 'news_daily.parquet')
    price = rq_api.load_price(P['raw'])
    codes = select_universe(cfg, price, news_daily, td)
    price = price[price['order_book_id'].isin(codes)]
    rq_senti = rq_api.load_rq_sentiment(P['raw'], codes)
    if rq_senti is None:
        log.warning('未找到米筐情绪指数，rq_senti 特征置 0')
    log.info(f'投资域 U={len(codes)}')

    dates = td[(td >= pd.Timestamp(d['start_date'])) & (td <= pd.Timestamp(d['end_date']))]
    names = list(d['technical']) + list(d['sentiment'])
    D, U, F = len(dates), len(codes), len(names)
    X = np.full((D, U, F), np.nan, np.float32)
    C = np.full((D, U), np.nan)
    LST, BUY, SELL = (np.zeros((D, U), bool) for _ in range(3))
    for j, code in enumerate(codes):
        f, c, l, b, s = _stock_panel(code, price, news_daily, rq_senti, dates, cfg)
        missing = set(names) - set(f.columns)
        if missing:
            raise KeyError(f'配置中的特征不存在：{missing}')
        X[:, j] = f[names].to_numpy(np.float32)
        C[:, j], LST[:, j], BUY[:, j], SELL[:, j] = c.to_numpy(), l.to_numpy(), b.to_numpy(), s.to_numpy()
        if (j + 1) % 500 == 0:
            log.info(f'已处理 {j + 1}/{U} 只股票')

    first = WARMUP                                              # 论文样本起点 + 指标预热期；未上市股票用掩码
    X, C, LST, BUY, SELL, dates = X[first:], C[first:], LST[first:], BUY[first:], SELL[first:], dates[first:]
    BUY &= LST
    SELL &= LST
    X[~np.isfinite(X)] = np.nan

    bench = rq_api.load_benchmark(P['raw']).reindex(dates).ffill().bfill().values
    test_idx = int(dates.searchsorted(pd.Timestamp(d['test_date'])))
    val_idx = int(dates.searchsorted(pd.Timestamp(d['val_date']))) if d.get('val_date') else test_idx
    if not (cfg['rl']['T'] < val_idx <= test_idx < len(dates)):
        raise ValueError(f'数据划分不合法：val_idx={val_idx}, test_idx={test_idx}, D={len(dates)}')

    # 标准化只用训练集（已上市样本）的统计量，避免未来信息
    mu, sd = np.nanmean(X[:val_idx], axis=(0, 1)), np.nanstd(X[:val_idx], axis=(0, 1))
    const = ~(sd > 1e-8)
    if const.any():
        log.warning(f'以下特征在训练集上为常数，标准化后为 0：{[n for n, k in zip(names, const) if k]}')
    X = (X - mu) / np.where(const, 1.0, sd)
    X = np.clip(np.nan_to_num(X, nan=0.0), -5, 5).astype(np.float32)

    out = P['processed'] / f'{d["name"]}.npz'
    np.savez_compressed(out, X=X, close=C, listed=LST, buy_ok=BUY, sell_ok=SELL, bench=bench,
                        dates=np.array(dates.strftime('%Y-%m-%d'), dtype=str),
                        codes=np.array(codes, dtype=str), features=np.array(names, dtype=str),
                        val_idx=val_idx, test_idx=test_idx, mu=mu, sd=sd)
    log.info(f'数据集 {out.name}: X{X.shape}，训练 {val_idx} 天 / 验证 {test_idx - val_idx} 天 / '
             f'测试 {len(dates) - test_idx} 天，F={F}；测试集起点 {dates[test_idx]:%Y-%m-%d}')
    log.info(f'已上市比例 {LST.mean():.2%}；停牌或涨停不可买入比例 {1 - BUY[LST].mean():.2%}，'
             f'停牌或跌停不可卖出比例 {1 - SELL[LST].mean():.2%}')
    return out


def load_dataset(path):
    z = np.load(path, allow_pickle=False)
    close = z['close']
    listed = z['listed'] if 'listed' in z.files else np.isfinite(close)
    return dict(X=z['X'], close=close, listed=listed, buy_ok=z['buy_ok'], sell_ok=z['sell_ok'],
                bench=z['bench'], dates=pd.to_datetime(z['dates']), codes=[str(c) for c in z['codes']],
                features=[str(f) for f in z['features']],
                val_idx=int(z['val_idx']), test_idx=int(z['test_idx']))
