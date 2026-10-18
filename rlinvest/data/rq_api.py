"""米筐 RQData 数据接口：股票池、交易日历、日线量价（含涨跌停价）、换手率、基准指数、新闻舆情、个股情绪指数。

结果以长表 (order_book_id, date, ...) 保存为 parquet，按块落盘，可断点续传。
"""
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..utils import get_logger

log = get_logger()
PRICE_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'total_turnover', 'limit_up', 'limit_down']


def init_rq(cfg):
    import rqdatac as rq
    user = cfg['rqdata'].get('username') or os.getenv('RQDATA_USERNAME')
    pwd = cfg['rqdata'].get('password') or os.getenv('RQDATA_PASSWORD')
    if user and pwd:
        rq.init(user, pwd)
    else:
        rq.init()
    return rq


def _to_long(df, value_cols=None):
    """把 rqdatac 返回的 (order_book_id, date) 多级索引表转为长表。"""
    df = df.reset_index()
    df = df.rename(columns={'tradedate': 'date', 'datetime': 'date', 'level_1': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    if value_cols:
        df = df[['order_book_id', 'date'] + value_cols]
    return df


def _chunks(codes, n):
    for i in range(0, len(codes), n):
        yield i // n, codes[i:i + n]


def _date_range(cfg):
    return cfg['rqdata']['start_date'], cfg['rqdata']['end_date']


# ============================ 下载 ============================
def fetch_instruments(rq, cfg, raw_dir: Path):
    """股票池：A 股普通股。论文口径：剔除北交所、样本期内曾被 ST、样本期内退市、截至样本期末上市不满一年的股票。
    这一筛选为保证数据连续，使用了样本期末的信息（论文已在四（一）说明）。配置了 rqdata.codes 时只取这些股票。"""
    u = cfg['rqdata']['universe']
    start, _ = _date_range(cfg)
    ins = rq.all_instruments(type='CS', market='cn')
    ins = ins[['order_book_id', 'symbol', 'listed_date', 'de_listed_date', 'exchange']].copy()
    codes = cfg['rqdata'].get('codes')
    if codes:
        ins = ins[ins['order_book_id'].isin(codes)].sort_values('order_book_id').reset_index(drop=True)
        missing = set(codes) - set(ins['order_book_id'])
        if missing:
            log.warning(f'以下代码在米筐股票列表中不存在：{sorted(missing)}')
        ins.to_csv(raw_dir / 'instruments.csv', index=False, encoding='utf-8-sig')
        log.info(f'股票池（指定代码）{len(ins)} 只：{dict(zip(ins["order_book_id"], ins["symbol"]))}')
        return ins
    if u.get('exclude_bj'):
        ins = ins[ins['order_book_id'].str.endswith(('.XSHG', '.XSHE'))]
    ins = ins[ins['listed_date'] <= u['listed_before']]
    _, end = _date_range(cfg)
    if u.get('exclude_delisted', True):
        # 论文：剔除样本期内退市的股票（只保留截至样本期末仍在上市的股票）
        alive = (ins['de_listed_date'] == '0000-00-00') | (ins['de_listed_date'] > end)
    else:
        alive = (ins['de_listed_date'] == '0000-00-00') | (ins['de_listed_date'] > start)
    ins = ins[alive]
    st = u.get('exclude_st')
    if st == 'ever':
        # 论文：剔除样本期内曾被实施 ST（含 *ST）的股票，按逐日 ST 状态判断
        ever = []
        for _, part in _chunks(list(ins['order_book_id']), cfg['rqdata']['chunk_size']):
            flag = rq.is_st_stock(part, start, end)
            if flag is not None and len(flag):
                ever += list(flag.columns[flag.any(axis=0)])
        ins = ins[~ins['order_book_id'].isin(ever)]
        log.info(f'样本期内曾被实施 ST 的股票 {len(set(ever))} 只，已剔除')
    elif st:
        ins = ins[~ins['symbol'].str.contains('ST')]
    ins = ins.sort_values('order_book_id').reset_index(drop=True)
    if u.get('max_stocks'):
        ins = ins.head(u['max_stocks'])
    ins.to_csv(raw_dir / 'instruments.csv', index=False, encoding='utf-8-sig')
    log.info(f'股票池 {len(ins)} 只 -> instruments.csv')
    return ins


def fetch_trading_dates(rq, cfg, raw_dir: Path):
    start, end = _date_range(cfg)
    # 多取一个月，用于把样本末尾收盘后的新闻对齐到下一交易日
    end = (pd.Timestamp(end) + pd.Timedelta(days=31)).strftime('%Y-%m-%d')
    dates = pd.to_datetime(rq.get_trading_dates(start, end))
    pd.Series(dates, name='date').to_csv(raw_dir / 'trading_dates.csv', index=False)
    log.info(f'交易日 {len(dates)} 天 -> trading_dates.csv')
    return dates


def fetch_price(rq, cfg, codes, raw_dir: Path):
    """前复权日线量价 + 涨跌停价 + 换手率，按块保存到 raw/price/part_xxxx.parquet。"""
    out = raw_dir / 'price'
    out.mkdir(exist_ok=True)
    s, e = _date_range(cfg)
    for i, part in tqdm(list(_chunks(list(codes), cfg['rqdata']['chunk_size'])), desc='量价数据'):
        f = out / f'part_{i:04d}.parquet'
        if f.exists():
            continue
        px = rq.get_price(part, start_date=s, end_date=e, frequency='1d', fields=PRICE_FIELDS,
                          adjust_type='pre', expect_df=True)
        if px is None or px.empty:
            continue
        px = _to_long(px, PRICE_FIELDS)
        try:
            to = rq.get_turnover_rate(part, start_date=s, end_date=e, fields='today', expect_df=True)
            to = _to_long(to).rename(columns={'today': 'turnover'})[['order_book_id', 'date', 'turnover']]
            px = px.merge(to, on=['order_book_id', 'date'], how='left')
        except Exception as ex:  # 部分账户无换手率权限
            log.warning(f'换手率获取失败（该列置空）：{ex}')
            px['turnover'] = float('nan')
        px.to_parquet(f, index=False)


def fetch_benchmark(rq, cfg, raw_dir: Path):
    s, e = _date_range(cfg)
    code = cfg['rqdata']['benchmark']
    px = rq.get_price(code, start_date=s, end_date=e, frequency='1d', fields=['close'], expect_df=True)
    px = _to_long(px, ['close'])
    px.to_parquet(raw_dir / 'benchmark.parquet', index=False)
    log.info(f'基准指数 {code} {len(px)} 天 -> benchmark.parquet')
    return px


NEWS_FIELDS = ['news_id', 'title', 'original_time', 'url', 'source', 'news_emotion_indicator',
               'news_positive_weight', 'news_negative_weight', 'company_relevance', 'company_emotion_indicator',
               'company_positive_weight', 'company_negative_weight']


def fetch_rq_news(rq, cfg, codes, raw_dir: Path):
    """米筐个股新闻（rqdatac.news.get_stock_news，2017 年起），逐只股票按年请求，保存到 raw/news_rq/{代码}.parquet。"""
    import rqdatac_news  # noqa: F401  导入后注册 rqdatac.news
    out = raw_dir / 'news_rq'
    out.mkdir(exist_ok=True)
    start, end = map(pd.Timestamp, _date_range(cfg))
    for code in tqdm(list(codes), desc='米筐新闻'):
        f = out / f'{code}.parquet'
        if f.exists():
            continue
        parts = []
        for y in range(start.year, end.year + 1):
            s, e = max(start, pd.Timestamp(y, 1, 1)), min(end, pd.Timestamp(y, 12, 31))
            df = rq.news.get_stock_news(code, s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d'))
            if df is not None and len(df):
                parts.append(df.reset_index())
        if not parts:
            log.warning(f'{code} 没有新闻数据')
            continue
        df = pd.concat(parts, ignore_index=True).drop_duplicates('news_id')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df[['order_book_id', 'datetime'] + [c for c in NEWS_FIELDS if c in df]].to_parquet(f, index=False)
        log.info(f'{code} 新闻 {len(df):,} 条')


def build_sentiment_from_news(cfg, raw_dir: Path):
    """由米筐新闻的公司情绪权重构造个股日度情绪指数，取值 [-1,1]。
    单条新闻得分 = 公司正面权重 - 公司负面权重（缺失时用新闻整体的权重），按公司相关度加权求日均值；
    收盘后发布的新闻归入下一交易日，与新闻清洗的对齐规则一致。"""
    from .news_clean import align_trading_day
    files = sorted((raw_dir / 'news_rq').glob('*.parquet'))
    if not files:
        log.warning('没有米筐新闻数据，无法构造情绪指数')
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    pos = df['company_positive_weight'].fillna(df['news_positive_weight'])
    neg = df['company_negative_weight'].fillna(df['news_negative_weight'])
    df['score'] = (pos - neg).clip(-1, 1)
    df['w'] = df['company_relevance'].fillna(1.0).clip(lower=0) + 1e-6
    df = align_trading_day(df.dropna(subset=['score']), load_trading_dates(raw_dir), cfg['news']['close_time'])
    df['ws'] = df['w'] * df['score']
    g = df.groupby(['order_book_id', 'date'])
    daily = (g['ws'].sum() / g['w'].sum()).rename('rq_senti').reset_index()
    out = raw_dir / 'rq_sentiment'
    out.mkdir(exist_ok=True)
    daily.to_parquet(out / 'part_0000.parquet', index=False)
    log.info(f'由米筐新闻构造日度情绪指数 {len(daily):,} 条，均值 {daily["rq_senti"].mean():+.3f}')
    return daily


def fetch_rq_sentiment(rq, cfg, codes, raw_dir: Path):
    """米筐个股情绪指数，取值 [-1,1]。rqdata.sentiment_source 为
    factor 时用 get_factor 读取账户开通的情绪因子（因子名填 rqdata.sentiment_factor）；
    news 时由米筐新闻的公司情绪权重聚合（需先 fetch_rq_news）。"""
    if cfg['rqdata'].get('sentiment_source', 'factor') == 'news':
        return build_sentiment_from_news(cfg, raw_dir)
    factor = cfg['rqdata'].get('sentiment_factor')
    if not factor:
        log.warning('未配置 rqdata.sentiment_factor，跳过米筐情绪指数（数据集中 rq_senti 置 0）')
        return None
    s, e = _date_range(cfg)
    out = raw_dir / 'rq_sentiment'
    out.mkdir(exist_ok=True)
    for i, part in tqdm(list(_chunks(list(codes), cfg['rqdata']['chunk_size'])), desc='米筐情绪指数'):
        f = out / f'part_{i:04d}.parquet'
        if f.exists():
            continue
        df = rq.get_factor(part, factor, start_date=s, end_date=e)
        if df is None or len(df) == 0:
            continue
        df = df.to_frame(factor) if isinstance(df, pd.Series) else df
        df = _to_long(df).rename(columns={factor: 'rq_senti'})
        df[['order_book_id', 'date', 'rq_senti']].to_parquet(f, index=False)


# ============================ 读取 ============================
def load_instruments(raw_dir: Path):
    return pd.read_csv(raw_dir / 'instruments.csv', dtype=str)


def load_trading_dates(raw_dir: Path):
    return pd.DatetimeIndex(pd.read_csv(raw_dir / 'trading_dates.csv', parse_dates=['date'])['date'])


def _load_parts(folder: Path, codes=None):
    files = sorted(folder.glob('part_*.parquet'))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df['order_book_id'].isin(codes)] if codes is not None else df


def load_price(raw_dir: Path, codes=None):
    df = _load_parts(raw_dir / 'price', codes)
    if df is None:
        raise FileNotFoundError(f'{raw_dir / "price"} 下没有行情数据，请先运行 01_fetch_market.py')
    return df


def load_benchmark(raw_dir: Path):
    return pd.read_parquet(raw_dir / 'benchmark.parquet').set_index('date')['close']


def load_rq_sentiment(raw_dir: Path, codes=None):
    return _load_parts(raw_dir / 'rq_sentiment', codes)
