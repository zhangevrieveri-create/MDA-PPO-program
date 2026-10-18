"""新闻清洗：合并爬取数据与历史数据、去 HTML、跨平台转载去重、按交易日对齐，生成图1示例与表1统计。"""
import re
from pathlib import Path

import pandas as pd

from ..utils import get_logger
from .sina_news import clean_name

log = get_logger()
_SPACE = re.compile(r'\s+')
_TAG = re.compile(r'<[^>]+>')
_PUNCT = re.compile(r'[\s　，。、；：！？“”‘’（）()\[\]【】《》<>|·…—\-_,.;:!?"\']')
COLUMNS = ['order_book_id', 'datetime', 'title', 'source']


def load_crawled_news(news_dir: Path):
    """读取 02_crawl_news.py 爬取的 raw/news/{order_book_id}.csv，以及米筐新闻 raw/news_rq/{order_book_id}.parquet。"""
    parts = []
    for f in sorted(news_dir.glob('*.csv')):
        df = pd.read_csv(f, parse_dates=['datetime'])
        if len(df):
            df['order_book_id'] = f.stem
            parts.append(df)
    for f in sorted((news_dir.parent / 'news_rq').glob('*.parquet')):
        df = pd.read_parquet(f, columns=['order_book_id', 'datetime', 'title', 'source'])
        if len(df):
            df['source'] = 'rqdata:' + df['source'].astype(str)
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(parts, ignore_index=True)[COLUMNS]


# =====================================================================
# 【历史新闻导入位置】
#   配置项 news.history_dir 指向历史新闻文件夹（null 时跳过）。
#   文件夹内可放任意个 .csv / .parquet 文件，每个文件需包含列：
#       datetime : 发布时间，如 2014-01-06 09:30:00
#       title    : 新闻标题
#       order_book_id 或 name : 米筐代码（如 300750.XSHE）或股票名称（如 宁德时代），二选一
#   导入后与爬取数据合并，统一进入下方的清洗、去重、交易日对齐流程。
# =====================================================================
def load_history_news(history_dir, instruments: pd.DataFrame):
    if not history_dir:
        return pd.DataFrame(columns=COLUMNS)
    history_dir = Path(history_dir)
    files = sorted(list(history_dir.glob('*.csv')) + list(history_dir.glob('*.parquet')))
    parts = [pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_csv(f) for f in files]
    if not parts:
        log.warning(f'{history_dir} 下没有历史新闻文件')
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    if 'order_book_id' not in df.columns:
        name2code = dict(zip(instruments['symbol'].map(clean_name), instruments['order_book_id']))
        df['order_book_id'] = df['name'].astype(str).map(clean_name).map(name2code)
        n0 = len(df)
        df = df.dropna(subset=['order_book_id'])
        if len(df) < n0:
            log.warning(f'历史新闻中 {n0 - len(df):,} 条的股票名称无法匹配到代码，已丢弃')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['source'] = 'history'
    log.info(f'导入历史新闻 {len(df):,} 条')
    return df.dropna(subset=['datetime'])[COLUMNS]


def clean_titles(df: pd.DataFrame):
    df = df.copy()
    df['title'] = df['title'].astype(str).map(lambda s: _SPACE.sub(' ', _TAG.sub('', s)).strip())
    df = df[df['title'].str.len() >= 4]
    # 不同财经平台大量转载：去掉标点空白后，同一股票同一天的相同标题只保留最早一条
    df['_key'] = df['title'].map(lambda s: _PUNCT.sub('', s))
    df['_day'] = df['datetime'].dt.normalize()
    df = df.sort_values('datetime').drop_duplicates(['order_book_id', '_day', '_key'])
    return df.drop(columns=['_key', '_day'])


def align_trading_day(df: pd.DataFrame, trading_dates: pd.DatetimeIndex, close_time='15:00'):
    """收盘后及非交易日发布的新闻归入下一交易日，保证 t 日状态只含 t 日收盘前可得的信息。"""
    hh, mm = map(int, close_time.split(':'))
    after = (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) >= hh * 60 + mm
    eff = df['datetime'].dt.normalize() + pd.to_timedelta(after.astype(int), unit='D')
    idx = trading_dates.searchsorted(eff.values, side='left')
    ok = idx < len(trading_dates)
    df = df[ok].copy()
    df['date'] = trading_dates[idx[ok]]
    return df


def news_stats(df: pd.DataFrame):
    """表1：舆情数据分布统计。"""
    m = df.groupby(df['date'].dt.to_period('M')).size()
    s = df.groupby('order_book_id').size()
    rows = [
        ('数据描述', '总股票数量', f'{s.size:,}只'),
        ('数据描述', '时间跨度', f'{df["date"].min():%Y.%m}-{df["date"].max():%Y.%m}'),
        ('数据描述', '新闻总数', f'{len(df):,}条'),
        ('时间维度', '平均每月新闻数', f'{m.mean():,.2f}条'),
        ('时间维度', '按月标准差', f'{m.std():,.2f}条'),
        ('时间维度', '最大月度新闻数', f'{m.max():,}条'),
        ('时间维度', '最小月度新闻数', f'{m.min():,}条'),
        ('个股维度', '平均每只股票新闻数', f'{s.mean():,.2f}条'),
        ('个股维度', '新闻最多的股票数量', f'{s.max():,}条'),
        ('个股维度', '新闻最少的股票数量', f'{s.min():,}条'),
        ('个股维度', '股票新闻数标准差', f'{s.std():,.2f}条'),
    ]
    return pd.DataFrame(rows, columns=['统计类别', '指标', '值'])


def build_clean_news(cfg, instruments, trading_dates):
    P = cfg['paths']
    crawled = load_crawled_news(P['raw'] / 'news')
    history = load_history_news(cfg['news'].get('history_dir'), instruments)
    parts = [x for x in (history, crawled) if len(x)]
    if not parts:
        raise RuntimeError('没有新闻数据：请先运行 02_crawl_news.py，或配置 news.history_dir')
    raw = pd.concat(parts, ignore_index=True)
    raw['datetime'] = pd.to_datetime(raw['datetime'])
    start, end = pd.Timestamp(cfg['rqdata']['start_date']), pd.Timestamp(cfg['rqdata']['end_date'])
    raw = raw[(raw['datetime'] >= start) & (raw['datetime'] < end + pd.Timedelta(days=1))]

    df = align_trading_day(clean_titles(raw), trading_dates, cfg['news']['close_time'])
    df = df[df['date'] <= end + pd.Timedelta(days=31)]
    df = df[['order_book_id', 'date', 'datetime', 'title', 'source']].sort_values(
        ['order_book_id', 'datetime']).reset_index(drop=True)
    df.to_parquet(P['interim'] / 'news_clean.parquet', index=False)

    # 图1：按日与股票索引清洗后的舆情新闻示例
    name = dict(zip(instruments['order_book_id'], instruments['symbol']))
    sample = df.sample(min(30, len(df)), random_state=0).sort_values(['date', 'order_book_id'])
    sample.insert(1, 'name', sample['order_book_id'].map(name))
    sample.to_csv(P['interim'] / 'news_sample.csv', index=False, encoding='utf-8-sig')

    stats = news_stats(df)
    stats.to_csv(P['interim'] / 'table1_news_stats.csv', index=False, encoding='utf-8-sig')
    log.info(f'新闻 {len(raw):,} 条（历史 {len(history):,} / 爬取 {len(crawled):,}）-> 清洗去重后 {len(df):,} 条')
    log.info('表1 舆情数据分布：\n' + stats.to_string(index=False))
    return df
