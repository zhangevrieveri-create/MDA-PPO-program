"""步骤1：米筐 RQData —— 股票池、交易日历、日线量价（含涨跌停价）与换手率、基准指数、新闻舆情、个股情绪指数。"""
import argparse

import _bootstrap  # noqa: F401
from rlinvest.data import rq_api
from rlinvest.utils import get_logger, load_config

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw = cfg['paths']['raw']
    get_logger(log_file=cfg['paths']['output'] / 'logs' / '01_fetch_market.log')

    rq = rq_api.init_rq(cfg)
    ins = rq_api.fetch_instruments(rq, cfg, raw)
    rq_api.fetch_trading_dates(rq, cfg, raw)
    rq_api.fetch_benchmark(rq, cfg, raw)
    rq_api.fetch_price(rq, cfg, ins['order_book_id'], raw)
    if cfg['rqdata'].get('news'):
        rq_api.fetch_rq_news(rq, cfg, ins['order_book_id'], raw)
    rq_api.fetch_rq_sentiment(rq, cfg, ins['order_book_id'], raw)
