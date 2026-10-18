"""步骤2：按个股名称爬取新浪财经新闻标题，保存在 data/raw/news/{order_book_id}.csv，可断点续爬。

历史新闻不在此导入：放到配置项 news.history_dir 指定的文件夹，步骤3会自动合并。
"""
import argparse

import _bootstrap  # noqa: F401
from rlinvest.data import rq_api
from rlinvest.data.sina_news import SinaNewsCrawler
from rlinvest.utils import get_logger, load_config

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=None)
    ap.add_argument('--codes', nargs='*', help='只爬取这些 order_book_id')
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw = cfg['paths']['raw']
    get_logger(log_file=cfg['paths']['output'] / 'logs' / '02_crawl_news.log')

    ins = rq_api.load_instruments(raw)
    if args.codes:
        ins = ins[ins['order_book_id'].isin(args.codes)]
    SinaNewsCrawler(cfg, raw / 'news').run(ins)
