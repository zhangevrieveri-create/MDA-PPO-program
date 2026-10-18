"""新浪财经个股新闻标题爬虫。

两个数据源：
  search     : 新浪搜索 JSON 接口，按“个股名称”检索新闻（论文做法）
  stock_list : 个股新闻列表页 vCB_AllNewsStock，按股票代码翻页
注意：两个公开接口的可翻页深度都有限（列表页约保留最近 1000 条），适合增量更新。
2014 年至今的完整历史新闻请放到【历史新闻导入位置】：配置项 news.history_dir，
读取逻辑见 rlinvest/data/news_clean.py 中的 load_history_news()。

输出：raw/news/{order_book_id}.csv，列为 datetime,title,url,source，可断点续爬。
"""
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from ..utils import get_logger

log = get_logger()
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')
TAG = re.compile(r'<[^>]+>')


def to_sina_symbol(order_book_id: str) -> str:
    code, ex = order_book_id.split('.')
    return ('sh' if ex == 'XSHG' else 'sz') + code


def _bj_time(item) -> pd.Timestamp:
    """搜索结果的 dataTime 可能是“3小时前”，统一用 ctime（Unix 秒）换算为北京时间。"""
    return pd.Timestamp(int(item['ctime']), unit='s') + pd.Timedelta(hours=8)


def clean_name(name: str) -> str:
    return name.replace('*', '').replace(' ', '')


class SinaNewsCrawler:
    LIST_URL = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{sym}.phtml'
    SEARCH_URL = 'https://search.sina.com.cn/api/news'
    LIST_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2})&nbsp;&nbsp;"
                          r"<a[^>]*href=['\"]([^'\"]+)['\"][^>]*>([^<]+)</a>")

    def __init__(self, cfg, out_dir: Path):
        self.c = cfg['news']
        self.start = pd.Timestamp(cfg['rqdata']['start_date'])
        self.end = pd.Timestamp(cfg['rqdata']['end_date']) + pd.Timedelta(days=1)
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)

    # ---------------- HTTP ----------------
    def _session(self):
        s = requests.Session()
        retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retry))
        s.mount('http://', HTTPAdapter(max_retries=retry))
        s.headers.update({'User-Agent': UA, 'Referer': 'https://finance.sina.com.cn/'})
        return s

    def _sleep(self):
        time.sleep(random.uniform(*self.c['sleep']))

    # ---------------- 数据源 1：按名称搜索 ----------------
    def crawl_search(self, sess, name):
        rows, seen, failed = [], set(), False
        for page in range(1, self.c['max_pages'] + 1):
            try:
                r = sess.get(self.SEARCH_URL, params=dict(q=name, page=page, size=self.c['page_size'], sort=1),
                             timeout=self.c['timeout'])
                items = (r.json().get('data') or {}).get('list') or []
            except Exception as e:
                log.debug(f'{name} 搜索第{page}页失败：{e}')
                failed = page == 1
                break
            new = [x for x in items if x.get('url') not in seen]
            if not new:
                break
            for x in new:
                seen.add(x['url'])
                rows.append((_bj_time(x), TAG.sub('', x['title']), x['url'].split('?from=')[0], 'search'))
            if _bj_time(new[-1]) < self.start:
                break
            self._sleep()
        return rows, failed

    # ---------------- 数据源 2：个股新闻列表页 ----------------
    def crawl_stock_list(self, sess, order_book_id):
        url, rows, seen, failed = self.LIST_URL.format(sym=to_sina_symbol(order_book_id)), [], set(), False
        for page in range(1, self.c['max_pages'] + 1):
            try:
                r = sess.get(url, params={'Page': page}, timeout=self.c['timeout'])
                r.encoding = 'gbk'
                items = self.LIST_PAT.findall(r.text)
            except Exception as e:
                log.debug(f'{order_book_id} 列表页第{page}页失败：{e}')
                failed = page == 1
                break
            new = [x for x in items if x[2] not in seen]
            if not new:
                break
            for d, t, link, title in new:
                seen.add(link)
                rows.append((f'{d} {t}:00', title.strip(), link, 'stock_list'))
            if pd.Timestamp(new[-1][0]) < self.start:
                break
            self._sleep()
        return rows, failed

    # ---------------- 单只股票 ----------------
    def crawl_one(self, order_book_id, name):
        f = self.out / f'{order_book_id}.csv'
        if f.exists():
            return order_book_id, -1
        sess, rows, failed = self._session(), [], False
        if 'search' in self.c['sources']:
            r, f1 = self.crawl_search(sess, clean_name(name))
            rows, failed = rows + r, failed or f1
        if 'stock_list' in self.c['sources']:
            r, f2 = self.crawl_stock_list(sess, order_book_id)
            rows, failed = rows + r, failed or f2
        if failed and not rows:
            raise ConnectionError(f'{order_book_id} 请求失败，未写入文件，下次运行将重试')
        df = pd.DataFrame(rows, columns=['datetime', 'title', 'url', 'source'])
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df = df[(df['datetime'] >= self.start) & (df['datetime'] < self.end)]
        if self.c['require_name_in_title']:
            df = df[df['title'].str.contains(re.escape(clean_name(name)), regex=True)]
        df = df.drop_duplicates('url').sort_values('datetime')
        tmp = f.with_suffix('.tmp')
        df.to_csv(tmp, index=False, encoding='utf-8-sig')
        tmp.replace(f)  # 原子写入，中断不会留下半个文件
        return order_book_id, len(df)

    def run(self, instruments: pd.DataFrame):
        jobs = list(zip(instruments['order_book_id'], instruments['symbol']))
        total = 0
        with ThreadPoolExecutor(self.c['workers']) as pool:
            futs = [pool.submit(self.crawl_one, c, n) for c, n in jobs]
            for fu in tqdm(as_completed(futs), total=len(futs), desc='新浪新闻'):
                try:
                    _, n = fu.result()
                    total += max(n, 0)
                except Exception as e:
                    log.warning(f'爬取失败：{e}')
        log.info(f'本次新增新闻 {total} 条，保存在 {self.out}')
