"""生成带已知信号的合成数据，格式与步骤1、2的输出一致，用于检验训练流程能否学到信号。

情绪 s_t = tanh(z_t)，z_t 为 AR(1) 过程（持续性 phi）；次日收益中含 alpha · s_t 的可预测部分。
新闻标题为 s_t 的带噪观测（正面/负面词），米筐情绪指数直接取 s_t。

python experiments/sanity_check/make_synthetic.py --config experiments/sanity_check/configs/synthetic.yaml
python experiments/sanity_check/make_synthetic.py --config ... --alpha 0.0008      # 弱信号
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlinvest.utils import load_config  # noqa: E402

POS = ['大增', '中标', '签订订单', '回购', '增持', '业绩预增']
NEG = ['减持', '立案调查', '亏损', '质押', '处罚', '下滑']


def generate(raw: Path, start, end, alpha=0.003, phi=0.97, n=12, news_prob=0.35, seed=1):
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    td = pd.bdate_range(start, pd.Timestamp(end) + pd.Timedelta(days=32))
    pd.Series(td, name='date').to_csv(raw / 'trading_dates.csv', index=False)
    d = td[td <= end]
    D = len(d)
    codes = [f'{600000 + i:06d}.XSHG' for i in range(n)]
    names = [f'测试股份{i}' for i in range(n)]
    pd.DataFrame({'order_book_id': codes, 'symbol': names, 'listed_date': '2010-01-01',
                  'de_listed_date': '0000-00-00', 'exchange': 'XSHG'}).to_csv(
        raw / 'instruments.csv', index=False, encoding='utf-8-sig')
    for sub in ['price', 'news', 'rq_sentiment']:
        (raw / sub).mkdir(exist_ok=True)

    mkt = rng.normal(0.0002, 0.012, D)
    pd.DataFrame({'order_book_id': '000300.XSHG', 'date': d, 'close': 3000 * np.exp(np.cumsum(mkt))}).to_parquet(
        raw / 'benchmark.parquet', index=False)
    prices, sentis = [], []
    for code, name in zip(codes, names):
        z, e = np.zeros(D), rng.normal(0, 1, D)
        for t in range(1, D):
            z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
        s = np.tanh(z)
        r = mkt * rng.uniform(.7, 1.3) + rng.normal(0, .016, D) + alpha * np.r_[0, s[:-1]]
        close = 10 * np.exp(np.cumsum(r))
        open_ = close * (1 + rng.normal(0, .004, D))
        vol = rng.lognormal(15, .4, D)
        vol[rng.random(D) < .005] = 0                                   # 约 0.5% 的交易日停牌
        prev = np.r_[close[0], close[:-1]]
        prices.append(pd.DataFrame({
            'order_book_id': code, 'date': d, 'open': open_, 'high': np.maximum(open_, close) * 1.004,
            'low': np.minimum(open_, close) * .996, 'close': close, 'volume': vol, 'total_turnover': vol * close,
            'limit_up': prev * 1.1, 'limit_down': prev * .9, 'turnover': vol / 1e8}))
        sentis.append(pd.DataFrame({'order_book_id': code, 'date': d, 'rq_senti': s}))
        k = rng.random(D) < news_prob
        words = [rng.choice(POS) if x > rng.normal(0, .3) else rng.choice(NEG) for x in s[k]]
        t = d[k] + pd.to_timedelta(rng.integers(9 * 60, 14 * 60, k.sum()), unit='m')
        pd.DataFrame({'datetime': t, 'title': [f'{name}{w}' for w in words], 'url': '', 'source': 'stock_list'}).to_csv(
            raw / 'news' / f'{code}.csv', index=False, encoding='utf-8-sig')
    pd.concat(prices).to_parquet(raw / 'price' / 'part_0000.parquet', index=False)
    pd.concat(sentis).to_parquet(raw / 'rq_sentiment' / 'part_0000.parquet', index=False)
    return D


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='experiments/sanity_check/configs/synthetic.yaml')
    ap.add_argument('--alpha', type=float, default=0.003, help='信号对次日收益的预测力（强信号 0.003，弱信号 0.0008）')
    ap.add_argument('--phi', type=float, default=0.97, help='情绪 AR(1) 持续性')
    ap.add_argument('--n', type=int, default=12, help='股票数量')
    ap.add_argument('--news-prob', type=float, default=0.35, help='每个交易日有新闻的概率')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    D = generate(cfg['paths']['raw'], cfg['rqdata']['start_date'], cfg['rqdata']['end_date'],
                 args.alpha, args.phi, args.n, args.news_prob, args.seed)
    print(f'合成数据 {D} 个交易日、{args.n} 只股票 -> {cfg["paths"]["raw"]}')
