"""汇总多组实验：测试集指标、持仓与真实信号的截面秩相关、训练诊断。

python experiments/sanity_check/compare.py --config experiments/sanity_check/configs/synthetic.yaml \
    --runs 默认=outputs_synthetic/u10_default 行为策略裁剪=outputs_synthetic/u10_behavior
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlinvest.utils import ROOT, load_config  # noqa: E402


def signal_panel(raw: Path):
    s = pd.read_parquet(raw / 'rq_sentiment' / 'part_0000.parquet')
    return s.pivot(index='date', columns='order_book_id', values='rq_senti')


def summarize(run_dir: Path, signal: pd.DataFrame):
    import torch
    out = {}
    if (run_dir / 'test_metrics.csv').exists():
        m = pd.read_csv(run_dir / 'test_metrics.csv', index_col=0)
        w = pd.read_csv(run_dir / 'test_weights.csv', index_col=0, parse_dates=True).drop(columns='cash')
        sig = signal.reindex(w.index)[w.columns]
        out.update({'PPO年化': m.loc['年化收益', 'PPO智能投资'], '等权年化': m.loc['年化收益', '等权再平衡'],
                    '相对等权超额': m.loc['年化收益', 'PPO智能投资'] - m.loc['年化收益', '等权再平衡'],
                    'PPO夏普': m.loc['夏普比率', 'PPO智能投资'], '等权夏普': m.loc['夏普比率', '等权再平衡'],
                    'PPO最大回撤': m.loc['最大回撤', 'PPO智能投资'], 'PPO日均换手': m.loc['日均换手率', 'PPO智能投资'],
                    '持仓-信号秩相关': w.rank(axis=1).corrwith(sig.rank(axis=1), axis=1).mean()})
    if (run_dir / 'best.pt').exists():
        ck = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
        out['最优模型更新轮次'] = ck.get('update')
    if (run_dir / 'train_log.csv').exists():
        log = pd.read_csv(run_dir / 'train_log.csv')
        tail = lambda k: log[k].dropna().tail(50).mean() if k in log else np.nan
        out.update({'最优验证夏普': log['val_sharpe'].max() if 'val_sharpe' in log else np.nan,
                    '后期裁剪比例': tail('clip_frac'), '后期KL': tail('approx_kl'), '后期价值损失': tail('value_loss'),
                    '后期σ': tail('sigma')})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='experiments/sanity_check/configs/synthetic.yaml')
    ap.add_argument('--runs', nargs='+', required=True, help='名称=结果目录')
    ap.add_argument('--out', default=None, help='结果 csv 路径')
    args = ap.parse_args()
    cfg = load_config(args.config)
    signal = signal_panel(cfg['paths']['raw'])
    res = {}
    for item in args.runs:
        name, path = item.split('=', 1)
        p = Path(path)
        res[name] = summarize(p if p.is_absolute() else ROOT / p, signal)
    df = pd.DataFrame(res)
    pd.set_option('display.width', 250)
    print(df.round(4).to_string())
    if args.out:
        df.to_csv(args.out, encoding='utf-8-sig')
