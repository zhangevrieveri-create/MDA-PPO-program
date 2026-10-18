"""预言机基准：直接按已知信号（米筐情绪特征）倾斜配置 w ∝ exp(k·s)，给出可学习收益的上限参照。

python experiments/sanity_check/oracle.py --config experiments/sanity_check/configs/synthetic.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlinvest.backtest import equal_weight_policy, metrics, run_backtest  # noqa: E402
from rlinvest.features.dataset import load_dataset  # noqa: E402
from rlinvest.utils import load_config  # noqa: E402


def tilt_policy(feature_idx, k):
    def f(obs, env):
        w = np.exp(k * obs[:, -1, feature_idx])
        return np.append(w / w.sum(), 0.0)
    return f


def oracle_table(cfg, ks=(0.5, 1.0, 2.0)):
    rl = cfg['rl']
    data = load_dataset(cfg['paths']['processed'] / f'{cfg["dataset"]["name"]}.npz')
    fi = data['features'].index('rq_senti')
    rows = []
    for seg in ['val', 'test']:
        pols = {'等权': equal_weight_policy(len(data['codes']))}
        pols.update({f'预言机 k={k}': tilt_policy(fi, k) for k in ks})
        for name, pol in pols.items():
            nav, _, trade = run_backtest(data, rl, pol, seg)
            m = metrics(nav, rl['rf'], trade=trade)
            rows.append({'数据段': seg, '策略': name, '年化收益': m['年化收益'], '夏普比率': m['夏普比率'],
                         '最大回撤': m['最大回撤'], '日均换手率': m['日均换手率']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='experiments/sanity_check/configs/synthetic.yaml')
    ap.add_argument('--out', default=None, help='结果 csv 路径')
    args = ap.parse_args()
    df = oracle_table(load_config(args.config))
    print(df.round(4).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False, encoding='utf-8-sig')
