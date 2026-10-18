"""精确梯度诊断：不经过强化学习，直接以“一步超额收益”为可微目标训练同一网络。

若此时仍学不到已知信号，问题在网络结构或优化（如 sigmoid 饱和），而不在强化学习的梯度方差。
输出：验证集上持仓（μ）与真实信号的截面秩相关、训练批次的一步超额收益、μ 的饱和比例。

python experiments/sanity_check/exact_gradient.py --iters 1000
python experiments/sanity_check/exact_gradient.py --iters 1000 --override '{"logit_l2": 0}'
python experiments/sanity_check/exact_gradient.py --override '{"head": "shared", "asset_encoding": false}'
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlinvest.features.dataset import load_dataset  # noqa: E402
from rlinvest.rl.env import PortfolioEnv, segment  # noqa: E402
from rlinvest.rl.model import ActorCritic, calibrate_bn, xavier_init  # noqa: E402
from rlinvest.utils import get_device, load_config, set_seed  # noqa: E402


def run(cfg, iters=1000, batch=128, log_every=250, seed=0):
    rl = cfg['rl']
    set_seed(seed)
    dev = get_device()
    d = load_dataset(cfg['paths']['processed'] / f'{cfg["dataset"]["name"]}.npz')
    X, P, T, U = d['X'], d['close'], rl['T'], len(d['codes'])
    fi = d['features'].index('rq_senti')
    lo, hi = segment(d, 'train', T)
    vlo, vhi = segment(d, 'val', T)
    net = ActorCritic(X.shape[2], U, rl).to(dev)
    xavier_init(net, rl['sigma_init'])
    calibrate_bn(net, PortfolioEnv(d, rl, lo, hi, random_start=True), dev)
    opt = torch.optim.Adam(net.parameters(), lr=rl['lr'])
    windows = np.lib.stride_tricks.sliding_window_view(X, T, axis=0)       # (D-T+1, U, F, T)
    win = lambda ts: torch.as_tensor(windows[ts - T + 1].transpose(0, 1, 3, 2), device=dev)
    rank = lambda a: a.argsort(1).argsort(1)
    vts = np.arange(vlo, vhi)

    def val_ic():
        with torch.no_grad():
            mu = net(win(vts))[0][:, :U].cpu().numpy()
        return float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(rank(mu), rank(X[vts, :, fi]))]))

    rows, t0 = [], time.time()
    for it in range(1, iters + 1):
        ts = np.random.randint(lo, hi, batch)
        r = torch.as_tensor(P[ts + 1] / P[ts] - 1, dtype=torch.float32, device=dev)
        mu = net(win(ts))[0]
        w = mu / mu.sum(-1, keepdim=True)
        excess = ((w[:, :U] * r).sum(-1) - r.mean(-1)).mean() * 100          # 一步超额收益（%）
        loss = -excess + rl.get('logit_l2', 0) * (net.mu_logit ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % log_every == 0:
            sat = ((mu < 0.02) | (mu > 0.98)).float().mean().item()
            rows.append(dict(iter=it, val_rank_ic=val_ic(), train_excess_pct=excess.item(), saturation=sat))
            print(f'iter {it} 验证集秩相关 {rows[-1]["val_rank_ic"]:+.3f} 训练超额 {excess.item():+.3f}% '
                  f'饱和比例 {sat:.2f} 用时 {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='experiments/sanity_check/configs/synthetic.yaml')
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--override', default='{}', help='覆盖 rl 配置的 JSON，如 {"logit_l2": 0}')
    ap.add_argument('--out', default=None, help='结果 csv 路径')
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg['rl'].update(json.loads(args.override))
    df = run(cfg, args.iters)
    if args.out:
        df.assign(override=args.override).to_csv(args.out, index=False, encoding='utf-8-sig')
