"""自检：不需要任何数据，几分钟内检查模型结构与训练流程是否与论文一致。

1. F=25 时表3总参数量应为 1,477,253，且与股票数 U 无关
2. 在随机数据上跑完整的 PPO 训练（缓存区存满 C 步更新一次并清空）、测试集回测与信号检验
3. 表6 消融开关、夏普比率差异检验，以及各对比方法的训练与回测流程

python scripts/00_selftest.py
"""
import tempfile
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from rlinvest.backtest import metrics, model_policy, run_backtest, sharpe_diff_test, signal_metrics
from rlinvest.baselines import build_policy, feature_view
from rlinvest.rl.env import select_and_normalize
from rlinvest.rl.model import ActorCritic, param_table
from rlinvest.rl.ppo import PPOTrainer
from rlinvest.utils import get_device, get_logger, load_config, set_seed

PAPER_TOTAL = 1_477_253

if __name__ == '__main__':
    log = get_logger()
    cfg = load_config()
    rl = dict(cfg['rl'])
    set_seed(0)

    # 1. 表3 参数量
    for U in (10, 3699):
        total = int(param_table(ActorCritic(25, U, rl))['参数数量'].iloc[-1])
        assert total == PAPER_TOTAL, f'U={U} 时总参数量 {total:,}，论文为 {PAPER_TOTAL:,}'
    log.info(f'[通过] 表3 总参数量 {PAPER_TOTAL:,}，与股票数无关')
    log.info('\n' + param_table(ActorCritic(25, 10, rl)).to_string(index=False))

    # 2. 随机数据上的训练与回测
    rng = np.random.default_rng(0)
    D, U, F = 400, 40, 25
    listed = np.ones((D, U), bool)
    listed[:150, -3:] = False                                  # 3 只股票样本中途上市
    close = 10 * np.cumprod(1 + rng.normal(0.0003, 0.02, (D, U)), 0)
    close[~listed] = np.nan
    X = rng.normal(size=(D, U, F)).astype(np.float32)
    X[~listed] = 0
    data = dict(X=X, close=close, listed=listed, buy_ok=listed.copy(), sell_ok=listed.copy(),
                bench=3000 * np.cumprod(1 + rng.normal(0, 0.012, D)),
                dates=__import__('pandas').bdate_range('2020-01-01', periods=D),
                codes=[f'S{i:03d}' for i in range(U)], features=[f'f{i}' for i in range(F)],
                val_idx=300, test_idx=300)                     # 与论文相同：只有训练集和测试集
    rl.update(C=64, B=16, E=2, K=200, micro_batch=8, adapt_interval=50)
    device = get_device()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = PPOTrainer(data, rl, device, Path(tmp))
        logs = trainer.train()
        assert trainer.buf.size == rl['K'] % rl['C'], '缓存区应在每次更新后清空'
        assert (Path(tmp) / 'best.pt').exists() and (Path(tmp) / 'last.pt').exists()
        trainer.model.eval()
        pol = model_policy(trainer.model, device, record=True)
        nav, w, trade = run_backtest(data, rl, pol, 'test')
        held = (w.drop(columns='cash') > 0).sum(axis=1)
        assert held.max() <= rl['m'], '每日持股数不应超过 m'
        m = metrics(nav, rl['rf'], trade=trade)
        s = signal_metrics(data, pol.scores)
    log.info(f'[通过] 训练 {rl["K"]} 步、更新 {rl["K"] // rl["C"]} 次；测试集年化 {m["年化收益"]:.2%}，'
             f'日均换手 {m["日均换手率"]:.2f}，最多同时持有 {held.max()} 只；平均 IC {s["平均IC"]:+.3f}')

    # 3. 表6 消融开关、夏普比率差异检验、对比方法
    a, ok = np.array([0.9, 0.2, 0.6, 0.1, 0.5]), np.ones(4, bool)
    w = select_and_normalize(a, 2, ok)
    assert np.allclose(w, [0.9 / 2, 0, 0.6 / 2, 0, 0.5 / 2]), w
    w = select_and_normalize(a, 2, ok, cash=False)
    assert np.allclose(w, [0.6, 0, 0.4, 0, 0]), w
    w = select_and_normalize(a, 2, ok, cash=False, weighting='equal')
    assert np.allclose(w, [0.5, 0, 0.5, 0, 0]), w
    r1 = rng.normal(0.001, 0.01, 500)
    t = sharpe_diff_test(r1, r1 + rng.normal(0, 0.002, 500))
    assert np.isfinite(t['z值']) and 0 <= t['p值'] <= 1
    assert abs(sharpe_diff_test(r1, r1)['夏普比率差']) < 1e-12
    log.info('[通过] 表6 消融开关（w/o RF、w/o DW）与夏普比率差异检验')
    bc = cfg['baselines']
    bc = {**bc, 'lstm': {**bc['lstm'], 'steps': 20}, 'eiie': {**bc['eiie'], 'steps': 20},
          'sood': {**bc['sood'], 'K': 128}}
    view = feature_view(data, data['features'][:22])
    with tempfile.TemporaryDirectory() as tmp:
        for k in ('linear', 'lstm', 'eiie', 'sood', 'random'):
            pol = build_policy(k, view, rl, bc, device, Path(tmp))
            nav, w, trade = run_backtest(view, rl, pol, 'test')
            assert (w.drop(columns='cash') > 0).sum(axis=1).max() <= rl['m'], k
            s = signal_metrics(data, pol.scores)
            assert np.isfinite(nav.iloc[-1]) and np.isfinite(s['平均IC']), k
    log.info('[通过] 对比方法（多元线性回归、LSTM、EIIE、Sood、随机排序）训练与回测流程')
    log.info('自检全部通过（随机数据上的收益没有意义，只检查流程）')
