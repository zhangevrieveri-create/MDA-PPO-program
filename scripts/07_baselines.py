"""步骤7：对比方法（论文表5、表7、表8）与夏普比率差异检验。
各方法只用训练集训练，在测试集上每日选得分最高的 m 只股票回测，与 MDA-PPO 用同一 run_backtest、metrics、signal_metrics。
若 outputs/<run>/test_nav.csv 已由 06_backtest.py 生成，则对 MDA-PPO 与各方法的逐日收益做 Ledoit-Wolf 夏普比率差异检验。

python scripts/07_baselines.py --config configs/default.yaml
python scripts/07_baselines.py --methods linear lstm random --features all     # 只跑部分方法、使用全部 25 维特征
"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from rlinvest.backtest import benchmark_nav, metrics, run_backtest, sharpe_diff_test, signal_metrics
from rlinvest.baselines import build_policy, feature_view
from rlinvest.features.dataset import load_dataset
from rlinvest.utils import get_device, get_logger, load_config, set_seed

NAMES = {'linear': '多元线性回归', 'lstm': 'LSTM选股', 'eiie': 'Jiang等（2017）', 'sood': 'Sood等（2023）',
         'random': '随机排序'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None)
    ap.add_argument('--methods', nargs='+', default=list(NAMES), choices=list(NAMES))
    ap.add_argument('--features', default=None, choices=['technical', 'all'], help='覆盖 baselines.features')
    ap.add_argument('--tag', default=None, help='MDA-PPO 的实验标签（与 05/06 相同），结果保存到其下的 baselines/')
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, name, rl, bc = cfg['paths'], cfg['dataset']['name'], dict(cfg['rl']), cfg['baselines']
    run = f'{name}_{args.tag}' if args.tag else name
    out = P['output'] / run / 'baselines'
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=P['output'] / 'logs' / f'07_baselines_{run}.log')
    rl['seed'] = bc['seed']
    set_seed(rl['seed'])
    device = get_device()

    data = load_dataset(P['processed'] / f'{name}.npz')
    feats = args.features or bc['features']
    view = feature_view(data, cfg['dataset']['technical']) if feats == 'technical' else data
    log.info(f'对比方法 {args.methods}；输入特征 {feats}（{len(view["features"])} 维）')

    bench = benchmark_nav(data, rl, 'test')
    navs, res, sig = {'基准指数': bench}, {'基准指数': metrics(bench, rl['rf'])}, {}
    mda = P['output'] / run / 'test_nav.csv'
    if mda.exists():
        nav = pd.read_csv(mda, index_col=0, parse_dates=True)['MDA-PPO']
        navs['MDA-PPO'], res['MDA-PPO'] = nav, metrics(nav, rl['rf'], bench)
        s = P['output'] / run / 'test_signal.csv'
        if s.exists():
            sig['MDA-PPO'] = pd.read_csv(s, index_col=0).iloc[:, 0]
    for k in args.methods:
        set_seed(rl['seed'])
        pol = build_policy(k, view, rl, bc, device, out)
        nav, w, trade = run_backtest(view, rl, pol, 'test')
        navs[NAMES[k]], res[NAMES[k]] = nav, metrics(nav, rl['rf'], bench, trade)
        sig[NAMES[k]] = pd.Series(signal_metrics(data, pol.scores, rl['rf']))
        log.info(f'[{NAMES[k]}] 年化 {res[NAMES[k]]["年化收益"]:.2%} 夏普 {res[NAMES[k]]["夏普比率"]:.3f} '
                 f'平均IC {sig[NAMES[k]]["平均IC"]:+.4f}')

    res, sig = pd.DataFrame(res), pd.DataFrame(sig)
    pd.DataFrame(navs).to_csv(out / 'test_nav.csv', encoding='utf-8-sig')
    res.to_csv(out / 'test_metrics.csv', encoding='utf-8-sig')
    sig.to_csv(out / 'test_signal.csv', encoding='utf-8-sig')
    log.info(f'表5 收益与风险：\n{res.round(4).to_string()}\n表7、表8 信号检验：\n{sig.round(4).to_string()}')

    if 'MDA-PPO' in navs:
        r0 = navs['MDA-PPO'].pct_change().dropna()
        tests = {k: sharpe_diff_test(r0, v.pct_change().dropna(), rl['rf'])
                 for k, v in navs.items() if k != 'MDA-PPO'}
        tests = pd.DataFrame(tests)
        tests.to_csv(out / 'sharpe_test.csv', encoding='utf-8-sig')
        log.info(f'MDA-PPO 与各方法的夏普比率差异检验（Ledoit-Wolf 2008，HAC）：\n{tests.round(4).to_string()}')
    else:
        log.info(f'未找到 {mda}，跳过夏普比率差异检验（先运行 06_backtest.py）')
    log.info(f'结果保存在 {out}')
