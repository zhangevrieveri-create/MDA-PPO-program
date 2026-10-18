"""步骤6：测试集（样本外）回测，对应论文表5、表7、表8。
对比：同一股票池的等权组合、基准指数；输出指标表、净值、每日持仓、信息系数与分组收益、图表。
其他对比方法（Jiang 等 2017、Sood 等 2023、LSTM 选股）需另行实现后，用同样的 run_backtest 与 signal_metrics 评价。"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from rlinvest.backtest import (benchmark_nav, equal_weight_policy, metrics, model_policy, run_backtest,
                               signal_metrics)
from rlinvest.data.rq_api import load_instruments
from rlinvest.features.dataset import load_dataset
from rlinvest.plotting import plot_nav, plot_training, plot_weights
from rlinvest.rl.ppo import load_model
from rlinvest.utils import get_device, get_logger, load_config

SEG_NAME = {'val': '验证集', 'test': '测试集'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=None)
    ap.add_argument('--ckpt', default='best.pt', help='best.pt 或 last.pt（没有验证集时二者相同）')
    ap.add_argument('--tag', default=None, help='实验标签，结果保存到 outputs/<数据集名>_<tag>/')
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, name = cfg['paths'], cfg['dataset']['name']
    run = f'{name}_{args.tag}' if args.tag else name
    out = P['output'] / run
    log = get_logger(log_file=P['output'] / 'logs' / f'06_backtest_{run}.log')
    device = get_device()

    data = load_dataset(P['processed'] / f'{name}.npz')
    model, ck = load_model(out / args.ckpt, device)
    rl = ck['cfg']
    log.info(f'加载 {args.ckpt}（第 {ck.get("update")} 次更新）')
    ins = load_instruments(P['raw'])
    code2name = dict(zip(ins['order_book_id'], ins['symbol']))

    segs = (['val'] if data['val_idx'] < data['test_idx'] else []) + ['test']
    for seg in segs:
        bench = benchmark_nav(data, rl, seg)
        ppo = model_policy(model, device, record=True)
        policies = {'MDA-PPO': ppo, '同池等权': equal_weight_policy()}
        navs, res = {}, {}
        for k, pol in policies.items():
            nav, weights, trade = run_backtest(data, rl, pol, seg)
            navs[k], res[k] = nav, metrics(nav, rl['rf'], bench, trade)
            if k == 'MDA-PPO':
                held = weights.loc[:, (weights > 0).any() | (weights.columns == 'cash')]   # 只保存出现过持仓的股票
                held.to_csv(out / f'{seg}_weights.csv', encoding='utf-8-sig')
                ppo_weights = held
        navs['基准指数'], res['基准指数'] = bench, metrics(bench, rl['rf'])
        res = pd.DataFrame(res)
        res.to_csv(out / f'{seg}_metrics.csv', encoding='utf-8-sig')
        pd.DataFrame(navs).to_csv(out / f'{seg}_nav.csv', encoding='utf-8-sig')
        sig = pd.Series(signal_metrics(data, ppo.scores, rl['rf']), name='MDA-PPO')
        sig.to_csv(out / f'{seg}_signal.csv', encoding='utf-8-sig')
        log.info(f'【{SEG_NAME[seg]}】{bench.index[0]:%Y-%m-%d} ~ {bench.index[-1]:%Y-%m-%d}\n'
                 f'表5 收益与风险：\n{res.round(4).to_string()}\n表7、表8 信号检验：\n{sig.round(4).to_string()}')
        plot_nav(navs, out / f'{seg}_nav.png', f'{SEG_NAME[seg]}净值曲线')
        top = ppo_weights.drop(columns='cash').mean().nlargest(12).index.tolist() + ['cash']
        plot_weights(ppo_weights[top].rename(columns=lambda c: code2name.get(c, c)),
                     out / f'{seg}_weights.png', f'{SEG_NAME[seg]}持仓权重（平均权重最高的 12 只与现金）')

    if (out / 'train_log.csv').exists():
        plot_training(pd.read_csv(out / 'train_log.csv'), out / 'train_curves.png')
    log.info(f'回测结果保存在 {out}')
