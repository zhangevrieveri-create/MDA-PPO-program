"""回测与评价指标（论文第四部分）：

表5  总收益率、年化收益率（总收益按复利换算）、夏普比率、卡玛比率（年化收益/最大回撤绝对值）、索提诺比率、最大回撤
表7  信息系数：策略评分 μ_{t,i} 与下一期收益率的横截面相关（IC）、秩相关（Rank IC）、ICIR、Rank ICIR，另给 Newey-West t 值
表8  按策略评分分十组：Top 10%、Bottom 10% 年化收益率，多空组合年化收益率与夏普比率
"""
import numpy as np
import pandas as pd
import torch

from .rl.env import PortfolioEnv, segment


def run_backtest(data, cfg, policy, name='test'):
    """在指定数据段逐日调仓。policy(obs, env) -> 目标权重。返回净值、每日目标权重、每日换手与成本。"""
    lo, hi = segment(data, name, cfg['T'])
    env = PortfolioEnv(data, cfg, lo, hi)
    obs, done = env.reset(), False
    nav, weights, trade = [1.0], [], []
    while not done:
        obs, _, done, info = env.step(policy(obs, env))
        nav.append(info['V'])
        weights.append(info['weights'])
        trade.append((info['turnover'], info['cost']))
    dates = data['dates'][lo:hi + 1]
    nav = pd.Series(nav, index=dates, name='nav')
    weights = pd.DataFrame(np.array(weights, dtype=np.float32), index=dates[:-1],
                           columns=list(data['codes']) + ['cash'])
    trade = pd.DataFrame(trade, index=dates[:-1], columns=['turnover', 'cost'])
    return nav, weights, trade


def model_policy(model, device, record=False):
    """测试阶段采用确定性策略：动作取 a_t = μ_t，同样截断、选前 m 只并归一化。record=True 时记录每日得分 μ_{t,i}。"""
    scores = {}

    @torch.no_grad()
    def f(obs, env):
        mu, _, _ = model(torch.as_tensor(obs, device=device)[None])
        mu = mu[0].cpu().numpy()
        if record:
            scores[env.t] = mu[:-1]
        return env.to_weights(mu)
    f.scores = scores
    return f


def equal_weight_policy():
    """同一股票池的等权组合：每日对全部已上市股票等权，不持现金。"""
    def f(obs, env):
        ok = env.selectable().astype(float)
        return np.append(ok / max(ok.sum(), 1.0), 0.0)
    return f


def benchmark_nav(data, cfg, name='test'):
    lo, hi = segment(data, name, cfg['T'])
    b = data['bench'][lo:hi + 1]
    return pd.Series(b / b[0], index=data['dates'][lo:hi + 1], name='benchmark')


def metrics(nav: pd.Series, rf=0.01, bench: pd.Series = None, trade: pd.DataFrame = None):
    r = nav.pct_change().dropna()
    n = len(r)
    ann = nav.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (1 - nav / nav.cummax()).max()
    down = r[r < 0].std() * np.sqrt(252)
    out = {'总收益率': nav.iloc[-1] - 1, '年化收益': ann, '年化波动': vol,
           '夏普比率': (ann - rf) / (vol + 1e-12), '卡玛比率': ann / (mdd + 1e-12),
           '索提诺比率': (ann - rf) / (down + 1e-12), '最大回撤': -mdd, '日胜率': (r > 0).mean()}
    if bench is not None:
        ex = r - bench.pct_change().dropna().reindex(r.index).fillna(0)
        out['超额年化收益'] = ex.mean() * 252
        out['信息比率'] = ex.mean() / (ex.std() + 1e-12) * np.sqrt(252)
    if trade is not None:
        out['日均换手率'] = trade['turnover'].mean()
        out['累计交易成本'] = trade['cost'].sum()
    return out


def newey_west_t(x, lags=5):
    """均值的 Newey-West t 值。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan


def sharpe_diff_test(r1, r2, rf=0.01, lags=None):
    """两个策略夏普比率之差的显著性检验（Ledoit 和 Wolf，2008，HAC 推断）。
    r1、r2 为同一组交易日的逐日收益；u = (μ1, μ2, γ1, γ2)（γ 为二阶原点矩），Δ = f(u) = μ1/√(γ1−μ1²) − μ2/√(γ2−μ2²)，
    用 delta 方法得 se(Δ) = √(∇fᵀ Ψ ∇f / n)，Ψ 为 (r−μ, r²−γ) 的 Newey-West（Bartlett 核）长期协方差。
    收益先减去日无风险利率；输出的夏普比率按 √252 年化，p 值为双侧。"""
    import math
    r = pd.concat([pd.Series(r1), pd.Series(r2)], axis=1).dropna().to_numpy(float) - rf / 252
    n = len(r)
    mu, gam = r.mean(0), (r ** 2).mean(0)
    var = gam - mu ** 2
    sr = mu / np.sqrt(var)
    grad = np.array([gam[0] / var[0] ** 1.5, -gam[1] / var[1] ** 1.5,
                     -mu[0] / (2 * var[0] ** 1.5), mu[1] / (2 * var[1] ** 1.5)])
    y = np.column_stack([r - mu, r ** 2 - gam])
    lags = int(4 * (n / 100) ** (2 / 9)) if lags is None else lags
    psi = y.T @ y / n
    for L in range(1, min(lags, n - 1) + 1):
        g = y[L:].T @ y[:-L] / n
        psi += (1 - L / (lags + 1)) * (g + g.T)
    se = float(np.sqrt(grad @ psi @ grad / n))
    z = (sr[0] - sr[1]) / se if se > 0 else np.nan
    k = np.sqrt(252)
    return {'夏普比率1': sr[0] * k, '夏普比率2': sr[1] * k, '夏普比率差': (sr[0] - sr[1]) * k,
            '标准误': se * k, 'z值': z, 'p值': math.erfc(abs(z) / math.sqrt(2)), '交易日数': n, 'HAC滞后阶数': lags}


def signal_metrics(data, scores: dict, rf=0.01, groups=10):
    """表7、表8：scores = {t: 第 t 日各股票得分}，与 t→t+1 收益做横截面检验。"""
    P, L = data['close'], data['listed']
    ic, ric, top, bot = [], [], [], []
    for t in sorted(scores):
        r = P[t + 1] / P[t] - 1
        ok = L[t] & np.isfinite(r) & np.isfinite(scores[t])
        if ok.sum() < 4:
            continue
        g = groups if ok.sum() >= 2 * groups else 2                 # 股票太少（如示例数据）时退化为按中位数分两组
        s, y = pd.Series(scores[t][ok]), pd.Series(r[ok])
        ic.append(s.corr(y))
        ric.append(s.rank().corr(y.rank()))
        q = pd.qcut(s.rank(method='first'), g, labels=False)
        top.append(y[q == g - 1].mean())
        bot.append(y[q == 0].mean())
    ic, ric, top, bot = map(np.array, (ic, ric, top, bot))
    ann = lambda x: np.prod(1 + x) ** (252 / len(x)) - 1
    ls = top - bot
    return {'平均IC': ic.mean(), 'IC标准差': ic.std(ddof=1), 'ICIR': ic.mean() / ic.std(ddof=1),
            'IC的NW-t值': newey_west_t(ic), 'Rank IC': ric.mean(), 'Rank ICIR': ric.mean() / ric.std(ddof=1),
            'Top 10%年化收益率': ann(top), 'Bottom 10%年化收益率': ann(bot),
            '多空组合年化收益率': ann(top) - ann(bot),
            '多空组合夏普比率': ls.mean() / ls.std(ddof=1) * np.sqrt(252), '交易日数': len(ic)}
