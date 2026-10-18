"""论文表5~表8 的对比方法。各方法每日按自己输出的个股得分选取前 m 只股票（论文 m=10），
与 MDA-PPO 使用同一回测环境（交易成本、停牌与涨跌停限制）和同一组评价指标。

多元线性回归   ：t 日特征对 t→t+1 收益的混合 OLS，得分为预测值；入选股票等权（表6 末行、表7、表8）
LSTM 选股      ：单层 LSTM 读取 T 日特征窗口预测次日收益（MSE），得分为预测值；入选股票等权
Jiang 等（2017）：EIIE——各股票共享参数的卷积评估器读取 T 日收盘价相对序列与上一日权重，另设现金偏置；
                  入选股票与现金按 softmax 配权；以扣除交易成本后的平均对数收益为目标直接梯度上升，
                  上一日权重取自组合向量记忆（PVM）。数据集中没有原始最高价、最低价，只用收盘价序列
Sood 等（2023） ：PPO + 差分夏普比率奖励，两层全连接网络（64，tanh）；训练时不计交易成本（与原文一致），回测时扣除
随机排序       ：每日随机得分（表7）
监督学习的目标收益先做横截面去均值并截断到 ±20%。超参数见配置 baselines 一节，论文未给出。
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .backtest import model_policy
from .rl.env import segment
from .rl.model import ActorCritic, MLPEncoder
from .rl.ppo import PPOTrainer
from .utils import get_logger

log = get_logger()


# ---------------- 数据 ----------------
def feature_view(data, names):
    """只保留指定特征的数据集视图（不复制其余字段）。"""
    idx = [data['features'].index(n) for n in names]
    return {**data, 'X': np.ascontiguousarray(data['X'][..., idx]), 'features': list(names)}


def train_targets(data, T):
    """训练段每个决策日 t 的次日收益（横截面去均值、截断）与可用掩码。"""
    lo, hi = segment(data, 'train', T)
    P = data['close']
    ts = np.arange(lo, hi)
    with np.errstate(invalid='ignore', divide='ignore'):
        y = P[ts + 1] / P[ts] - 1
    ok = data['listed'][ts] & np.isfinite(y)
    y = np.where(ok, y, np.nan)
    y = np.clip(y - np.nanmean(y, 1, keepdims=True), -0.2, 0.2)
    return ts, np.nan_to_num(y).astype(np.float32), ok


def topm_equal(scores, env, m):
    """入选得分最高的 m 只已上市股票，等权，不持现金。"""
    s = np.where(env.selectable(), scores, -np.inf)
    k = min(m, int(np.isfinite(s).sum()))
    w = np.zeros(len(scores) + 1)
    if k > 0:
        w[np.argpartition(-s, k - 1)[:k]] = 1.0 / k
    else:
        w[-1] = 1.0
    return w


def score_policy(score_fn, m):
    """score_fn(obs, env) -> 各股票得分；记录得分用于信号检验。"""
    scores = {}

    def f(obs, env):
        s = np.asarray(score_fn(obs, env), float)
        scores[env.t] = s
        return topm_equal(s, env, m)
    f.scores = scores
    return f


# ---------------- 多元线性回归 ----------------
def fit_linear(data, cfg):
    ts, y, ok = train_targets(data, cfg['T'])
    F = data['X'].shape[2]
    A, b = np.zeros((F + 1, F + 1)), np.zeros(F + 1)
    for i, t in enumerate(ts):                                 # 逐日累加正规方程，避免全市场时一次性展开
        Z = np.c_[np.ones(ok[i].sum()), data['X'][t, ok[i]]]
        A += Z.T @ Z
        b += Z.T @ y[i, ok[i]]
    beta = np.linalg.solve(A + 1e-6 * np.eye(F + 1), b)
    return lambda obs, env: obs[:, -1, :] @ beta[1:] + beta[0]


# ---------------- LSTM 选股 ----------------
class LSTMScorer(nn.Module):
    def __init__(self, F, hidden):
        super().__init__()
        self.lstm = nn.LSTM(F, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                                      # x: (n,T,F)
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def fit_lstm(data, cfg, c, device):
    T = cfg['T']
    ts, y, ok = train_targets(data, T)
    X = data['X']
    model = LSTMScorer(X.shape[2], c['hidden']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
    rng = np.random.default_rng(cfg['seed'])
    for step in range(1, c['steps'] + 1):
        xs, ys = [], []
        for i in rng.choice(len(ts), c['days_per_batch'], replace=False):
            j = np.flatnonzero(ok[i])
            j = rng.choice(j, min(len(j), c['stocks_per_day']), replace=False)
            xs.append(X[ts[i] - T + 1:ts[i] + 1, j].transpose(1, 0, 2))
            ys.append(y[i, j])
        x = torch.as_tensor(np.concatenate(xs), device=device)
        loss = ((model(x) - torch.as_tensor(np.concatenate(ys), device=device)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, c['steps'] // 5) == 0:
            log.info(f'[LSTM] step {step}/{c["steps"]} mse={loss.item():.6f}')
    model.eval()

    @torch.no_grad()
    def score(obs, env):
        return model(torch.as_tensor(obs, device=device)).cpu().numpy()
    return score


# ---------------- Jiang 等（2017）EIIE ----------------
class EIIE(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 2, 3)
        self.conv2 = nn.Conv1d(2, 20, T - 2)
        self.out = nn.Linear(21, 1)                            # 等价于 1×1 卷积：20 个特征 + 上一日权重
        self.cash_bias = nn.Parameter(torch.zeros(1))

    def forward(self, px, w_prev):                             # px: (N,U,T)，w_prev: (N,U)
        N, U, T = px.shape
        h = torch.relu(self.conv2(torch.relu(self.conv1(px.reshape(N * U, 1, T))))).reshape(N, U, 20)
        return self.out(torch.cat([h, w_prev[..., None]], -1)).squeeze(-1)      # 各股票得分 (N,U)

    def weights(self, s, selectable, m):
        """入选得分最高的 m 只股票与现金按 softmax 配权；返回 (N,U+1)，最后一维为现金。"""
        s = s.masked_fill(~selectable, -torch.inf)
        k = min(m, s.shape[1])
        thr = s.detach().topk(k, dim=1).values[:, -1:]
        logits = torch.cat([s.masked_fill(s < thr, -torch.inf), self.cash_bias.expand(len(s), 1)], 1)
        return torch.softmax(logits, 1)


def price_windows(data, ts, T):
    """t 日的收盘价相对序列 P_{t−τ}/P_t，τ = T−1..0；缺失（未上市）取 1。返回 (len(ts),U,T)。"""
    P = data['close']
    with np.errstate(invalid='ignore', divide='ignore'):
        w = np.stack([P[t - T + 1:t + 1].T / P[t][:, None] for t in ts])
    return np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)


def fit_eiie(data, cfg, c, device):
    T, m, tau, rf_d = cfg['T'], cfg['m'], cfg['tau'], cfg['rf'] / 252
    lo, hi = segment(data, 'train', T)
    P, L = data['close'], data['listed']
    U = P.shape[1]
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.nan_to_num(P[1:] / P[:-1] - 1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)   # R[t]: t→t+1
    pvm = np.zeros((len(P), U), np.float32)                    # 组合向量记忆：各日股票权重
    model = EIIE(T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
    rng = np.random.default_rng(cfg['seed'])
    nb = min(c['batch_days'], hi - lo - 1)
    for step in range(1, c['steps'] + 1):
        t0 = int(rng.integers(lo + 1, hi - nb + 1))
        ts = np.arange(t0, t0 + nb)
        tt = lambda x: torch.as_tensor(x, device=device)
        px, prev = tt(price_windows(data, ts, T)), tt(pvm[ts - 1])
        w = model.weights(model(px, prev), tt(L[ts]), m)       # (nb,U+1)
        r = tt(R[ts])
        g = 1 + R[ts - 1]                                      # 上一日权重经价格变动后的实际持仓
        drift = pvm[ts - 1] * g
        drift = tt(drift / np.maximum(drift.sum(1, keepdims=True) + (1 - pvm[ts - 1].sum(1, keepdims=True)) * (1 + rf_d),
                                      1e-8))
        cost = tau * (w[:, :-1] - drift).abs().sum(1)
        ret = (w[:, :-1] * r).sum(1) + w[:, -1] * rf_d - cost
        loss = -torch.log(torch.clamp(1 + ret, min=1e-6)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        pvm[ts] = w[:, :-1].detach().cpu().numpy()
        if step % max(1, c['steps'] // 5) == 0:
            log.info(f'[EIIE] step {step}/{c["steps"]} 平均对数收益={-loss.item():+.5f}')
    model.eval()
    return model


def eiie_policy(model, data, cfg, device):
    scores = {}

    @torch.no_grad()
    def f(obs, env):
        t = env.t
        px = torch.as_tensor(price_windows(data, [t], cfg['T']), device=device)
        prev = torch.as_tensor(env.w[None, :-1], dtype=torch.float32, device=device)
        s = model(px, prev)
        scores[t] = s[0].cpu().numpy()
        return model.weights(s, torch.as_tensor(env.selectable()[None], device=device), cfg['m'])[0].cpu().numpy()
    f.scores = scores
    return f


# ---------------- Sood 等（2023）----------------
def fit_sood(data, cfg, c, device, out_dir):
    tc = {**cfg, 'w1': 0.0, 'w2': 1.0, 'w3': 0.0, 'tau': c['train_tau'], 'K': c['K'],
          'cash': True, 'weighting': 'score'}
    model_fn = lambda F, U, cc: ActorCritic(F, U, cc, encoder=MLPEncoder(F, cc, c['hidden']))
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = PPOTrainer(data, tc, device, out_dir, model_fn=model_fn)
    trainer.train()
    trainer.model.eval()
    return trainer.model


# ---------------- 统一入口 ----------------
def build_policy(method, data, cfg, bcfg, device, out_dir):
    """训练（只用训练集）并返回回测策略；策略带 scores 属性供信号检验。"""
    t0 = time.time()
    m = cfg['m']
    if method == 'linear':
        pol = score_policy(fit_linear(data, cfg), m)
    elif method == 'lstm':
        pol = score_policy(fit_lstm(data, cfg, bcfg['lstm'], device), m)
    elif method == 'eiie':
        pol = eiie_policy(fit_eiie(data, cfg, bcfg['eiie'], device), data, cfg, device)
    elif method == 'sood':
        pol = model_policy(fit_sood(data, cfg, bcfg['sood'], device, out_dir / 'sood'), device, record=True)
    elif method == 'random':
        rng = np.random.default_rng(cfg['seed'])
        pol = score_policy(lambda obs, env: rng.random(env.U), m)
    else:
        raise ValueError(f'未知对比方法 {method}')
    log.info(f'[{method}] 训练用时 {time.time() - t0:.0f}s')
    return pol
