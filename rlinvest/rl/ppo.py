"""PPO-Clip 训练（论文第三部分（五）与算法1）。

for k = 1..K:
  1-2. 状态特征提取，策略网络/价值网络前向传播
  3.   课程噪声注入与动作采样：σ_noise,k = σ0·exp(−β·k)，a_t = μ_t + (σ_t + σ_noise,k) ⊙ ξ_t，ξ_t ~ N(0, I)
       环境内：截断 -> 取得分最高的 m 只股票 -> 与现金一起归一化；对数概率按截断前的 a_t 计算
  4.   缓存区 D 按时间顺序存放当前策略产生的连续轨迹；|D| = C 时：
         用更新前的价值网络 V_φold 在 D 上从后向前递推一次 GAE（d_t 为轨迹结束标记）：
           δ_t = R_t + γ(1−d_t)V_φold(s_{t+1}) − V_φold(s_t)，M̂_t = δ_t + γλ(1−d_t)M̂_{t+1}
           V_t^target = V_φold(s_t) + M̂_t
         for e = 1..E：把 D 中 C 个样本随机打乱，依次取大小为 B 的小批量更新
           L_policy = −mean(min(r_i M̂_i, clip(r_i, 1−ε_i, 1+ε_i) M̂_i))，r_i = π_θ(a_i|s_i)/π_θold(a_i|s_i)
           L_value  = mean((V_φ(s_i) − V_i^target)²)，L_total = L_policy + 0.5·L_value
         θ_old ← θ，φ_old ← φ，清空 D
  5.   k mod 100 = 0：自适应学习率 η = η0·exp(−α·ΔL_policy/L_policy)
       裁剪参数按样本所处交易日的市场波动率：ε_t = ε0·(1 + 0.5·σ_t / σ̄_{t,W})
新旧策略概率均按含课程噪声的分布 N(μ, diag((σ+σ_noise)²)) 计算。
"""
import copy
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..backtest import metrics, model_policy, run_backtest
from ..utils import get_logger
from .env import PortfolioEnv, segment
from .model import ActorCritic, calibrate_bn, param_table, xavier_init

log = get_logger()
LOG_RATIO_CLIP = 20.0      # [实现] log 概率比的截断范围，防止数值溢出


class RunningStd:
    """[实现] 折扣回报 G_t = γG_{t-1} + r_t 的滑动标准差，用于奖励缩放（不改变最优策略）。"""

    def __init__(self, gamma):
        self.gamma, self.G, self.n, self.mean, self.M2 = gamma, 0.0, 0, 0.0, 0.0

    def update(self, r, end):
        self.G = self.gamma * self.G + r
        self.n += 1
        d = self.G - self.mean
        self.mean += d / self.n
        self.M2 += d * (self.G - self.mean)
        if end:
            self.G = 0.0

    @property
    def std(self):
        return math.sqrt(self.M2 / self.n) if self.n > 1 and self.M2 > 0 else 1.0


class RolloutBuffer:
    """采样缓存区 D：按时间顺序存放当前策略的连续轨迹。状态不重复保存，只记交易日下标 t，用时从数据集重建。"""

    def __init__(self, C, act_dim):
        self.C, self.size = C, 0
        self.t = np.zeros(C, np.int64)
        self.t_next = np.zeros(C, np.int64)
        self.act = np.zeros((C, act_dim), np.float32)
        self.logp, self.noise, self.rew, self.val = (np.zeros(C, np.float32) for _ in range(4))
        self.done = np.zeros(C, bool)

    def add(self, t, act, logp, noise, rew, val, done, t_next):
        i = self.size
        self.t[i], self.act[i], self.logp[i], self.noise[i] = t, act, logp, noise
        self.rew[i], self.val[i], self.done[i], self.t_next[i] = rew, val, done, t_next
        self.size += 1

    def full(self):
        return self.size >= self.C

    def clear(self):
        self.size = 0


def gae(rew, val, next_val, done, gamma, lam):
    """在连续轨迹上从后向前递推：δ_t = R_t + γ(1−d_t)V(s_{t+1}) − V(s_t)，M̂_t = δ_t + γλ(1−d_t)M̂_{t+1}。"""
    adv, g = np.zeros_like(rew), 0.0
    for i in range(len(rew) - 1, -1, -1):
        nd = 0.0 if done[i] else 1.0
        delta = rew[i] + gamma * nd * next_val[i] - val[i]
        g = delta + gamma * lam * nd * g
        adv[i] = g
    return adv


class PPOTrainer:
    def __init__(self, data, cfg, device, out_dir, model_fn=None):
        """model_fn(F, U, cfg) -> ActorCritic：对比方法用来替换网络主干，默认为论文的 MDA-PPO 网络。"""
        self.data, self.c, self.dev, self.out = data, cfg, device, out_dir
        self.env = PortfolioEnv(data, cfg, *segment(data, 'train', cfg['T']), random_start=True)
        self.U, self.F = self.env.U, self.env.n_features
        self.has_val = data['val_idx'] < data['test_idx']
        self.model = (model_fn or ActorCritic)(self.F, self.U, cfg).to(device)
        xavier_init(self.model, cfg.get('sigma_init', 0.1))
        lo, hi = self.env.lo, self.env.hi
        ts = np.random.randint(lo, hi, size=min(64, hi - lo))
        calibrate_bn(self.model, (self.obs_batch(ts[i:i + 8]) for i in range(0, len(ts), 8)), device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.buf = RolloutBuffer(cfg['C'], self.U + 1)
        self.rnorm = RunningStd(cfg['gamma'])
        assert cfg['B'] <= cfg['C'], '小批量大小 B 不能大于缓存区容量 C'

        # 市场波动率 σ_t（基准指数日对数收益的滚动标准差）与逐日裁剪参数 ε_t = ε0·(1 + 0.5·σ_t/σ̄_{t,W})
        r = np.r_[0.0, np.diff(np.log(data['bench']))]
        sig = pd.Series(r).rolling(cfg['vol_win']).std().bfill()
        sig_w = sig.shift(1).rolling(cfg['W'], min_periods=1).mean().bfill()
        eps = cfg['eps0'] * (1 + 0.5 * sig / sig_w)
        if cfg.get('eps_bounds'):
            eps = eps.clip(*cfg['eps_bounds'])
        self.eps_t = eps.to_numpy(np.float32)

        self.table3 = param_table(self.model) if model_fn is None else None
        if self.table3 is not None:
            log.info('表3 核心架构参数数量：\n' + self.table3.to_string(index=False))

    # ---------------- 工具 ----------------
    def obs_batch(self, ts):
        X, T = self.data['X'], self.c['T']
        return np.stack([X[t - T + 1:t + 1].transpose(1, 0, 2) for t in ts])

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

    def _train_mode(self):
        self.model.train()
        self.model.enc.bn.eval()              # BatchNorm 统计量保持冻结

    def noise_at(self, k, n_updates):
        clock = k if self.c.get('noise_clock', 'step') == 'step' else n_updates
        return self.c['sigma0'] * math.exp(-self.c['beta'] * clock)

    # ---------------- 4. PPO-Clip 优化 ----------------
    def update(self):
        c, b, n = self.c, self.buf, self.buf.size
        scale = self.rnorm.std if c['reward_norm'] else 1.0
        rew = b.rew[:n] / scale
        val = b.val[:n]                                                # V_φold(s_t)，采样时即当前（更新前）的网络
        next_val = np.r_[val[1:], 0.0].astype(np.float32)              # 连续轨迹中 s_{t+1} 就是下一条样本
        last = n - 1
        if not b.done[last]:                                           # 最后一条样本的 s_{t+1} 不在缓存区内，单独估值
            with torch.no_grad():
                next_val[last] = self.model(self._t(self.obs_batch([b.t_next[last]])))[2].item()
        adv = gae(rew, val, next_val, b.done[:n], c['gamma'], c['lam'])
        ret = self._t(adv + val)                                       # V^target = V_φold(s) + M̂
        ev = 1 - np.var(adv) / (np.var(adv + val) + 1e-8)              # 价值网络解释方差
        if c['adv_norm']:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = self._t(adv)
        act, logp_old, noise = self._t(b.act[:n]), self._t(b.logp[:n]), self._t(b.noise[:n])[:, None]
        eps = self._t(self.eps_t[b.t[:n]])
        B, mb = c['B'], max(1, c.get('micro_batch') or c['B'])

        stats = []
        for _ in range(c['E']):
            perm = np.random.permutation(n)
            for s in range(0, n, B):
                idx = perm[s:s + B]
                self.opt.zero_grad()
                acc = np.zeros(4)
                for m0 in range(0, len(idx), mb):                     # 梯度累积：微批量之和等于整个小批量的均值
                    ix = idx[m0:m0 + mb]
                    mu, sigma, v = self.model(self._t(self.obs_batch(b.t[ix])))
                    lp = Normal(mu, sigma + noise[ix]).log_prob(act[ix]).sum(-1)
                    log_r = (lp - logp_old[ix]).clamp(-LOG_RATIO_CLIP, LOG_RATIO_CLIP)
                    rho, a, e = log_r.exp(), adv[ix], eps[ix]
                    l_pol = -torch.min(rho * a, torch.clamp(rho, 1 - e, 1 + e) * a).mean()
                    l_val = ((v - ret[ix]) ** 2).mean()
                    loss = (l_pol + c['value_coef'] * l_val) * len(ix) / len(idx)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('损失出现非有限值，请检查数据与学习率')
                    loss.backward()
                    with torch.no_grad():
                        kl = ((rho - 1) - log_r).mean().item()
                        clipf = ((rho - 1).abs() > e).float().mean().item()
                    acc += np.array([l_pol.item(), l_val.item(), kl, clipf]) * len(ix) / len(idx)
                if c.get('max_grad_norm'):
                    nn.utils.clip_grad_norm_(self.model.parameters(), c['max_grad_norm'])
                self.opt.step()
                stats.append(acc)
        b.clear()                                                      # θ_old ← θ，φ_old ← φ，D ← ∅
        out = dict(zip(['policy_loss', 'value_loss', 'approx_kl', 'clip_frac'], np.mean(stats, 0)))
        out.update(explained_var=float(ev), eps=float(eps.mean()))
        return out

    # ---------------- 5. 自适应学习率 ----------------
    def adapt_lr(self, L_now, L_prev, L_scale):
        """η = η0·exp(−α·ΔL_policy / L_policy)；L 取策略损失的指数滑动平均，分母取 |L| 的滑动平均。"""
        c = self.c
        lr = c['lr'] * math.exp(-c['alpha'] * (L_now - L_prev) / (L_scale + 1e-8))
        if c.get('lr_bounds'):
            lr = float(np.clip(lr, *c['lr_bounds']))
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr

    # ---------------- 验证 ----------------
    def evaluate(self, name='val'):
        self.model.eval()
        nav, *_ = run_backtest(self.data, self.c, model_policy(self.model, self.dev), name)
        self._train_mode()
        return metrics(nav, self.c['rf'])

    # ---------------- 主训练循环 ----------------
    def train(self):
        c = self.c
        self._train_mode()
        obs = self.env.reset()
        lr, n_updates, last = c['lr'], 0, {}
        L_ema = L_abs = L_prev = None
        ep_values, logs, rewards = [], [], []
        best, t0 = -np.inf, time.time()

        for k in range(1, c['K'] + 1):
            # 1-2. 前向传播
            with torch.no_grad():
                mu, sigma, v = self.model(self._t(obs)[None])
            # 3. 课程噪声注入与动作采样
            noise = self.noise_at(k, n_updates)
            mu, sigma = mu[0].cpu().numpy(), sigma[0].cpu().numpy()
            std = sigma + noise
            a = (mu + std * np.random.randn(self.U + 1)).astype(np.float32)
            logp = float(Normal(torch.as_tensor(mu), torch.as_tensor(std)).log_prob(torch.as_tensor(a)).sum())
            t = self.env.t
            obs2, r, done, info = self.env.step(self.env.to_weights(a))
            self.rnorm.update(r, done)
            rewards.append(r)
            self.buf.add(t, a, logp, noise, r, v.item(), done, self.env.t)
            if done:
                ep_values.append(info['V'])
                obs = self.env.reset()
            else:
                obs = obs2

            # 4. PPO-Clip 优化：缓存区存满 C 步后更新一次并清空
            if self.buf.full():
                last = self.update()
                n_updates += 1
                L = last['policy_loss']
                L_ema = L if L_ema is None else 0.9 * L_ema + 0.1 * L
                L_abs = abs(L) if L_abs is None else 0.9 * L_abs + 0.1 * abs(L)
                if self.has_val and n_updates % c['eval_interval'] == 0:
                    m = self.evaluate('val')
                    last.update(val_sharpe=m['夏普比率'], val_return=m['年化收益'])
                    if m['夏普比率'] > best:
                        best = m['夏普比率']
                        self.save(self.out / 'best.pt', extra=dict(update=n_updates, step=k, val=m))
                    log.info(f'[验证] update {n_updates} 夏普={m["夏普比率"]:.3f} 年化={m["年化收益"]:.2%} '
                             f'回撤={m["最大回撤"]:.2%}（最优 {best:.3f}）')

            # 5. 自适应学习率（每 adapt_interval 步）
            if k % c['adapt_interval'] == 0:
                if L_ema is not None and L_prev is not None:
                    lr = self.adapt_lr(L_ema, L_prev, L_abs)
                L_prev = L_ema
                logs.append(dict(step=k, update=n_updates, lr=lr, noise=noise, reward=float(np.mean(rewards)),
                                 episode_value=np.mean(ep_values[-10:]) if ep_values else np.nan, **last))
                rewards, last = [], {}
                if k % (c['adapt_interval'] * 10) == 0:
                    L = logs[-1]
                    g = lambda key, f='.4f': format(L.get(key, np.nan), f)
                    log.info(f'step {k}/{c["K"]} update={n_updates} reward={g("reward", "+.4f")} '
                             f'lr={lr:.2e} noise={noise:.3f} 近10条轨迹终值={g("episode_value")} '
                             f'用时={time.time() - t0:.0f}s')

        self.save(self.out / 'last.pt', extra=dict(update=n_updates, step=k))
        if not (self.out / 'best.pt').exists():
            # 论文没有验证集：模型与超参数只在训练集上确定，直接使用训练结束时的模型
            shutil.copy(self.out / 'last.pt', self.out / 'best.pt')
        return pd.DataFrame(logs)

    def save(self, path, extra=None):
        torch.save({'model': self.model.state_dict(), 'cfg': copy.deepcopy(self.c), 'F': self.F, 'U': self.U,
                    'codes': self.data['codes'], 'features': self.data['features'], **(extra or {})}, path)


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(ck['F'], ck['U'], ck['cfg']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, ck
