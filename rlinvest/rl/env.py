"""投资组合交易环境（论文第三部分（四））。

状态 s_t   ：过去 T 日的特征窗口，形状 (U,T,F)
动作 a_t   ：U+1 维，前 U 维为各股票得分，最后一维为现金头寸（论文记为下标 0）
选股与配权：Ã_t = clip(a_t, 0, 1)；S_t = Top_m(Ã_{t,1..U})（只在已上市股票中选）；
            A_{t,i} = Ã_{t,i} / (Ã_{t,0} + Σ_{j∈S_t} Ã_{t,j} + 10⁻⁸)，i ∈ S_t ∪ {0}，其余股票权重为 0
调仓      ：t 日收盘按 A_t 调仓（停牌不可交易、涨停不可买、跌停不可卖），获得 t→t+1 的收益
奖励      ：R_t = w1·R_short + w2·DSR + w3·Penalty
            R_short = A_t·(P_{t+1}−P_t)/P_t − τ·‖A_t − A_{t−1}‖₁   （扣除按换手量计提的交易成本）
            DSR     = [B_{t−1}ΔM − ½M_{t−1}ΔB] / (B_{t−1} − M²_{t−1})^{3/2}（Moody & Saffell, 2001）
            Penalty = −(2x_t − 1)²，x_t = Σ_{i∈S_t} A_{t,i} = 1 − A_{t,0}
"""
import math
import random

import numpy as np


def segment(data, name, T):
    """返回数据段在时间轴上的 [lo, hi]：t∈[lo, hi) 上决策，使用 close[lo..hi]。"""
    D, v, s = len(data['close']), data['val_idx'], data['test_idx']
    seg = {'train': (T - 1, v - 1), 'val': (v - 1, s - 1), 'test': (s - 1, D - 1)}[name]
    if seg[1] <= seg[0]:
        raise ValueError(f'数据集中没有 {name} 段（论文默认只有训练集与测试集）')
    return seg


def select_and_normalize(a, m, selectable, eps=1e-8, cash=True, weighting='score'):
    """论文式：截断、取得分最高的 m 只股票、与现金一起归一化。a 的最后一维为现金。
    表6 消融：cash=False 去掉无风险资产（w/o RF，现金得分视为 0）；weighting='equal' 去掉动态权重
    （w/o DW，入选股票等分股票仓位）。"""
    A = np.clip(a, 0.0, 1.0)
    U = len(A) - 1
    if not cash:
        A[U] = 0.0
    score = np.where(selectable, A[:U], -np.inf)
    n_ok = int(np.isfinite(score).sum())
    k = min(m, n_ok)
    w = np.zeros_like(A)
    if k > 0:
        sel = np.argpartition(-score, k - 1)[:k]
        denom = A[U] + A[sel].sum() + eps
        if denom > 1e-6:
            w[sel] = A[sel].sum() / k / denom if weighting == 'equal' else A[sel] / denom
        elif not cash:
            w[sel] = 1.0 / k               # 得分全为 0 且不允许持有现金时等权
    w[U] = 1.0 - w[:U].sum()               # 现金头寸（含分母中 ε 带来的微小余量）
    return w


class PortfolioEnv:
    def __init__(self, data, cfg, lo, hi, random_start=False):
        self.X, self.P = data['X'], data['close']
        self.listed = data['listed']
        self.buy_ok, self.sell_ok = data['buy_ok'], data['sell_ok']
        self.c, self.lo, self.hi, self.random_start = cfg, lo, hi, random_start
        self.U, self.T = self.P.shape[1], cfg['T']
        self.n_features = self.X.shape[2]
        assert lo >= self.T - 1, '数据段起点不足一个历史窗口'

    def reset(self):
        L = self.c.get('ep_len')
        if self.random_start and L:
            self.t = random.randint(self.lo, max(self.lo, self.hi - L))
            self.end = min(self.hi, self.t + L)
        else:
            self.t, self.end = self.lo, self.hi
        self.w = np.zeros(self.U + 1)
        self.w[-1] = 1.0                       # 初始全部为现金
        self.V = 1.0
        self.M, self.Bm = 0.0, self.c.get('dsr_init_B', 1e-4)   # DSR 一阶矩 M_0、二阶矩 B_0
        return self.obs()

    def obs(self, t=None):
        t = self.t if t is None else t
        return self.X[t - self.T + 1:t + 1].transpose(1, 0, 2)    # (U,T,F)

    def selectable(self):
        return self.listed[self.t]

    def to_weights(self, a):
        return select_and_normalize(a, self.c['m'], self.selectable(), cash=self.c.get('cash', True),
                                    weighting=self.c.get('weighting', 'score'))

    def apply_limits(self, w):
        """停牌不可交易、涨停不可买入、跌停不可卖出；受限部分维持原持仓，超出部分从增仓中按比例扣减。"""
        if not self.c['trade_limits']:
            return w
        old, new = self.w[:-1], w[:-1].copy()
        blocked = ((new > old) & ~self.buy_ok[self.t]) | ((new < old) & ~self.sell_ok[self.t])
        new[blocked] = old[blocked]
        excess = new.sum() - 1.0
        inc = np.clip(new - old, 0, None)
        if excess > 1e-10 and inc.sum() > 0:
            new -= inc * min(1.0, excess / inc.sum())
        new = np.clip(new, 0, None)
        return np.append(new, max(0.0, 1.0 - new.sum()))

    def step(self, w):
        """w：目标权重（已选股、归一化），最后一维为现金。"""
        c = self.c
        w = self.apply_limits(w)
        r = np.nan_to_num(self.P[self.t + 1] / self.P[self.t] - 1, nan=0.0, posinf=0.0, neginf=0.0)
        rf_d = c['rf'] / 252
        turnover = float(np.abs(w[:-1] - self.w[:-1]).sum())
        cost = c['tau'] * turnover
        R_short = float((w[:-1] * r).sum() + w[-1] * rf_d) - cost
        V0 = self.V
        self.V = V0 * (1 + R_short)

        # 差分夏普比率：在 η=0 处对夏普比率泰勒展开（矩估计用更新前的 M_{t−1}、B_{t−1}）
        dM, dB = R_short - self.M, R_short ** 2 - self.Bm
        var = self.Bm - self.M ** 2
        dsr = (self.Bm * dM - 0.5 * self.M * dB) / var ** 1.5 if var > 1e-12 else 0.0
        if c.get('dsr_clip'):
            dsr = float(np.clip(dsr, -c['dsr_clip'], c['dsr_clip']))
        self.M += c['dsr_eta'] * dM
        self.Bm += c['dsr_eta'] * dB

        x = 1.0 - w[-1]                                       # 当日股票总仓位
        penalty = -(2 * x - 1) ** 2
        reward = c['w1'] * R_short + c['w2'] * dsr + c['w3'] * penalty

        growth = np.append(1 + r, 1 + rf_d)
        g = float((w * growth).sum())
        self.w = w * growth / g if g > 0 else w               # 价格变动后的实际持仓（下一日的 A_{t−1}）
        self.t += 1
        info = dict(V=self.V, ret=R_short, cost=cost, turnover=turnover, weights=w, x=x, dsr=dsr,
                    penalty=penalty)
        return self.obs(), reward, self.t >= self.end, info
