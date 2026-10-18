"""策略网络与价值网络（论文第三部分（三）、算法1 步骤1-2；参数量与表3一致）。

共享主干：
  Ŝ_t       = BatchNorm(S_t)
  H^LSTM    = Bi-LSTM(Ŝ_t)                          ∈ R^{U×T×256}
  H^trans   = Transformer-Encoder(H^LSTM)           ∈ R^{U×T×256}
  C^time    = MHSA(H^trans)   （每只股票沿时间维度） ∈ R^{U×T×256}
  C^asset   = MHSA(H^trans⊤)⊤（每个交易日沿股票维度）∈ R^{U×T×256}
  C_t       = [C^time, C^asset]                     ∈ R^{U×T×512}
  h_{t,i}   = (1/T) Σ_τ C_{τ,i}                     ∈ R^{512}，i = 1,…,U
  H_global  = (1/U) Σ_i h_{t,i}                     ∈ R^{512}
输出（各股票共享同一组输出层参数）：
  μ_{t,i} = sigmoid(w_a2ᵀ GELU(W_a1 h_{t,i} + b_a1) + b_a2)，σ_{t,i} = softplus(w_a3ᵀ GELU(W_a1 h_{t,i} + b_a1) + b_a3)
  μ_{t,0}、σ_{t,0}（现金）：同一隐藏层作用于 H_global，输出层用现金专用的一行参数 (w_a2,0, b_a2,0)、(w_a3,0, b_a3,0)
  H_critic = GELU(W_c1 H_global + b_c1)，V_φ(s_t) = W_c2 H_critic + b_c2
代码中动作向量的最后一维为现金（论文记为下标 0）。
F=25 时总参数量 1,477,253；共享输出层使参数量与股票数 U 无关。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class Encoder(nn.Module):
    def __init__(self, F, c):
        super().__init__()
        d = 2 * c['lstm_hidden']
        self.bn = nn.BatchNorm1d(F, affine=False)          # 状态标准化，无可训练参数（表3不计）
        self.lstm = nn.LSTM(F, c['lstm_hidden'], c['lstm_layers'], batch_first=True, bidirectional=True)
        layer = nn.TransformerEncoderLayer(d, c['n_heads'], c['ff_dim'], dropout=c['dropout'],
                                           activation='gelu', batch_first=True)
        self.trans = nn.TransformerEncoder(layer, c['n_layers'], enable_nested_tensor=False)
        mode = c.get('dual_attention', 'both')                # 消融：both / time / asset / none
        mha = lambda: nn.MultiheadAttention(d, c['n_heads'], dropout=c['dropout'], batch_first=True)
        self.time_att = mha() if mode in ('both', 'time') else None
        self.asset_att = mha() if mode in ('both', 'asset') else None
        self.out_dim = 2 * d

    @staticmethod
    def _att(m, h):
        # need_weights=False 走 scaled_dot_product_attention 的省显存实现（股票维度 U≈3700 时必需）
        return m(h, h, h, need_weights=False)[0]

    def forward(self, x):                                      # x: (N,U,T,F)
        N, U, T, F = x.shape
        x = self.bn(x.reshape(-1, F)).reshape(N * U, T, F)
        h, _ = self.lstm(x)                                    # (N·U,T,d)
        h = self.trans(h)
        d = h.shape[-1]
        ct = self._att(self.time_att, h) if self.time_att is not None else h            # C^time
        ha = h.reshape(N, U, T, d).transpose(1, 2).reshape(N * T, U, d)
        ca = self._att(self.asset_att, ha) if self.asset_att is not None else ha         # C^asset
        ca = ca.reshape(N, T, U, d).transpose(1, 2)
        C = torch.cat([ct.reshape(N, U, T, d), ca], -1)        # (N,U,T,2d)
        return C.mean(2)                                       # h_{t,i}: (N,U,512)


class MLPEncoder(nn.Module):
    """对比方法 Sood 等（2023）的主干：每只股票把 T×F 窗口展平后经两层全连接网络（各股票共享参数）。"""

    def __init__(self, F, c, hidden=64):
        super().__init__()
        self.bn = nn.BatchNorm1d(F, affine=False)
        self.net = nn.Sequential(nn.Linear(c['T'] * F, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.out_dim = hidden

    def forward(self, x):                                      # x: (N,U,T,F)
        N, U, T, F = x.shape
        x = self.bn(x.reshape(-1, F)).reshape(N, U, T * F)
        return self.net(x)                                     # (N,U,hidden)


class ActorCritic(nn.Module):
    def __init__(self, F, U, c, encoder=None):
        super().__init__()
        self.enc = encoder if encoder is not None else Encoder(F, c)   # 对比方法可替换主干，输出层结构不变
        g, h = self.enc.out_dim, c['mlp_hidden']
        self.actor_hidden = nn.Linear(g, h)                    # W_a1, b_a1
        self.mu_head = nn.Linear(h, 2)                         # 第0行：各股票共享 (w_a2, b_a2)；第1行：现金 (w_a2,0, b_a2,0)
        self.sigma_head = nn.Linear(h, 2)                      # 同上，(w_a3, b_a3) 与 (w_a3,0, b_a3,0)
        self.critic_hidden = nn.Linear(g, h)                   # W_c1, b_c1
        self.value_head = nn.Linear(h, 1)                      # W_c2, b_c2

    @staticmethod
    def _out(layer, zs, zc):
        stock = zs @ layer.weight[0] + layer.bias[0]           # (N,U)
        cash = zc @ layer.weight[1] + layer.bias[1]            # (N,)
        return torch.cat([stock, cash[:, None]], -1)           # (N,U+1)，最后一维为现金

    def forward(self, x):
        hu = self.enc(x)                                       # h_{t,i}: (N,U,512)
        g = hu.mean(1)                                         # H_global: (N,512)
        zs, zc = Fn.gelu(self.actor_hidden(hu)), Fn.gelu(self.actor_hidden(g))
        mu = torch.sigmoid(self._out(self.mu_head, zs, zc))   # μ ∈ [0,1]^{U+1}
        sigma = Fn.softplus(self._out(self.sigma_head, zs, zc)) + 1e-4
        v = self.value_head(Fn.gelu(self.critic_hidden(g))).squeeze(-1)
        return mu, sigma, v


def xavier_init(model, sigma_init=0.1, out_gain=0.01):
    """算法1：所有权重矩阵 Xavier 均匀初始化，偏置置 0。
    [实现] 策略输出层用小增益（初始 μ≈0.5），σ 的偏置使初始标准差约为 sigma_init。"""
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            nn.init.zeros_(p)
    for layer in (model.mu_head, model.sigma_head):
        nn.init.xavier_uniform_(layer.weight, gain=out_gain)
    nn.init.constant_(model.sigma_head.bias, float(np.log(np.expm1(sigma_init))))


@torch.no_grad()
def calibrate_bn(model, obs_batches, device):
    """在训练段随机状态上估计 BatchNorm 统计量后冻结，保证 PPO 新旧策略概率计算一致。"""
    bn = model.enc.bn
    bn.reset_running_stats()
    bn.momentum = None                     # 累计平均
    bn.train()
    for x in obs_batches:
        xx = torch.as_tensor(x, device=device)
        bn(xx.reshape(-1, xx.shape[-1]))
    bn.eval()


def param_table(model: ActorCritic):
    """表3：核心架构参数数量。"""
    cnt = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
    e = model.enc
    lstm_b = sum(p.numel() for n, p in e.lstm.named_parameters() if n.endswith('_reverse'))
    lstm_f = cnt(e.lstm) - lstm_b
    rows = [('特征提取', 'Bi-LSTM', '前向LSTM', lstm_f), ('特征提取', 'Bi-LSTM', '后向LSTM', lstm_b)]
    for i, layer in enumerate(e.trans.layers, 1):
        rows += [('特征提取', f'Transformer编码器层{i}', '多头注意力', cnt(layer.self_attn)),
                 ('特征提取', f'Transformer编码器层{i}', '前馈网络', cnt(layer.linear1) + cnt(layer.linear2)),
                 ('特征提取', f'Transformer编码器层{i}', '层归一化×2', cnt(layer.norm1) + cnt(layer.norm2))]
    rows += [('双重注意力', '双重注意力', '时间注意力', cnt(e.time_att)),
             ('双重注意力', '双重注意力', '标的注意力', cnt(e.asset_att)),
             ('决策输出层', '策略网络', '隐藏层', cnt(model.actor_hidden)),
             ('决策输出层', '策略网络', '均值输出', cnt(model.mu_head)),
             ('决策输出层', '策略网络', '标准差输出', cnt(model.sigma_head)),
             ('决策输出层', '价值网络', '隐藏层', cnt(model.critic_hidden)),
             ('决策输出层', '价值网络', '价值输出', cnt(model.value_head))]
    df = pd.DataFrame(rows, columns=['主架构', '模块', '子架构', '参数数量'])
    total = cnt(model)
    assert df['参数数量'].sum() == total
    df['百分比'] = (df['参数数量'] / total).map('{:.2%}'.format)
    df.loc[len(df)] = ['模型总计', '', '', total, '100.00%']
    return df
