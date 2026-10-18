"""论文图表：净值曲线、持仓权重、训练曲线（静态 PNG）。"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# 分类色按固定顺序分配（同一策略在所有图中颜色一致）
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
OTHER, CASH = '#b4b2a9', '#dddcd5'

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.grid': True, 'axes.axisbelow': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.labelcolor': INK2, 'lines.linewidth': 1.6, 'savefig.dpi': 200,
})


def plot_nav(navs: dict, path, title):
    """净值曲线：图例 + 线尾直接标注（≤4 条）。"""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (name, nav) in enumerate(navs.items()):
        ax.plot(nav.index, nav.values, color=SERIES[i], label=name, zorder=3 if i == 0 else 2)
    ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)
    if len(navs) <= 4:                       # 线尾标注：按终值排序并保持最小间距，避免文字重叠
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.055
        ends = sorted(((nav.iloc[-1], name) for name, nav in navs.items()))
        ys = []
        for y, _ in ends:
            ys.append(max(y, ys[-1] + gap) if ys else y)
        shift = max(0.0, ys[-1] - (hi - gap / 2))
        for (y, name), y_txt in zip(ends, ys):
            ax.annotate(f'{name} {y:.2f}', (navs[name].index[-1], y), xytext=(1.01, y_txt - shift),
                        textcoords=('axes fraction', 'data'), va='center', fontsize=8.5, color=INK2,
                        annotation_clip=False)
    ax.set_title(title)
    ax.set_ylabel('净值')
    ax.legend(loc='upper left', ncol=len(navs), fontsize=9)
    ax.margins(x=0.0)
    fig.subplots_adjust(right=0.82)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_weights(weights: pd.DataFrame, path, title, top=7):
    """持仓权重堆叠面积图：平均权重前 7 的股票单独着色，其余归入“其他”，现金为浅灰。"""
    stocks = weights.drop(columns='cash')
    order = stocks.mean().sort_values(ascending=False).index
    keep = list(order[:top])
    df = stocks[keep].copy()
    if len(order) > top:
        df['其他'] = stocks[order[top:]].sum(axis=1)
    df['现金'] = weights['cash']
    colors = SERIES[:len(keep)] + ([OTHER] if len(order) > top else []) + [CASH]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.stackplot(df.index, df.T.values, labels=df.columns, colors=colors, edgecolor=SURFACE, linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel('权重')
    ax.grid(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8.5)
    ax.margins(x=0.0)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_training(logs: pd.DataFrame, path):
    """训练曲线：每个量单独一个子图（不使用双纵轴）。"""
    panels = [('reward', '缓存区平均奖励'), ('policy_loss', '策略损失 L_policy'), ('value_loss', '价值损失 L_value'),
              ('val_sharpe', '验证集夏普比率'), ('eps', '自适应裁剪参数 ε（批均值）'), ('lr', '自适应学习率 η_t'),
              ('approx_kl', 'KL(π_k‖π_θ)'), ('clip_frac', '裁剪比例'), ('explained_var', '价值网络解释方差')]
    panels = [(k, t) for k, t in panels if k in logs and logs[k].notna().any()]
    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
    for ax, (k, t) in zip(axes.flat, panels):
        s = logs[['step', k]].dropna()
        sparse = len(s) < 40                     # 稀疏序列（如验证集夏普）加标记点
        ax.plot(s['step'], s[k], color=SERIES[0], linewidth=1.2, marker='o' if sparse else None, markersize=4)
        ax.set_title(t, fontsize=10.5)
        ax.set_xlabel('训练步数 k')
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
