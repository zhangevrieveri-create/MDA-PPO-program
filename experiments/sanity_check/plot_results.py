"""绘制强信号合成数据上各组的验证集夏普曲线，并标出等权与预言机参照线。

python experiments/sanity_check/plot_results.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlinvest.plotting import INK2, MUTED, SERIES, plt  # noqa: E402

RES = Path(__file__).resolve().parent / 'results'
SHOW = {'A': '改动前代码', 'C': '+σ 上限', 'E': '+σ 上限 + 等权控制变量', 'G': '+ logit 约束（当前默认）'}


def main():
    curves = pd.read_csv(RES / 'val_sharpe_curves.csv')
    curves = curves[curves['实验'] == '强信号']
    oracle = pd.read_csv(RES / 'oracle_strong_signal.csv')
    val = oracle[oracle['数据段'] == 'val'].set_index('策略')['夏普比率']

    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, (run, label) in enumerate(SHOW.items()):
        c = curves[curves['组'] == run].sort_values('更新轮次')
        ax.plot(c['更新轮次'], c['验证集夏普'], color=SERIES[i], label=label, marker='o', markersize=3)
    xmax = curves['更新轮次'].max()
    for name, text, style in [('等权', '等权', '--'), ('预言机 k=1.0', '预言机（已知信号）', ':')]:
        ax.axhline(val[name], color=MUTED, linewidth=1.0, linestyle=style, zorder=1)
        ax.annotate(f'{text} {val[name]:.2f}', (xmax, val[name]), xytext=(0, 4), textcoords='offset points',
                    ha='right', va='bottom', fontsize=8.5, color=INK2)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)
    ax.set_title('强信号合成数据：验证集夏普比率随训练的变化')
    ax.set_xlabel('更新轮次（每轮 100 个交易步）')
    ax.set_ylabel('验证集夏普比率')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.16), ncol=4, fontsize=9)
    fig.subplots_adjust(right=0.84)
    fig.savefig(RES / 'val_sharpe_strong_signal.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
