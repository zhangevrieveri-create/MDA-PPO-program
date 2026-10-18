"""步骤4：计算表2技术指标，融合新浪新闻情绪与米筐情绪指数，构建 RL 数据集 (npz)。"""
import argparse

import _bootstrap  # noqa: F401
from rlinvest.features.dataset import build_dataset
from rlinvest.utils import get_logger, load_config

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    get_logger(log_file=cfg['paths']['output'] / 'logs' / '04_build_dataset.log')
    build_dataset(cfg)
