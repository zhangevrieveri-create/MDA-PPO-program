"""通用工具：配置加载、日志、随机种子。"""
import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
_FMT = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')


def _merge(base, over):
    for k, v in over.items():
        base[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def _read_yaml(path: Path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if 'base' in cfg:
        cfg = _merge(_read_yaml(path.parent / cfg.pop('base')), cfg)
    return cfg


def load_config(path=None):
    """读取 yaml 配置；支持 `base: default.yaml` 继承，只需写出要覆盖的字段。"""
    cfg = _read_yaml(Path(path) if path else ROOT / 'configs' / 'default.yaml')
    for k, v in cfg['paths'].items():
        p = Path(v)
        cfg['paths'][k] = p if p.is_absolute() else ROOT / p
        cfg['paths'][k].mkdir(parents=True, exist_ok=True)
    return cfg


def get_logger(name='rlinvest', log_file=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(_FMT)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(_FMT)
        logger.addHandler(fh)
    return logger


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(tf32=True):
    """优先使用 GPU；Ampere 及以上显卡开启 TF32 矩阵乘法加速。"""
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        return torch.device('cuda')
    return torch.device('cpu')
