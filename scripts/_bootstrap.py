"""让 scripts/ 下的脚本无需安装即可 import rlinvest。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
