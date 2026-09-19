"""pytest 根配置：确保项目根目录在 sys.path 中，让 tests/ 能 import app / agents / graph。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
