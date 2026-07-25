"""
Adds the project root and src/ to sys.path so that PyCharm's Run button
and plain `python scripts/train.py` both work without installing the package.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"

for p in (_PROJECT_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

