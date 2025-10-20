import os
import sys
import importlib
from pathlib import Path

PKG_NAME = "food101"  # adjust if package name differs
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

def test_src_on_path():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    assert str(SRC) in sys.path

def test_pkg_imports():
    mod = importlib.import_module(PKG_NAME)
    assert hasattr(mod, "__version__") or True  # version optional
