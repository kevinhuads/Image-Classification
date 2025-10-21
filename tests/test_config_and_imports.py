import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

def test_src_on_path():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    assert str(SRC) in sys.path

def test_core_modules_import():
    """
    In a flat layout, verify the core modules import without executing heavy code.
    Only require that at least one core module exists to keep this smoke test light.
    """
    candidates = ["train", "eval", "infer", "data", "transforms"]
    imported = []
    for name in candidates:
        try:
            importlib.import_module(name)
            imported.append(name)
        except Exception:
            pass
    assert imported, f"None of the expected modules importable from src/: {candidates}"
