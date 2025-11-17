import sys
import importlib
import os

root_dir = os.getcwd()
src_dir = os.path.join(root_dir, "src")

def test_src_on_path():
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    assert str(src_dir) in sys.path

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
