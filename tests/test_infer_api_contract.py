# tests/test_infer_api_contract.py
import importlib
import inspect
import os
import types
import pytest
from typing import Callable, Optional

PKG_NAME = "food101"


def _import_pkg_or_fail() -> types.ModuleType:
    try:
        return importlib.import_module(PKG_NAME)
    except Exception as exc:
        pytest.fail(f"Package '{PKG_NAME}' is not importable: {exc}")


def _find_callable_in_module(module: types.ModuleType, names) -> Optional[Callable]:
    for name in names:
        attr = getattr(module, name, None)
        if callable(attr):
            return attr
    return None


def test_infer_function_contract():
    """
    Locate an inference entrypoint. Acceptable forms:
      - a callable named one of infer/inference/predict at package root or in common submodules (utils,inference,api),
      - OR a CLI-style module 'food101.infer' (module presence is accepted).
    If a callable is found, calling it with an invalid input should raise an Exception (demonstrates input validation).
    """
    pkg = _import_pkg_or_fail()

    candidate_names = ("infer", "inference", "predict", "predict_image", "run_inference")
    candidates = []

    # root-level functions
    fn = _find_callable_in_module(pkg, candidate_names)
    if fn:
        candidates.append(fn)

    # common submodules
    for sub in ("utils", "inference", "api", "service"):
        try:
            submod = importlib.import_module(f"{PKG_NAME}.{sub}")
        except Exception:
            submod = None
        if submod:
            fn = _find_callable_in_module(submod, candidate_names)
            if fn:
                candidates.append(fn)

    # Accept CLI module 'food101.infer' as a valid inference exposure.
    try:
        infer_mod = importlib.import_module(f"{PKG_NAME}.infer")
    except Exception:
        infer_mod = None

    if not candidates and infer_mod is None:
        # As a fallback, check for top-level script file in package directory (src/food101/infer.py)
        pkg_path = os.path.dirname(pkg.__file__)
        script_path = os.path.join(pkg_path, "infer.py")
        if os.path.exists(script_path):
            infer_mod = True  # mark presence
    assert candidates or infer_mod, (
        "No public inference function or 'infer' CLI module found. Expected a callable named one of "
        f"{candidate_names} in package root or submodules, or a module 'food101.infer' present."
    )

    if candidates:
        # Use first candidate for a simple contract check: calling with invalid input should raise a controlled exception.
        test_fn = candidates[0]
        # Some inference functions accept variable args; we deliberately pass an invalid input to ensure predictable failure.
        with pytest.raises(Exception):
            test_fn(None)
