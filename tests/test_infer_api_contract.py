import importlib
import inspect
import os
import types
import pytest
from typing import Callable, Optional

def _find_callable_in_module(module: types.ModuleType, names) -> Optional[Callable]:
    for name in names:
        attr = getattr(module, name, None)
        if callable(attr):
            return attr
    return None

def test_infer_function_contract_flat_layout():
    """
    Acceptable forms in a flat layout:
      - a callable named one of infer/inference/predict at module root (in modules: infer, utils, inference, api, service),
      - OR a module 'infer' present (CLI-style: src/infer.py).
    If a callable is found, calling it with invalid input should raise an Exception (input validation).
    """
    candidate_names = ("infer", "inference", "predict", "predict_image", "run_inference")
    candidates = []

    # root-level modules to check
    for mod_name in ("infer", "utils", "inference", "api", "service"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        fn = _find_callable_in_module(mod, candidate_names)
        if fn:
            candidates.append(fn)

    # Accept presence of an 'infer' module even without a callable (CLI-only).
    try:
        infer_mod = importlib.import_module("infer")
    except Exception:
        infer_mod = None

    # As a fallback, check for a top-level script file src/infer.py
    if infer_mod is None:
        script_path = os.path.join("src", "infer.py")
        if os.path.exists(script_path):
            infer_mod = True

    assert candidates or infer_mod, (
        "No public inference function or 'infer.py' module found. "
        "Expected a callable named one of "
        f"{candidate_names} in modules infer/utils/inference/api/service, or a module 'infer' present."
    )

    if candidates:
        test_fn = candidates[0]
        with pytest.raises(Exception):
            test_fn(None)
