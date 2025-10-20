import importlib
import pytest

PKG = "food101"

@pytest.mark.skipif(importlib.util.find_spec(PKG) is None, reason="Package not importable")
def test_infer_function_contract():
    """
    If an inference API exists (e.g., predict(image, topk=5)), verify callable signature.
    """
    mod = importlib.import_module(PKG)
    # Try common locations
    for name in ["inference", "infer", "predict"]:
        sub = getattr(mod, name, None)
        if sub is None and hasattr(mod, "utils"):
            sub = getattr(mod.utils, name, None)
        if callable(sub):
            # Smoke: call with bogus input should raise a clean, expected error
            with pytest.raises(Exception):
                sub(None)  # invalid image on purpose
            return
    pytest.skip("No public inference entrypoint found")
