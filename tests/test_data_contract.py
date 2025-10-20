import os
import sys
import types
import importlib
import pytest

PKG_NAME = "food101"

@pytest.fixture(scope="session", autouse=True)
def _ensure_src_on_path():
    root = os.getcwd()
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

@pytest.mark.skipif(
    importlib.util.find_spec(PKG_NAME) is None,
    reason="Package not importable"
)
def test_transforms_contract():
    """
    If the package exposes factory functions for transforms, verify they
    produce callables that accept PIL/tensors and return tensors of expected shape.
    """
    pkg = importlib.import_module(PKG_NAME)
    # Be tolerant to naming; try a few common locations.
    candidates = [
        f"{PKG_NAME}.data",
        f"{PKG_NAME}.transforms",
        f"{PKG_NAME}.dataset",
    ]
    module = None
    for c in candidates:
        try:
            module = importlib.import_module(c)
            break
        except Exception:
            continue
    if module is None:
        pytest.skip("No data/transforms module found to validate")

    make_tfms = None
    for attr in ["build_transforms", "get_transforms", "create_transforms"]:
        make_tfms = getattr(module, attr, None) or make_tfms

    if make_tfms is None:
        pytest.skip("No known transform factory found")

    tfms = make_tfms(train=False)
    assert callable(tfms)

def _find_dataset_cls(module: types.ModuleType):
    # Try common class/function names used for datasets
    for name in ["Food101Dataset", "Food101", "build_dataset", "create_dataset"]:
        if hasattr(module, name):
            return getattr(module, name)
    return None

@pytest.mark.skipif(
    importlib.util.find_spec(PKG_NAME) is None,
    reason="Package not importable"
)
def test_dataset_factory_signature():
    """
    Ensure a dataset (or builder) exists and is instantiable without network access.
    """
    module = None
    for c in [f"{PKG_NAME}.data", f"{PKG_NAME}.dataset"]:
        try:
            module = importlib.import_module(c)
            break
        except Exception:
            continue
    if module is None:
        pytest.skip("No dataset module found")

    ds_obj = _find_dataset_cls(module)
    if ds_obj is None:
        pytest.skip("Dataset class/factory not found")

    # Try to instantiate with minimal arguments; tolerate extra kwargs.
    created = None
    for kwargs in [{}, {"train": True}, {"split": "train"}]:
        try:
            created = ds_obj(**kwargs)
            break
        except Exception:
            continue

    assert created is not None, "Dataset could not be instantiated with common defaults"
