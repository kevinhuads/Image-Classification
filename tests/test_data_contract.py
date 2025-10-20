# tests/test_data_contract.py
import importlib
import inspect
import pkgutil
import types
import os
import pytest
from typing import Optional, Tuple

PKG_NAME = "food101"


def _import_pkg_or_fail() -> types.ModuleType:
    try:
        return importlib.import_module(PKG_NAME)
    except Exception as exc:
        pytest.fail(f"Package '{PKG_NAME}' is not importable: {exc}")


def _is_callable_like(obj) -> bool:
    if callable(obj):
        return True
    if hasattr(obj, "__call__"):
        return True
    return False


def test_transforms_contract():
    """
    Verify a transforms factory exists and returns either:
      - a single callable transform, or
      - a tuple/list/dict containing callable transforms (e.g., (train_tfms, val_tfms)).
    """
    _import_pkg_or_fail()

    candidates = ["data", "transforms", "dataset", "datasets"]
    mod = None
    for name in candidates:
        try:
            mod = importlib.import_module(f"{PKG_NAME}.{name}")
            break
        except Exception:
            mod = None

    assert mod is not None, (
        f"Expected one of {candidates} submodules under '{PKG_NAME}' but none were importable."
    )

    factory = None
    for candidate in ("build_transforms", "get_transforms", "create_transforms", "make_transforms"):
        factory = getattr(mod, candidate, None)
        if callable(factory):
            break

    assert callable(factory), (
        "No transform factory found. Expected a callable named one of "
        "build_transforms/get_transforms/create_transforms/make_transforms in the data/transforms module."
    )

    # Call factory with minimal pattern if possible
    try:
        sig = inspect.signature(factory)
        if "train" in sig.parameters:
            result = factory(train=False)
        else:
            result = factory()
    except Exception as exc:
        pytest.fail(f"Transform factory raised an exception when called with minimal args: {exc}")

    if _is_callable_like(result):
        return

    if isinstance(result, (tuple, list)):
        assert all(_is_callable_like(x) for x in result), (
            "Transform factory returned a sequence but not all elements are callable-like."
        )
        return

    if isinstance(result, dict):
        assert all(_is_callable_like(v) for v in result.values()), (
            "Transform factory returned a dict but not all values are callable-like."
        )
        return

    pytest.fail(
        "Transforms object is not callable and is not a sequence/dict of callable transforms. "
        f"Got: {type(result)!r}"
    )


# ----------------------------
# Dataset-discovery & signature
# ----------------------------
DATASET_NAME_HINTS = (
    "dataset",
    "datasets",
    "dataloader",
    "dataloaders",
    "data",
)


def _iter_package_modules(pkg: types.ModuleType):
    """
    Yield (module_name, module) for modules under the package (first-level).
    """
    if not getattr(pkg, "__path__", None):
        return
    for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        full = f"{PKG_NAME}.{name}"
        try:
            mod = importlib.import_module(full)
            yield full, mod
        except Exception:
            # ignore import errors from optional modules
            continue


def _find_dataset_candidates(pkg: types.ModuleType) -> list[Tuple[str, object]]:
    """
    Search the package root and its first-level submodules for attributes that look like
    dataset classes or dataset factory callables. Return list of (qualified_name, attr).
    """
    candidates: list[Tuple[str, object]] = []

    # Inspect package-level attributes first
    for attr_name in dir(pkg):
        if attr_name.startswith("_"):
            continue
        lower = attr_name.lower()
        if any(h in lower for h in DATASET_NAME_HINTS) or attr_name.endswith("Dataset"):
            try:
                attr = getattr(pkg, attr_name)
            except Exception:
                continue
            if inspect.isclass(attr) or callable(attr):
                candidates.append((f"{PKG_NAME}.{attr_name}", attr))

    # Inspect first-level submodules
    for mod_name, mod in _iter_package_modules(pkg):
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            lower = attr_name.lower()
            if any(h in lower for h in DATASET_NAME_HINTS) or attr_name.endswith("Dataset"):
                try:
                    attr = getattr(mod, attr_name)
                except Exception:
                    continue
                if inspect.isclass(attr) or callable(attr):
                    candidates.append((f"{mod_name}.{attr_name}", attr))

        # also consider the module itself if its name is suggestive (e.g., data, dataset)
        base = mod_name.split(".")[-1].lower()
        if any(h == base for h in DATASET_NAME_HINTS):
            candidates.append((mod_name, mod))

    return candidates


def _signature_looks_ok(obj) -> bool:
    """
    Heuristic: inspect signature of callable/class __init__ and accept if:
     - no required positional-only parameters, OR
     - accepts a commonly-named dataset arg (root, root_dir, data_dir, split, train, download)
    If signature isn't introspectable, accept the candidate (C-extension or dynamic factory).
    """
    try:
        if inspect.isclass(obj):
            sig = inspect.signature(obj.__init__)
            params = list(sig.parameters.values())[1:]  # skip 'self'
        else:
            sig = inspect.signature(obj)
            params = list(sig.parameters.values())
    except (ValueError, TypeError):
        # Unable to get signature (C-extension or dynamic); accept as plausible
        return True

    # required positional params
    required = [
        p for p in params
        if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if not required:
        return True

    allowed_param_names = {"root", "root_dir", "data_dir", "split", "train", "download"}
    param_names = {p.name for p in params}
    if param_names & allowed_param_names:
        return True

    return False


def test_dataset_factory_signature():
    """
    Discover dataset-like constructors in the package and assert at least one has a
    reasonable signature (no required unnamed positional params, or accepts common dataset args).
    This avoids actually constructing datasets (which would require files or heavy deps).
    """
    pkg = _import_pkg_or_fail()

    candidates = _find_dataset_candidates(pkg)
    # Remove duplicates by qualified name
    seen = set()
    uniq = []
    for qname, attr in candidates:
        if qname in seen:
            continue
        seen.add(qname)
        uniq.append((qname, attr))

    if not uniq:
        # Provide helpful diagnostic info: list files in package directory (if available).
        pkg_path = getattr(pkg, "__file__", None)
        listing = ""
        if pkg_path:
            pkg_dir = os.path.dirname(pkg_path)
            try:
                listing = ", ".join(sorted(os.listdir(pkg_dir)))
            except Exception:
                listing = "<unable to list package directory>"
        pytest.fail(
            "No dataset-like constructors/factories discovered in the package. "
            f"Searched package root and first-level modules. Package dir listing: {listing}"
        )

    # Validate signatures; if any candidate looks OK, test passes.
    for qname, attr in uniq:
        if _signature_looks_ok(attr):
            return

    # If we get here, we discovered candidates but none had acceptable signatures.
    bad_list = ", ".join(q for q, _ in uniq)
    pytest.fail(
        "Found dataset-like candidates but none have a compatible signature. "
        f"Discovered candidates: {bad_list}. "
        "Signatures should accept either no required positional args or a common dataset arg like "
        "'root', 'split', or 'train'."
    )
