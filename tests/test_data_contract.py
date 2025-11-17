import importlib
import inspect
import pytest

# ---- utilities ----
def _is_callable_like(obj) -> bool:
    return callable(obj) or hasattr(obj, "__call__")

# ---- tests ----
def test_transforms_contract_flat_layout():
    """
    Verify a transforms factory exists in a flat src/ layout and returns either:
      - a single callable transform, or
      - a tuple/list/dict of callable transforms.
    Searched modules: data, transforms, dataset, datasets (first hit wins).
    """
    candidates = ["data", "transforms", "dataset", "datasets"]
    mod = None
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            break
        except Exception:
            mod = None
    assert mod is not None, f"Expected one of {candidates} modules in src/, but none import."

    factory = None
    for candidate in ("build_transforms", "get_transforms", "create_transforms", "make_transforms"):
        factory = getattr(mod, candidate, None)
        if callable(factory):
            break
    assert callable(factory), (
        "No transform factory found. Expected a callable named one of "
        "build_transforms/get_transforms/create_transforms/make_transforms in the data/transforms module."
    )

    # Minimal call contract
    try:
        sig = inspect.signature(factory)
        if "train" in sig.parameters:
            result = factory(train=False)
        else:
            result = factory()
    except Exception as exc:
        pytest.fail(f"Transform factory raised an exception with minimal args: {exc}")

    if _is_callable_like(result):
        return
    if isinstance(result, (tuple, list)):
        assert all(_is_callable_like(x) for x in result), "Sequence return must contain callable-like elements."
        return
    if isinstance(result, dict):
        assert all(_is_callable_like(v) for v in result.values()), "Dict return must have callable-like values."
        return
    pytest.fail(f"Unexpected transforms object type: {type(result)!r}")

def test_dataset_factory_signature_flat_layout():
    """
    Discover dataset-like constructors in flat modules and assert at least one has a
    reasonable signature (no required unnamed positional args, or accepts common dataset args).
    """
    import os

    # Probe a few likely modules without enforcing any single name.
    module_names = ["data", "dataset", "datasets", "dataloader", "dataloaders"]
    discovered: list[tuple[str, object]] = []

    for name in module_names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        # scan module attributes
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            lower = attr_name.lower()
            if any(h in lower for h in ("dataset", "dataloader", "data")) or attr_name.endswith("Dataset"):
                try:
                    attr = getattr(mod, attr_name)
                except Exception:
                    continue
                if inspect.isclass(attr) or callable(attr):
                    discovered.append((f"{name}.{attr_name}", attr))

        # also accept the module itself if its name is suggestive
        if name in ("data", "dataset", "datasets"):
            discovered.append((name, mod))

    if not discovered:
        # Provide directory listing to aid debugging
        listing = ", ".join(sorted(os.listdir("src"))) if os.path.isdir("src") else "<no src dir>"
        pytest.fail(
            "No dataset-like constructors/factories discovered in flat modules. "
            f"Searched modules: {module_names}. src/ listing: {listing}"
        )

    def _signature_looks_ok(obj) -> bool:
        try:
            if inspect.isclass(obj):
                sig = inspect.signature(obj.__init__)
                params = list(sig.parameters.values())[1:]  # skip self
            else:
                sig = inspect.signature(obj)
                params = list(sig.parameters.values())
        except (ValueError, TypeError):
            return True
        required = [
            p for p in params
            if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if not required:
            return True
        allowed = {"root", "root_dir", "data_dir", "split", "train", "download"}
        names = {p.name for p in params}
        return bool(names & allowed)

    assert any(_signature_looks_ok(attr) for _, attr in discovered), (
        "Found dataset-like candidates but none have a compatible signature. "
        f"Discovered: {', '.join(q for q, _ in discovered)}"
    )
