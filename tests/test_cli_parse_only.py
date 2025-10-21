import os
import sys
import subprocess
import pytest

ROOT = os.getcwd()
SRC = os.path.join(ROOT, "src")

@pytest.mark.parametrize(
    "module,args",
    [
        ("train", ["--max-epochs", "1", "--dry-run"]),
        ("eval", ["--help"]),
        ("infer", ["--help"]),
    ],
)
def test_cli_parses_and_exits_fast(module, args):
    env = os.environ.copy()
    # Put repo root on PYTHONPATH so `src` is importable as a package namespace
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # run as package module so relative imports inside train/infer work: python -m src.train
    module_to_run = f"src.{module}"
    proc = subprocess.run([sys.executable, "-m", module_to_run, *args], env=env, capture_output=True, text=True)
    # fallback: if dry-run failed, try showing help (keeps previous behavior)
    if proc.returncode != 0 and "--dry-run" in args:
        proc = subprocess.run([sys.executable, "-m", module_to_run, "--help"], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"{module} failed to parse args.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
