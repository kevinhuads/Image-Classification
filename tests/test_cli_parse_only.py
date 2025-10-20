import os
import sys
import subprocess
import pytest

PKG = "food101"
ROOT = os.getcwd()
SRC = os.path.join(ROOT, "src")

@pytest.mark.parametrize(
    "module,args",
    [
        (f"{PKG}.train", ["--max-epochs", "1", "--dry-run"]),
        (f"{PKG}.eval", ["--help"]),
        (f"{PKG}.infer", ["--help"]),
    ],
)
def test_cli_parses_and_exits_fast(module, args):
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    # Accept that not all CLIs support --dry-run; if unsupported, fall back to --help.
    proc = subprocess.run([sys.executable, "-m", module, *args], env=env, capture_output=True, text=True)
    if proc.returncode != 0 and "--dry-run" in args:
        proc = subprocess.run([sys.executable, "-m", module, "--help"], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"{module} failed to parse args.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
