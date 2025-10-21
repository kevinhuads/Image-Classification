import os
import sys
import subprocess
import pytest

ROOT_DIR = os.getcwd()
SRC_DIR = os.path.join(ROOT_DIR, "src")

@pytest.mark.parametrize("script", ["train.py", "infer.py", "eval.py"])
def test_cli_help_runs(script):
    """
    Verify the CLI responds to `--help` in a flat src/ layout.
    Strategy:
      - If a file exists at src/<script>, run as a module: `python -m <stem> --help`
      - Else, if a top-level <script> exists at repo root, run it directly.
    """
    src_script_path = os.path.join(SRC_DIR, script)
    root_path = os.path.join(ROOT_DIR, script)
    stem = os.path.splitext(script)[0]

    if os.path.exists(src_script_path):
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([sys.executable, "-m", stem, "--help"], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"Module '{stem} --help' failed.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        return

    if os.path.exists(root_path):
        proc = subprocess.run([sys.executable, root_path, "--help"], capture_output=True, text=True)
        assert proc.returncode == 0, f"Script '{root_path} --help' failed.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        return

    pytest.fail(f"Neither src/{script} nor {script} exists; cannot smoke-test CLI.")

def test_streamlit_app_present():
    """
    The demo 'app.py' is required at repository root.
    """
    demo_path = os.path.join(ROOT_DIR, "app.py")
    if not os.path.exists(demo_path):
        pytest.fail("Required demo entrypoint 'app.py' not found at repository root.")

    try:
        import streamlit  # noqa: F401
    except Exception as exc:
        pytest.fail(f"'streamlit' is required for the demo but is not importable: {exc}")

    assert os.path.exists(demo_path)
