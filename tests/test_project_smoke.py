import os
import sys
import subprocess
import pytest

root_dir = os.getcwd()
src_dir = os.path.join(root_dir, "src")

@pytest.mark.parametrize("script", ["train.py", "infer.py", "eval.py"])
def test_cli_help_runs(script):
    """
    Verify the CLI responds to `--help` in a flat src/ layout.
    We run modules as package members (src.<module>) so relative imports inside them work.
    """
    src_script_path = os.path.join(src_dir , script)
    root_path = os.path.join(root_dir, script)
    stem = os.path.splitext(script)[0]

    if os.path.exists(src_script_path):
        env = os.environ.copy()
        # Put repo root on PYTHONPATH so `src` is importable
        env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([sys.executable, "-m", f"src.{stem}", "--help"], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"Module 'src.{stem} --help' failed.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        return

    if os.path.exists(root_path):
        proc = subprocess.run([sys.executable, root_path, "--help"], capture_output=True, text=True)
        assert proc.returncode == 0, f"Script '{root_path} --help' failed.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        return

    pytest.fail(f"Neither src/{script} nor {script} exists; cannot smoke-test CLI.")

def test_streamlit_app_present():
    demo_path = os.path.join(root_dir, "app.py")
    if not os.path.exists(demo_path):
        pytest.fail("Required demo entrypoint 'app.py' not found at repository root.")
    try:
        import streamlit  # noqa: F401
    except Exception as exc:
        pytest.fail(f"'streamlit' is required for the demo but is not importable: {exc}")
    assert os.path.exists(demo_path)
