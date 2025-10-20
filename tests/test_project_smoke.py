# tests/test_project_smoke.py
import os
import sys
import subprocess
import pytest

PKG_NAME = "food101"  # adjust if package name differs
ROOT_DIR = os.getcwd()
SRC_DIR = os.path.join(ROOT_DIR, "src")


def _is_package_importable():
    try:
        __import__(PKG_NAME)
        return True
    except Exception:
        # try with src on sys.path
        if SRC_DIR not in sys.path:
            sys.path.insert(0, SRC_DIR)
        try:
            __import__(PKG_NAME)
            return True
        except Exception:
            return False


def test_package_importable():
    """
    Ensure the package is importable. This is a hard requirement.
    Fail with a clear message if import fails.
    """
    if not _is_package_importable():
        pytest.fail(
            f"Package '{PKG_NAME}' is not importable. "
            "Ensure package is installed or that 'src/' contains the package and PYTHONPATH is correct."
        )


@pytest.mark.parametrize("script", ["train.py", "infer.py", "eval.py"])
def test_cli_help_runs(script):
    """
    Verify the CLI responds to `--help`. This test fails if the CLI is missing or if running it fails.
    Execution strategy:
      - Prefer running as a module (python -m food101.<name>) when the package layout uses src/ and
        package-relative imports are present. This avoids "attempted relative import" errors.
      - If only a top-level script exists at repo root, run that script directly.
    """
    root_path = os.path.join(ROOT_DIR, script)
    src_script_path = os.path.join(SRC_DIR, PKG_NAME, script)
    module_name = f"{PKG_NAME}.{os.path.splitext(script)[0]}"

    # Determine available method(s)
    root_exists = os.path.exists(root_path)
    src_script_exists = os.path.exists(src_script_path)
    package_importable = _is_package_importable()

    if not (root_exists or src_script_exists or package_importable):
        pytest.fail(
            f"CLI '{script}' not found and package '{PKG_NAME}' is not importable. "
            "Expected one of: top-level script, src/<pkg>/<script>, or installed package."
        )

    # If src layout present or package importable, run as module to preserve package context (preferred).
    if src_script_exists or package_importable:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([SRC_DIR, existing]) if existing else SRC_DIR
        proc = subprocess.run([sys.executable, "-m", module_name, "--help"], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(
                f"Running module '{module_name} --help' failed with return code {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        return

    # Fallback: run top-level script directly (only if present)
    if root_exists:
        proc = subprocess.run([sys.executable, root_path, "--help"], capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(
                f"Running script '{root_path} --help' failed with return code {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        return


def test_streamlit_app_present():
    """
    The demo 'app.py' is required. Fail if missing or if 'streamlit' is not importable.
    """
    demo_path = os.path.join(ROOT_DIR, "app.py")
    if not os.path.exists(demo_path):
        pytest.fail("Required demo entrypoint 'app.py' not found at repository root.")

    try:
        import streamlit  # noqa: F401
    except Exception as exc:
        pytest.fail(f"'streamlit' is required for the demo but is not importable: {exc}")

    # Optional: run a light smoke `--version` or `--help` style check, but avoid launching the UI.
    # streamlit has a CLI, but it's unnecessary here; presence of importable streamlit is sufficient.
    assert os.path.exists(demo_path), "app.py should exist (sanity check)."
