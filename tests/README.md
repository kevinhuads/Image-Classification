# tests

This folder contains minimal smoke tests and a CI entrypoint.

- `test_project_smoke.py`:
  - verifies the package is importable (attempts `import food101`, falling back to `src` path),
  - calls `--help` on CLI scripts to ensure they are runnable without invoking heavy computation,
  - verifies the presence of `app.py` (the Streamlit demo entrypoint).

These tests are intentionally light-weight so the CI runs quickly. Add integration tests that exercise training/evaluation once a small, deterministic dataset and fixed configuration are available.