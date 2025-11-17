# tests/conftest.py
import sys
import os

root_dir = os.getcwd()
src_dir = os.path.join(root_dir, "src")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)