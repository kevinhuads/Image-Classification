# config.py
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent  
DEFAULT_DATA = PROJECT_ROOT / "data"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "food101"

def _load_yaml(path: Path | None):
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    artifacts_dir: Path
    ckpt_name: str

    @property
    def images_dir(self) -> Path: return self.data_dir / "images"
    @property
    def meta_dir(self) -> Path:   return self.data_dir / "meta"
    @property
    def ckpt_path(self) -> Path:  return self.artifacts_dir / self.ckpt_name
    @property
    def results_csv(self) -> Path:return self.artifacts_dir / "results.csv"

def load_config(config_file: Path | None = None) -> AppConfig:
    env = os.environ
    yml = _load_yaml(config_file or PROJECT_ROOT / "config.yaml")

    data_dir = Path(
        env.get("FOOD101_DATA_DIR")
        or yml.get("data_dir")
        or DEFAULT_DATA
    ).resolve()

    artifacts_dir = Path(
        env.get("FOOD101_ARTIFACT_DIR")
        or yml.get("artifacts_dir")
        or DEFAULT_ARTIFACTS
    ).resolve()

    ckpt_name = env.get("FOOD101_CKPT_NAME") or yml.get("ckpt_name") or "best.pth"

    return AppConfig(data_dir=data_dir, artifacts_dir=artifacts_dir, ckpt_name=ckpt_name)
