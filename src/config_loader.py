from pathlib import Path
import yaml

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf8") as f:
        return yaml.safe_load(f) or {}
    
def merge_yaml_with_cli(yaml_cfg: dict, cli_args: dict) -> dict:
    """
    Return a merged dict where CLI args (cli_args) override YAML keys when not None.
    cli_args expected to be dict-like (e.g., vars(args) from argparse).
    """
    merged = dict(yaml_cfg)  # base
    for k, v in cli_args.items():
        if v is not None:
            merged[k] = v
    return merged
