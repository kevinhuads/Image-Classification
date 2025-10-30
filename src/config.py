from pathlib import Path
import random, numpy as np, torch, yaml

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf8") as f:
        return yaml.safe_load(f) or {}
    
    
def apply_yaml_to_args(args, yaml_cfg):
    """
    For every key in yaml_cfg, set attr on args only if CLI left it None.
    This gives CLI priority when a user explicitly overrides a YAML value.
    """
    for k, v in (yaml_cfg.items() if yaml_cfg else []):
        if not hasattr(args, k):
            # ignore unknown keys or add mapping if you renamed CLI args
            continue
        cur = getattr(args, k)
        # If CLI left it None OR (it's a bool False and YAML provides a bool), override
        if cur is None or (isinstance(cur, bool) and isinstance(v, bool) and cur == False):
            setattr(args, k, v)
    return args

def resolve_paths(args):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if args.data_folder is None:
        raise ValueError("data_folder must be provided (via CLI or --config).")
    data_folder = Path(args.data_folder).expanduser()
    if not data_folder.is_absolute():
        data_folder = (PROJECT_ROOT / data_folder).resolve()
    args.data_folder = str(data_folder)

    # coerce numeric fields that may come from YAML as strings
    for k in ("seed","num_workers","batch_size","epochs"):
        v = getattr(args, k, None)
        if v is not None: setattr(args, k, int(v))
    for k in ("lr","weight_decay"):
        v = getattr(args, k, None)
        if v is not None: setattr(args, k, float(v))
    return args

def set_seed(seed: int = 3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device_and_pin():
    has_cuda = torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    pin_memory = has_cuda or has_xpu
    return device, device.type, pin_memory
