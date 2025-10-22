import argparse
from PIL import Image
from config_loader import load_yaml
from utils import load_model_and_meta, predict_pil, topk_labels

def apply_yaml_to_args(args, yaml_cfg):
    """
    For every key in yaml_cfg, set attr on args only if CLI left it None.
    This gives CLI priority when a user explicitly overrides a YAML value.
    """
    for k, v in (yaml_cfg.items() if yaml_cfg else []):
        if not hasattr(args, k):
            # ignore unknown keys or add mapping if you renamed CLI args
            continue
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path to YAML config")
    parser.add_argument("--image_path", help="Path to input image (jpg/png)", default=None)
    parser.add_argument("--ckpt", default=r"artifacts\headonly_food101.pth")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    yaml_cfg = {}
    if args.config:
        yaml_cfg = load_yaml(args.config)

    args = apply_yaml_to_args(args, yaml_cfg)

    if args.image_path is None:
        raise ValueError("image_path must be provided via CLI or --config")

    model, device, preprocess, classes = load_model_and_meta(args.ckpt)
    img = Image.open(args.image_path).convert("RGB")
    preds = predict_pil(img, model, device, preprocess, topk=args.topk)
    labeled = topk_labels(preds, classes)
    for label, prob in labeled:
        print(f"{label}: {prob:.4f}")

if __name__ == "__main__":
    main()
