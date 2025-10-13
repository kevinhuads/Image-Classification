# infer.py
import argparse
from PIL import Image
from utils import load_model_and_meta, predict_pil, topk_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image (jpg/png)")
    parser.add_argument("--ckpt", default=r"D:\Python\food-101\headonly_food101.pth")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    model, device, preprocess, classes = load_model_and_meta(args.ckpt)
    img = Image.open(args.image_path).convert("RGB")
    preds = predict_pil(img, model, device, preprocess, topk=args.topk)
    labeled = topk_labels(preds, classes)
    for label, prob in labeled:
        print(f"{label}: {prob:.4f}")

if __name__ == "__main__":
    main()
