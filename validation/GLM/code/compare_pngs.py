import argparse
import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def open_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_to_match(a: Image.Image, b: Image.Image) -> Tuple[Image.Image, Image.Image]:
    if a.size == b.size:
        return a, b
    b_resized = b.resize(a.size, Image.LANCZOS)
    return a, b_resized


def compute_metrics(a: Image.Image, b: Image.Image) -> dict:
    a_arr = np.asarray(a).astype(np.float64)
    b_arr = np.asarray(b).astype(np.float64)
    mse = np.mean((a_arr - b_arr) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)

    # Normalized cross-correlation per channel, then average
    def ncc(x, y):
        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = np.sqrt((x0 ** 2).sum() * (y0 ** 2).sum())
        if denom == 0:
            return 0.0
        return float((x0 * y0).sum() / denom)

    ncc_vals = [ncc(a_arr[..., i], b_arr[..., i]) for i in range(3)]
    ncc_mean = float(np.mean(ncc_vals))

    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ncc_r": float(ncc_vals[0]),
        "ncc_g": float(ncc_vals[1]),
        "ncc_b": float(ncc_vals[2]),
        "ncc_mean": ncc_mean,
    }


def make_side_by_side(a: Image.Image, b: Image.Image, label_a: str, label_b: str) -> Image.Image:
    w, h = a.size
    gap = 10
    title_h = 30
    out = Image.new("RGB", (w * 2 + gap, h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)

    # Optional font (system dependent); fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((5, 5), label_a, fill=(0, 0, 0), font=font)
    draw.text((w + gap + 5, 5), label_b, fill=(0, 0, 0), font=font)

    out.paste(a, (0, title_h))
    out.paste(b, (w + gap, title_h))
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare two PNGs: side-by-side + metrics")
    parser.add_argument("--a", required=True, help="Our generated PNG path")
    parser.add_argument("--b", required=True, help="Reference PNG path")
    parser.add_argument("--out", default="out/GLM_compare.png", help="Output side-by-side image path")
    parser.add_argument("--report", default="out/GLM_compare.txt", help="Metrics report output path")
    args = parser.parse_args()

    img_a = open_image_rgb(args.a)
    img_b = open_image_rgb(args.b)
    img_a, img_b = resize_to_match(img_a, img_b)

    metrics = compute_metrics(img_a, img_b)
    side = make_side_by_side(img_a, img_b, label_a=os.path.basename(args.a), label_b=os.path.basename(args.b))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    side.save(args.out)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("Saved:", args.out)
    print("Report:", args.report)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


