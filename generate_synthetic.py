#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def make_sample(w=128, h=128, num_classes=3):
    bg = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    img = Image.new('RGB', (w,h), bg)
    mask = Image.new('L', (w,h), 0)
    draw = ImageDraw.Draw(img)
    mdraw = ImageDraw.Draw(mask)

    # class 1: rectangle
    x1,y1 = random.randint(5,40), random.randint(5,40)
    x2,y2 = random.randint(60,120), random.randint(60,120)
    draw.rectangle([x1,y1,x2,y2], fill=(255,0,0))
    mdraw.rectangle([x1,y1,x2,y2], fill=1)

    # class 2: circle
    r = random.randint(10,30)
    cx, cy = random.randint(20,108), random.randint(20,108)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0,255,0))
    mdraw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=2)

    return img, mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data')
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--size', type=int, default=128)
    args = ap.parse_args()

    out = Path(args.out); (out/'images').mkdir(parents=True, exist_ok=True); (out/'masks').mkdir(parents=True, exist_ok=True)
    for i in range(args.num):
        img, msk = make_sample(args.size, args.size, num_classes=3)
        img.save(out/'images'/f'img_{i:04d}.png')
        msk.save(out/'masks'/f'img_{i:04d}.png')
    print(f"[ok] wrote {args.num} samples to {out}/images and {out}/masks")

if __name__ == '__main__':
    main()
