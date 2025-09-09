#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from utils.unet import UNet
from utils.datasets import colorize_mask

def load_model(ckpt_path: str):
    data = torch.load(ckpt_path, map_location='cpu')
    num_classes = data.get('num_classes', 3)
    deeplab = data.get('deeplab', False)
    if deeplab:
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    else:
        model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    model.load_state_dict(data['model'])
    model.eval()
    return model, num_classes

def preprocess(img: Image.Image, size=128):
    arr = np.array(img.convert('RGB').resize((size,size))).astype('float32')/255.0
    arr = (arr.transpose(2,0,1) - 0.5)/0.5
    return torch.from_numpy(arr).unsqueeze(0)

def main():
    ap = argparse.ArgumentParser(description="Inference script for U-Net/DeepLabV3.")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--input', required=True, help='folder of images')
    ap.add_argument('--out', default='out_masks')
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--num-classes', type=int, default=3)
    args = ap.parse_args()

    model, num_classes_ckpt = load_model(args.checkpoint)
    num_classes = args.num_classes or num_classes_ckpt

    in_dir = Path(args.input); out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for p in sorted(list(in_dir.glob('*.png')) + list(in_dir.glob('*.jpg'))):
            img = Image.open(p)
            x = preprocess(img, args.img_size)
            y = model(x)['out'] if isinstance(model, dict) else model(x)
            msk = torch.argmax(y[0], dim=0).cpu().numpy().astype('uint8')
            rgb = colorize_mask(msk, num_classes=num_classes)
            Image.fromarray(rgb).save(out_dir / f"{p.stem}_mask.png")
            print(f"[ok] saved {out_dir / (p.stem + '_mask.png')}")

if __name__ == '__main__':
    main()
