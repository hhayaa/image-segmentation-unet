#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from utils.unet import UNet
from utils.datasets import SegFolder

def get_model(num_classes: int, deeplab: bool):
    if deeplab:
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    else:
        model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

def main():
    ap = argparse.ArgumentParser(description="Train U-Net (or DeepLabV3) for segmentation.")
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-classes', type=int, default=3)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--deeplab', action='store_true', help='use DeepLabV3 instead of U-Net')
    ap.add_argument('--out', default='runs')
    args = ap.parse_args()

    ds = SegFolder(args.data_dir, image_size=(args.img_size, args.img_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.num_classes, args.deeplab).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1e9
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / 'best.pt'

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        total = 0.0
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optim.zero_grad()
            logits = model(imgs)
            if isinstance(logits, dict):  # DeepLab output
                logits = logits['out']
            loss = criterion(logits, masks)
            loss.backward()
            optim.step()
            total += loss.item()*imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg = total/len(ds)
        print(f"epoch {epoch+1}: avg_loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({'model': model.state_dict(),
                        'num_classes': args.num_classes,
                        'deeplab': args.deeplab}, ckpt)
            print(f"[ok] saved {ckpt}")

if __name__ == '__main__':
    main()
