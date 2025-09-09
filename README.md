# Image Segmentation Demo — U‑Net / DeepLabV3

 PyTorch repo for **semantic segmentation**. Two models included:
- **U‑Net (from scratch)**
- **DeepLabV3‑ResNet50** (from `torchvision`)

Comes with a tiny **synthetic dataset generator** so you can run end‑to‑end without downloading anything. Swap in your own dataset later.

## What’s here
- `train_unet.py` — trains U‑Net on your dataset (or the synthetic one).
- `infer_unet.py` — runs inference and saves colored masks.
- `utils/datasets.py` — dataset + transforms.
- `utils/unet.py` — small, readable U‑Net.
- `generate_synthetic.py` — makes a toy dataset (`data/`) with images + masks.
- `requirements.txt` — PyTorch + torchvision + Pillow + numpy + tqdm.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) create toy data: data/images/*.png and data/masks/*.png (class ids 0..N)
python generate_synthetic.py --num 200

# 2) train (U-Net)
python train_unet.py --data-dir data --epochs 5 --batch-size 8 --num-classes 3

# 3) inference on a folder
python infer_unet.py --checkpoint runs/best.pt --input data/images --out out_masks --num-classes 3
```

## Your data format
```
data/
  images/  img_000.png, img_001.png, ...
  masks/   img_000.png, img_001.png, ...  # single‑channel, values in [0..num_classes-1]
```
Change `--num-classes` accordingly.

## Notes
- U‑Net is intentionally small & readable. Good for learning or quick baselines.
- DeepLabV3 training skeleton is included in `train_unet.py` (flag `--deeplab`).

MIT License
