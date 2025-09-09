from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class SegFolder(Dataset):
    """Images in data/images/*.png, masks in data/masks/*.png
    Masks: single-channel, pixel values are class ids.
    """
    def __init__(self, data_dir: str, image_size: Tuple[int,int]=(128,128)):
        self.root = Path(data_dir)
        self.images = sorted((self.root/'images').glob('*.png'))
        self.masks  = sorted((self.root/'masks').glob('*.png'))
        assert len(self.images) == len(self.masks) and len(self.images) > 0, "Empty dataset or mismatch"
        self.size = image_size

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB').resize(self.size)
        msk = Image.open(self.masks[idx]).convert('L').resize(self.size, Image.NEAREST)
        img = np.array(img).astype('float32')/255.0
        img = (img.transpose(2,0,1) - 0.5)/0.5  # simple norm
        msk = np.array(msk).astype('int64')
        return torch.from_numpy(img), torch.from_numpy(msk)

def colorize_mask(msk: np.ndarray, num_classes: int=3):
    # simple fixed palette
    palette = np.array([
        [0,0,0],        # 0 background
        [255,0,0],      # 1
        [0,255,0],      # 2
        [0,0,255],      # 3
        [255,255,0],    # 4
        [255,0,255],    # 5
        [0,255,255],    # 6
    ], dtype=np.uint8)
    palette = palette[:max(num_classes,1)]
    rgb = palette[msk.clip(0, len(palette)-1)]
    return rgb
