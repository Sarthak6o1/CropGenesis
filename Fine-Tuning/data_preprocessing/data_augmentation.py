import os
import uuid
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import random

SOURCE_DIR = Path(r"C:\Users\gsart\Downloads\plantvillage\color")
SEG_DIR = Path(r"C:\Users\gsart\Downloads\plantvillage\segmented")
GRAY_DIR = Path(r"C:\Users\gsart\Downloads\plantvillage\grayscale")
AUG_PER_IMAGE = 5
IMG_EXTS = {".jpg", ".jpeg", ".png"}
IMG_SIZE = None

random.seed(42)

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

geom_aug = A.Compose(
    [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.7),
        A.Perspective(p=0.3),
        A.Affine(shear=10, p=0.2),
    ],
    additional_targets={"mask": "image", "gray": "image"},
)

color_aug = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([A.GaussNoise(p=0.5), A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5)], p=0.4),
        A.ColorJitter(p=0.4),
    ]
)

for cls in sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()]):
    src_dir = cls
    seg_out_dir = SEG_DIR / cls.name
    gray_out_dir = GRAY_DIR / cls.name
    color_out_dir = SOURCE_DIR / cls.name
    ensure(seg_out_dir)
    ensure(gray_out_dir)
    ensure(color_out_dir)

    files = [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    idx = 0
    for p in tqdm(files, desc=f"Augmenting {cls.name}"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        mask_path = seg_out_dir / p.name
        gray_path = gray_out_dir / p.name
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_tmp = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([95, 255, 255]))
            mask_img = mask_tmp
        if gray_path.exists():
            gray_img = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        else:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)
            mask_img = cv2.resize(mask_img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
            gray_img = cv2.resize(gray_img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        for _ in range(AUG_PER_IMAGE):
            geom = geom_aug(image=img, mask=mask_img, gray=gray_img)
            aug_img = geom["image"]
            aug_mask = geom["mask"]
            aug_gray = geom["gray"]
            ca = color_aug(image=aug_img)
            aug_img = ca["image"]
            if aug_mask.ndim == 3:
                aug_mask = cv2.cvtColor(aug_mask, cv2.COLOR_BGR2GRAY)
            _, aug_mask = cv2.threshold(aug_mask, 1, 255, cv2.THRESH_BINARY)
            if aug_gray.ndim == 3:
                aug_gray = cv2.cvtColor(aug_gray, cv2.COLOR_BGR2GRAY)

            uid = uuid.uuid4().hex[:12]
            out_img_name = f"{p.stem}_aug_{uid}{p.suffix}"
            out_mask_name = f"{p.stem}_aug_{uid}.png"
            out_gray_name = f"{p.stem}_aug_{uid}.png"

            cv2.imwrite(str(color_out_dir / out_img_name), aug_img)
            cv2.imwrite(str(seg_out_dir / out_mask_name), aug_mask)
            cv2.imwrite(str(gray_out_dir / out_gray_name), aug_gray)

        idx += 1
