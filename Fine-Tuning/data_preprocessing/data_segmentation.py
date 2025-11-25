import os
import cv2
import numpy as np
from tqdm import tqdm

SOURCE_DIR = r"C:\Users\gsart\Downloads\plantvillage\color"
SEG_DIR = r"C:\Users\gsart\Downloads\plantvillage\segmented"
GRAY_DIR = r"C:\Users\gsart\Downloads\plantvillage\grayscale"

def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def segment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([95, 255, 255]))
    return cv2.bitwise_and(img, img, mask=mask)

for cls in os.listdir(SOURCE_DIR):
    src = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(src):
        continue

    seg_out = os.path.join(SEG_DIR, cls)
    gray_out = os.path.join(GRAY_DIR, cls)

    ensure(seg_out)
    ensure(gray_out)

    print(f"Segmenting: {cls}")

    for img_name in tqdm(os.listdir(src)):
        path = os.path.join(src, img_name)
        img = cv2.imread(path)
        if img is None:
            continue

        seg = segment(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(os.path.join(seg_out, img_name), seg)
        cv2.imwrite(os.path.join(gray_out, img_name), gray)

print("Segmentation Complete")
