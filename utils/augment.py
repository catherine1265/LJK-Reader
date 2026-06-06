import cv2
import numpy as np
from skimage.feature import hog

CHAR_H, CHAR_W = 32, 32


def augment_char(img, n=30):
    h, w    = img.shape
    results = [img]
    for _ in range(n):
        aug = img.copy()

        #Rotation ±15 degree
        angle = np.random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug   = cv2.warpAffine(aug, M, (w, h), borderValue=255)

        #Scale 85-115%
        sc    = np.random.uniform(0.85, 1.15)
        aug   = cv2.resize(aug, (max(1,int(w*sc)), max(1,int(h*sc))))
        aug   = cv2.resize(aug, (w, h))

        # Noise
        noise = np.random.normal(0, 8, aug.shape).astype(np.int16)
        aug   = np.clip(aug.astype(np.int16)+noise, 0, 255).astype(np.uint8)

        #Blur
        if np.random.rand() > 0.5:
            aug = cv2.GaussianBlur(aug, (3,3), 0)

        #Shear horizontal
        dx   = np.random.uniform(-4, 4)
        pts1 = np.float32([[0,0],[w,0],[0,h]])
        pts2 = np.float32([[dx,0],[w+dx,0],[0,h]])
        M    = cv2.getAffineTransform(pts1, pts2)
        aug  = cv2.warpAffine(aug, M, (w,h), borderValue=255)

        results.append(aug)
    return results


def extract_hog(img):
    return hog(img, orientations=9,
               pixels_per_cell=(8,8),
               cells_per_block=(2,2),
               block_norm='L2-Hys')


def preprocess_char(crop_bgr):
    b, g, r = cv2.split(crop_bgr.astype(np.float32))
    gray    = np.clip(r*0.8+g*0.15+b*0.05, 0, 255).astype(np.uint8)
    p_lo, p_hi = np.percentile(gray, 5), np.percentile(gray, 95)
    if p_hi > p_lo:
        gray = np.clip((gray.astype(np.float32)-p_lo)
                       /(p_hi-p_lo)*255, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return cv2.resize(bw, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
