import cv2
import numpy as np
import os
import pickle

from utils.warp import detect_corner_squares, perspective_warp
from utils.bubble_scan import scan_grid, apply_clahe
from utils.augment import extract_hog, CHAR_H, CHAR_W
from train.train_model import load_or_train
from utils.ocr_svm import N_CHARS

MODEL_PATH = 'svm_emnist.pkl'

# ── Load model sekali saat import ────────────────────────────
_bundle = None

def _get_bundle():
    global _bundle
    if _bundle is None:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                _bundle = pickle.load(f)
        else:
            _bundle = load_or_train()
    return _bundle


# ════════════════════════════════════════════════════
#  WARP
# ════════════════════════════════════════════════════
def warp_ljk(img_pil):
    """
    Input : PIL Image
    Output: (warped_np BGR, success bool)
    """
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    selected, candidates, img_h, img_w = detect_corner_squares(img_np)
    if len(selected) < 4:
        return img_np, False
    warped = perspective_warp(img_np, selected)
    return warped, True


# ════════════════════════════════════════════════════
#  DETEKSI NAMA
# ════════════════════════════════════════════════════
def detect_nama(warped):
    """
    Input : warped BGR numpy array (1000x1414)
    Output: string nama
    """
    X1, Y1, X2, Y2 = 40, 270, 520, 850
    ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    NUM_COLS = 20

    roi      = warped[Y1:Y2, X1:X2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_h, roi_w = roi_gray.shape

    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_eq  = clahe.apply(roi_gray)
    roi_inv = cv2.bitwise_not(roi_eq)

    blur     = cv2.GaussianBlur(roi_gray, (3,3), 0)
    _, bw    = cv2.threshold(blur, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_proj = np.sum(bw, axis=1) / 255

    bubble_start = next((i for i,v in enumerate(row_proj)
                         if v > roi_w * 0.05), 0)
    bubble_end   = next((i for i in range(roi_h-1,0,-1)
                         if row_proj[i] > roi_w * 0.05), roi_h)

    bubble_h = bubble_end - bubble_start
    row_h    = bubble_h / 26
    col_w    = roi_w / NUM_COLS

    nama_result = []
    for col in range(NUM_COLS):
        x0 = int(col     * col_w) + 2
        x1 = int((col+1) * col_w) - 2
        scores = []
        for row in range(26):
            y0   = bubble_start + int(row     * row_h) + 2
            y1   = bubble_start + int((row+1) * row_h) - 2
            cell = roi_inv[y0:y1, x0:x1]
            if cell.size == 0:
                scores.append(0.0); continue
            ch, cw = cell.shape
            center = cell[ch//4:3*ch//4, cw//4:3*cw//4]
            scores.append(float(np.mean(center)) if center.size > 0 else 0.0)

        arr  = np.array(scores)
        mean = arr.mean(); std = arr.std() + 1e-6
        z    = (arr - mean) / std
        best_row = int(np.argmax(z))
        best_z   = z[best_row]
        second_z = sorted(z)[-2]
        if best_z > 1.2 and (best_z - second_z) > 0.5:
            nama_result.append(ALPHABET[best_row])
        else:
            nama_result.append('_')

    return ''.join(nama_result).replace('_', ' ').strip()


# ════════════════════════════════════════════════════
#  DETEKSI NIM
# ════════════════════════════════════════════════════
def detect_nim(warped):
    """
    Input : warped BGR numpy array (1000x1414)
    Output: string NIM
    """
    x1, y1, x2, y2 = 550, 270, 790, 500
    roi_gray = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    results, _, _, _, _, _ = scan_grid(
        roi_gray, num_cols=10, num_rows=10,
        labels=[str(i) for i in range(10)], per_row=False)
    return ''.join(r or '_' for r in results)


# ════════════════════════════════════════════════════
#  DETEKSI TANGGAL
# ════════════════════════════════════════════════════
def detect_tanggal(warped):
    """
    Input : warped BGR numpy array (1000x1414)
    Output: string tanggal
    """
    x1, y1, x2, y2 = 820, 270, 970, 500
    roi_gray = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    results, _, _, _, _, _ = scan_grid(
        roi_gray, num_cols=6, num_rows=10,
        labels=[str(i) for i in range(10)], per_row=False)
    return ''.join(r or '_' for r in results)


# ════════════════════════════════════════════════════
#  DETEKSI JAWABAN
# ════════════════════════════════════════════════════
def detect_answers(warped, total_soal=50):
    """
    Input : warped BGR numpy array (1000x1414), jumlah soal
    Output: dict {nomor_soal: jawaban} e.g. {1:'A', 2:'C', ...}
    """
    CHOICES = ['A','B','C','D','E']
    ROI_JAWABAN = [
        ('1-10',    70,  930, 190, 1150,  1),
        ('11-20',   70, 1160, 190, 1390, 11),
        ('21-30',  250,  930, 380, 1150, 21),
        ('31-40',  259, 1160, 380, 1390, 31),
        ('41-50',  450,  930, 580, 1150, 41),
        ('51-60',  450, 1160, 580, 1390, 51),
        ('61-70',  650,  930, 780, 1150, 61),
        ('71-80',  650, 1160, 780, 1390, 71),
        ('81-90',  830,  930, 950, 1150, 81),
        ('91-100', 830, 1160, 950, 1390, 91),
    ]

    all_answers = {}
    soal_done   = 0

    for label, x1, y1, x2, y2, q_start in ROI_JAWABAN:
        if soal_done >= total_soal:
            break
        soal_di_blok = min(10, total_soal - soal_done)
        roi_gray = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        r_blok, _, _, _, _, _ = scan_grid(
            roi_gray, num_cols=5, num_rows=10,
            labels=CHOICES, per_row=True)
        for i, ans in enumerate(r_blok[:soal_di_blok]):
            all_answers[q_start + i] = ans
        soal_done += soal_di_blok

    return all_answers


# ════════════════════════════════════════════════════
#  HEATMAP BUBBLE
# ════════════════════════════════════════════════════
def get_heatmap(warped):
    """
    Input : warped BGR numpy array (1000x1414)
    Output: RGB numpy array heatmap area jawaban
    """
    # Crop area jawaban keseluruhan
    y1, y2 = 930, 1390
    x1, x2 = 70, 950
    roi = warped[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq    = clahe.apply(gray)
    inv   = cv2.bitwise_not(eq)

    # Normalize ke 0-255 untuk colormap
    norm  = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(roi, 0.4, heatmap_bgr, 0.6, 0)

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
