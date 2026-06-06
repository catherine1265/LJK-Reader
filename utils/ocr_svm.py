import cv2
import numpy as np
import re
from utils.augment import extract_hog, CHAR_H, CHAR_W

N_CHARS = {
    'RUANGAN'         : 5,
    'KODE_KELAS'      : 4,
    'NO_MEJA'         : 2,
    'NAMA_MATA_KULIAH': 13,
}


def preprocess(roi_bgr):
    b, g, r = cv2.split(roi_bgr.astype(np.float32))
    gray = np.clip(r*0.8 + g*0.15 + b*0.05, 0, 255).astype(np.uint8)

    h, w = gray.shape
    gray = gray[int(h*0.08):h-int(h*0.08),
                int(w*0.08):w-int(w*0.08)]
    if gray.size == 0: return None

    p_lo, p_hi = np.percentile(gray, 10), np.percentile(gray, 90)
    if p_hi > p_lo:
        gray = np.clip((gray.astype(np.float32)-p_lo)
                       /(p_hi-p_lo)*255, 0, 255).astype(np.uint8)

    gray = cv2.filter2D(gray, -1,
                        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

    h, w  = gray.shape
    scale = max(1.0, 64/h)
    return cv2.resize(gray, (int(w*scale), int(h*scale)),
                      interpolation=cv2.INTER_CUBIC)

def binarize(gray):
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw

def segment_fixed(binary, n_chars):
    h, w   = binary.shape
    char_w = w // n_chars
    chars  = []

    for i in range(n_chars):
        x0   = i * char_w
        x1   = (i+1) * char_w if i < n_chars-1 else w
        crop = binary[:, x0:x1]

        row_proj = np.sum(crop, axis=1)
        rows     = np.where(row_proj > 0)[0]
        if len(rows) > 0:
            y0, y1 = rows[0], rows[-1]+1
            crop   = crop[y0:y1, :]

        if crop.size == 0:
            crop = np.zeros((CHAR_H, CHAR_W), dtype=np.uint8)

        crop = cv2.resize(crop, (CHAR_W, CHAR_H),
                          interpolation=cv2.INTER_AREA)
        chars.append(crop)

    debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for i in range(1, n_chars):
        x = i * char_w
        cv2.line(debug, (x, 0), (x, h), (0, 255, 0), 2)

    return chars, debug

def predict_text(roi_bgr, bundle, label=''):
    gray = preprocess(roi_bgr)
    if gray is None:
        empty = np.zeros((10,10), np.uint8)
        return '', empty, cv2.cvtColor(empty, cv2.COLOR_GRAY2BGR)

    binary       = binarize(gray)
    n            = N_CHARS.get(label, 6)
    chars, debug = segment_fixed(binary, n)

    feats  = np.array([extract_hog(c) for c in chars])
    labels = bundle['clf'].predict(feats)
    known  = set(bundle['chars'])
    result = ''.join(l for l in labels if l in known)
    return result, binary, debug

def postprocess(label, text):
    text = text.strip()
    if label == 'NAMA_MATA_KULIAH':
        text = re.sub(r'^[^A-Za-z]+', '', text)
        corrections = {
            'Pomputer': 'Computer', 'Gomputer': 'Computer',
            'Yision'  : 'Vision',
        }
        for wrong, right in corrections.items():
            text = re.sub(wrong, right, text, flags=re.IGNORECASE)
        text = text.title()
    elif label in ('KODE_KELAS', 'RUANGAN'):
        text = text.replace(' ', '').upper()
    elif label == 'NO_MEJA':
        text = ''.join(re.findall(r'\d+', text))
    return text
