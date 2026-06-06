import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from google.colab import files
from utils.augment import preprocess_char

SAVE_DIR   = 'dataset/chars'
os.makedirs(SAVE_DIR, exist_ok=True)

CHAR_ROIS = {
    'A': (825, 668, 852, 710),
    '1': (853, 668, 868, 710),
    '7': (868, 668, 893, 710),
    '0': (893, 668, 923, 710),
    '9': (923, 668, 955, 710),
    'L': (570, 658, 607, 710),
    'K': (607, 658, 648, 710),
    'O': (648, 658, 695, 710),
    '2': (695, 658, 740, 710),
    '1': (855, 775, 878, 855),
    '7': (878, 775, 920, 855),
}


def collect_from_uploads():
    print('Upload semua foto LJK (bisa sekaligus)...')
    uploaded = files.upload()
    total    = 0

    for fname, data in uploaded.items():
        nparr   = np.frombuffer(data, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if raw_img is None:
            print(f'  Gagal baca: {fname}'); continue

        try:
            img_h, img_w = raw_img.shape[:2]
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
            contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for c in contours:
                area = cv2.contourArea(c)
                x,y,w,h = cv2.boundingRect(c)
                if w==0 or h==0: continue
                sol = area/float(w*h)
                asp = w/float(h)
                mn  = img_w*img_h*0.0003
                mx  = img_w*img_h*0.04
                if mn<area<mx and 0.4<asp<2.5 and sol>=0.65:
                    candidates.append({'cx':x+w//2,'cy':y+h//2,
                                       'x':x,'y':y,'w':w,'h':h})
            corners = {'TL':(0,0),'TR':(img_w,0),
                       'BR':(img_w,img_h),'BL':(0,img_h)}
            sel = {}
            for lbl,(tx,ty) in corners.items():
                if candidates:
                    sel[lbl] = min(candidates,
                        key=lambda a:(a['cx']-tx)**2+(a['cy']-ty)**2)
            if len(sel)==4:
                pts = np.float32([
                    [sel['TL']['cx'],sel['TL']['cy']],
                    [sel['TR']['cx'],sel['TR']['cy']],
                    [sel['BR']['cx'],sel['BR']['cy']],
                    [sel['BL']['cx'],sel['BL']['cy']]])
                s=pts.sum(1); d=np.diff(pts,axis=1)
                rect=np.zeros((4,2),dtype='float32')
                rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
                rect[1]=pts[np.argmin(d)]; rect[3]=pts[np.argmax(d)]
                mW=max(int(np.linalg.norm(rect[2]-rect[3])),
                       int(np.linalg.norm(rect[1]-rect[0])))
                mH=max(int(np.linalg.norm(rect[1]-rect[2])),
                       int(np.linalg.norm(rect[0]-rect[3])))
                dst=np.float32([[0,0],[mW-1,0],[mW-1,mH-1],[0,mH-1]])
                M  =cv2.getPerspectiveTransform(rect,dst)
                page=cv2.warpPerspective(raw_img,M,(mW,mH))
                page=cv2.resize(page,(1000,1414))
            else:
                page = raw_img
        except Exception:
            page = raw_img

        #Crop tiap karakter
        saved_this = 0
        for char, (x1,y1,x2,y2) in CHAR_ROIS.items():
            crop = page[y1:y2, x1:x2]
            if crop.size == 0: continue
            bw   = preprocess_char(crop)
            idx  = len([f for f in os.listdir(SAVE_DIR)
                        if f.startswith(f'{char}__')])
            cv2.imwrite(f'{SAVE_DIR}/{char}__{idx:04d}.png', bw)
            saved_this += 1

        print(f'  {fname}: {saved_this} karakter disimpan')
        total += saved_this

    print(f'\n  Total tersimpan: {total} karakter')
    print(f'  Distribusi:')
    for char in sorted(CHAR_ROIS.keys()):
        n = len([f for f in os.listdir(SAVE_DIR)
                 if f.startswith(f'{char}__')])
        print(f'    {char}: {n} sampel')


def preview_dataset():
    chars = sorted(set(f.split('__')[0]
                       for f in os.listdir(SAVE_DIR)
                       if f.endswith('.png')))
    if not chars:
        print('Dataset kosong.'); return

    fig, axes = plt.subplots(len(chars), 5,
                             figsize=(12, 2*len(chars)))
    if len(chars) == 1: axes = [axes]

    for i, char in enumerate(chars):
        files_c = sorted([f for f in os.listdir(SAVE_DIR)
                          if f.startswith(f'{char}__')])[:5]
        for j in range(5):
            axes[i][j].axis('off')
            if j < len(files_c):
                img = cv2.imread(
                    os.path.join(SAVE_DIR, files_c[j]),
                    cv2.IMREAD_GRAYSCALE)
                axes[i][j].imshow(img, cmap='gray')
                if j == 0:
                    axes[i][j].set_title(f'"{char}"',
                                         fontsize=11,
                                         fontweight='bold')

    plt.suptitle('Dataset — 5 sampel pertama per karakter',
                 fontweight='bold')
    plt.tight_layout()
    plt.show()
