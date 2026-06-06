import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_document_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    opened = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    dilated = cv2.dilate(opened, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest = None
    max_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 5000:

            peri = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(
                cnt,
                0.02 * peri,
                True
            )

            if len(approx) == 4 and area > max_area:
                largest = approx
                max_area = area

    return largest, gray, thresh, dilated


def detect_corner_squares(img):
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Threshold gelap ───────────────────────────────────────────────────
    _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=1)

    # ── Cari semua kontur ─────────────────────────────────────────────────
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # ── Filter: solid + mendekati persegi ────────────────────────────────
    candidates = []
    for c in contours:
        area    = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        if w == 0 or h == 0: continue
        solidity = area / float(w * h)
        aspect   = w / float(h)
        min_a = img_w * img_h * 0.0003
        max_a = img_w * img_h * 0.04
        if min_a < area < max_a and 0.4 < aspect < 2.5 and solidity >= 0.65:
            candidates.append({
                'area': area, 'x':x,'y':y,'w':w,'h':h,
                'cx': x+w//2, 'cy': y+h//2
            })

    # ── Dari semua kandidat, ambil 1 terdekat ke tiap sudut gambar ────────
    corners = {
        'TL': (0,        0       ),
        'TR': (img_w,    0       ),
        'BR': (img_w,    img_h   ),
        'BL': (0,        img_h   ),
    }

    selected = {}
    for label, (tx, ty) in corners.items():
        if not candidates:
            break
        best = min(candidates,
                   key=lambda a: (a['cx']-tx)**2 + (a['cy']-ty)**2)
        selected[label] = best

    return selected, candidates, img_h, img_w


def perspective_warp(img, selected):
    if len(selected) == 4:
        pts = np.array([
            [selected['TL']['cx'], selected['TL']['cy']],
            [selected['TR']['cx'], selected['TR']['cy']],
            [selected['BR']['cx'], selected['BR']['cy']],
            [selected['BL']['cx'], selected['BL']['cy']],
        ], dtype='float32')

        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.zeros((4,2), dtype='float32')
        rect[0]=pts[np.argmin(s)];    rect[2]=pts[np.argmax(s)]
        rect[1]=pts[np.argmin(diff)]; rect[3]=pts[np.argmax(diff)]

        maxW = max(int(np.linalg.norm(rect[2]-rect[3])),
                   int(np.linalg.norm(rect[1]-rect[0])))
        maxH = max(int(np.linalg.norm(rect[1]-rect[2])),
                   int(np.linalg.norm(rect[0]-rect[3])))
        dst  = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]],
                        dtype='float32')

        M      = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxW, maxH))
        warped = cv2.resize(warped, (1000, 1414))

        print(f'\n  Warp selesai -> 1000x1414 px')
        plt.figure(figsize=(8,11))
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title('Hasil Warp 1000x1414', fontweight='bold')
        plt.axis('off'); plt.tight_layout(); plt.show()

        return warped
    return img
