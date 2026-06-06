import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def get_bubble_range(gray):
    """Auto-detect baris pertama & terakhir bubble."""
    blur = cv2.GaussianBlur(gray,(3,3),0)
    _,bw = cv2.threshold(blur,0,255,
                         cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    rp   = np.sum(bw,axis=1)/255
    w    = gray.shape[1]
    b0   = next((i for i,v in enumerate(rp) if v>w*0.03), 0)
    b1   = next((i for i in range(len(rp)-1,0,-1)
                 if rp[i]>w*0.03), gray.shape[0])
    return b0, b1

def scan_grid(gray, num_cols, num_rows, labels,
              z_thresh=1.2, z_gap=0.5, per_row=False):
    """
    Scan grid num_cols x num_rows.
    per_row=False → tiap kolom = 1 item (untuk NIM/nama)
    per_row=True  → tiap baris = 1 item, kolom = pilihan (untuk jawaban)
    Return: results list, density_map, b0, b1, row_h, col_w
    """
    eq  = apply_clahe(gray)
    inv = cv2.bitwise_not(eq)
    h,w = gray.shape
    b0,b1 = get_bubble_range(gray)
    bh    = b1 - b0
    row_h = bh / num_rows
    col_w = w  / num_cols

    density_map = np.zeros((num_rows, num_cols), dtype=float)
    raw         = np.zeros((num_rows, num_cols), dtype=float)

    for r in range(num_rows):
        for c in range(num_cols):
            y0 = b0 + int(r*row_h)+2;  y1b = b0+int((r+1)*row_h)-2
            x0 = int(c*col_w)+2;       x1b = int((c+1)*col_w)-2
            cell = inv[y0:y1b, x0:x1b]
            if cell.size==0: continue
            ch,cw = cell.shape
            cx = cell[ch//4:3*ch//4, cw//4:3*cw//4]
            raw[r,c] = float(np.mean(cx)) if cx.size>0 else 0.0

    results = []
    if not per_row:
        # per kolom → tiap kolom = 1 karakter
        for c in range(num_cols):
            arr  = raw[:,c]
            mean,std = arr.mean(), arr.std()+1e-6
            z    = (arr-mean)/std
            density_map[:,c] = z
            best = int(np.argmax(z))
            bz   = z[best]; sz = sorted(z)[-2]
            if bz>z_thresh and (bz-sz)>z_gap and best<len(labels):
                results.append(labels[best])
            else:
                results.append(None)
    else:
        # per baris → tiap baris = 1 soal
        for r in range(num_rows):
            arr  = raw[r,:]
            mean,std = arr.mean(), arr.std()+1e-6
            z    = (arr-mean)/std
            density_map[r,:] = z
            best = int(np.argmax(z))
            bz   = z[best]; sz = sorted(z)[-2]
            if bz>z_thresh and (bz-sz)>z_gap and best<len(labels):
                results.append(labels[best])
            else:
                results.append(None)

    return results, density_map, b0, b1, row_h, col_w

def show_detection(roi_bgr, results, density_map,
                   b0, b1, row_h, col_w,
                   row_labels, col_labels,
                   title, per_row=False):
    """Tampilkan 3-panel: ROI highlight + CLAHE + heatmap z-score."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    eq   = apply_clahe(gray)
    h,w  = gray.shape
    num_rows = len(row_labels)
    num_cols = len(col_labels)

    # Panel 1: highlight bubble terpilih
    vis = roi_bgr.copy()
    # garis grid
    for c in range(num_cols):
        cv2.line(vis,(int(c*col_w),b0),(int(c*col_w),b1),(255,140,0),1)
    for r in range(num_rows):
        cv2.line(vis,(0,int(b0+r*row_h)),(w,int(b0+r*row_h)),(200,200,200),1)

    if not per_row:
        for c,res in enumerate(results):
            if res is None: continue
            r   = row_labels.index(res) if res in row_labels else -1
            if r<0: continue
            x0,x1b = int(c*col_w), int((c+1)*col_w)
            y0,y1b = int(b0+r*row_h), int(b0+(r+1)*row_h)
            cv2.rectangle(vis,(x0,y0),(x1b,y1b),(0,255,0),2)
            cv2.putText(vis,res,(x0+2,y1b-2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,220,0),1)
        result_str = ''.join(r or '_' for r in results)
    else:
        for r,res in enumerate(results):
            if res is None: continue
            c   = col_labels.index(res) if res in col_labels else -1
            if c<0: continue
            x0,x1b = int(c*col_w), int((c+1)*col_w)
            y0,y1b = int(b0+r*row_h), int(b0+(r+1)*row_h)
            cv2.rectangle(vis,(x0,y0),(x1b,y1b),(0,255,0),2)
            cv2.putText(vis,res,(x0+2,y1b-2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,220,0),1)
        result_str = ' '.join(r or '-' for r in results)

    fig, axes = plt.subplots(1,3, figsize=(18,6))

    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Hasil Deteksi\n→ {result_str}',
                      fontweight='bold', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(eq, cmap='gray')
    axes[1].set_title('Setelah CLAHE\n(lighting diratakan)',
                      fontweight='bold', fontsize=10)
    axes[1].axis('off')

    im = axes[2].imshow(density_map, aspect='auto', cmap='RdYlGn',
                        interpolation='nearest', vmin=-3, vmax=3)
    axes[2].set_yticks(range(num_rows))
    axes[2].set_yticklabels(row_labels, fontsize=7)
    axes[2].set_xticks(range(num_cols))
    axes[2].set_xticklabels(col_labels, fontsize=7)
    axes[2].set_title('Z-score per Cell\n(hijau=paling gelap)',
                      fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=axes[2])

    plt.suptitle(title, fontsize=13, fontweight='bold',
                 color='darkgreen', y=1.01)
    plt.tight_layout()
    plt.show()

    return result_str
