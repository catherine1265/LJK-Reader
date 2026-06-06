# AOL_CV — LJK Scanner

Pipeline pemrosesan Lembar Jawaban Komputer (LJK) menggunakan OpenCV + SVM.

## Struktur Repo

```
AOL_CV/
├── README.md
├── requirements.txt
├── utils/
│   ├── display.py          # show(), show_row(), print_step()
│   ├── warp.py             # deteksi sudut + perspective warp
│   ├── bubble_scan.py      # scan_grid(), show_detection()
│   ├── ocr_svm.py          # predict_text(), postprocess()
│   └── augment.py          # augment_char(), extract_hog()
├── train/
│   ├── collect_data.py     # collect_from_uploads(), preview_dataset()
│   └── train_model.py      # load_or_train(), retrain_svm()
└── main.ipynb              # Notebook utama (jalankan di Google Colab)
```

## Cara Pakai

1. Buka `main.ipynb` di Google Colab
2. Run semua cell secara berurutan
3. Upload foto LJK saat diminta
4. Hasil scan otomatis muncul + bisa download Excel

## Status Fitur

| Fitur | Status |
|-------|--------|
| Perspective Warp | ✅ Selesai |
| Deteksi Nama | ✅ Selesai |
| Deteksi NIM | ✅ Selesai |
| Deteksi Tanggal | ✅ Selesai |
| Deteksi Jawaban | ✅ Selesai |
| OCR Tulisan Tangan | ✅ Selesai |
| EDA | 🚧 Belum tersedia |
