import cv2
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow_datasets as tfds
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.augment import augment_char, extract_hog, CHAR_H, CHAR_W

SAVE_DIR   = 'dataset/chars'
MODEL_PATH = 'svm_emnist.pkl'

ALL_CHARS = sorted(
    list('0123456789') +
    list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
    list('abcdefghijklmnopqrstuvwxyz')
)


def label_to_char(label):
    label = int(label)
    if label < 10:
        return str(label)
    elif label < 36:
        return chr(ord('A') + label - 10)
    else:
        return chr(ord('a') + label - 36)


def load_or_train():
    if os.path.exists(MODEL_PATH):
        print('Model ditemukan, loading...')
        with open(MODEL_PATH, 'rb') as f:
            bundle = pickle.load(f)
        if 'chars' not in bundle:
            bundle['chars'] = ALL_CHARS
        print(f'  Karakter: {len(bundle["chars"])} kelas: {sorted(bundle["chars"])}'
)
        return bundle

    print('Training model with EMNIST ByClass AND custom uploaded characters (if any).')
    print('  This will take a while (~10-15 minutes or more depending on sample size and resources).')

    X_all, y_all = [], []
    current_chars = set()

    if os.path.exists(SAVE_DIR) and os.listdir(SAVE_DIR):
        print(f'Collecting custom characters from {SAVE_DIR}...')
        for fname in os.listdir(SAVE_DIR):
            if not fname.endswith('.png'): continue
            char = fname.split('__')[0]
            img  = cv2.imread(os.path.join(SAVE_DIR, fname), cv2.IMREAD_GRAYSCALE)
            if img is None: continue


            augmented_imgs = augment_char(img, n=10)

            for aug_img in augmented_imgs:
                feat = extract_hog(aug_img)
                X_all.append(feat)
                y_all.append(char)
            current_chars.add(char)
        print(f'  Collected {len(X_all)} custom samples for {len(current_chars)} characters: {sorted(list(current_chars))}')
    else:
        print(f'  No custom characters found in {SAVE_DIR}. Skipping custom sample collection.')


    print('Downloading + loading EMNIST ByClass (~624 MB) and combining with collected samples...')
    ds = tfds.load('emnist/byclass', split='train', as_supervised=True)

    MAX_PER_CLS_EMNIST = 100
    emnist_counts = {c: 0 for c in ALL_CHARS}

    for img, label in tfds.as_numpy(ds):
        char = label_to_char(label)
        if char not in ALL_CHARS: continue
        if emnist_counts[char] >= MAX_PER_CLS_EMNIST:
            continue

        emnist_img = np.rot90(img.squeeze(), k=3)
        emnist_img = np.fliplr(emnist_img)
        emnist_img = cv2.resize(emnist_img, (CHAR_W, CHAR_H))
        _, emnist_bw = cv2.threshold(emnist_img, 50, 255, cv2.THRESH_BINARY)

        feat = extract_hog(emnist_bw)
        X_all.append(feat)
        y_all.append(char)
        current_chars.add(char)
        emnist_counts[char] += 1

    if not X_all:
        print('No data to train with after collecting custom and EMNIST samples!'); return None

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f'\n  Total samples for training : {len(X_all)}')
    print(f'  Total unique characters    : {len(current_chars)}: {sorted(list(current_chars))}')


    print('Training SVM classifier...')
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                       random_state=42, verbose=False))
    ])
    clf.fit(X_all, y_all)

    bundle = {'clf': clf, 'chars': sorted(list(current_chars))}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'\nModel saved → {MODEL_PATH}')
    print(f'Characters in model ({len(bundle["chars"])})\n    {bundle["chars"]}')
    return bundle


def retrain_svm():
    X, y = [], []
    chars_found = []

    for fname in os.listdir(SAVE_DIR):
        if not fname.endswith('.png'): continue
        char = fname.split('__')[0]
        img  = cv2.imread(os.path.join(SAVE_DIR, fname),
                          cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        augmented = augment_char(img, n=30)
        for aug in augmented:
            feat = extract_hog(aug)
            X.append(feat)
            y.append(char)
        chars_found.append(char)

    if not X:
        print('  Tidak ada data!'); return None

    X = np.array(X)
    y = np.array(y)
    print(f'  Training SVM: {len(X)} sampel, '
          f'{len(set(y))} kelas: {sorted(set(y))}')

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                       random_state=42))
    ])
    clf.fit(X, y)

    bundle = {'clf': clf, 'chars': sorted(set(y))}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'  Model tersimpan → {MODEL_PATH}')
    print(f'  Karakter: {sorted(set(y))}')
    return bundle
