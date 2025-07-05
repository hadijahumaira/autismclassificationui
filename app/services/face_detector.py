# face_detector.py

import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch

# Inisialisasi MTCNN sekali
mtcnn = MTCNN(keep_all=True)

def detect_face_from_image(image_file):
    # Baca gambar dari file Flask (file-like object)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is not None and probs is not None:
        valid_faces = [(box, prob) for box, prob in zip(boxes, probs) if prob and prob > 0.9]
        if valid_faces:
            img_center = np.array([img_rgb.shape[1] // 2, img_rgb.shape[0] // 2])
            best_box = min(valid_faces, key=lambda b: np.linalg.norm(img_center - [(b[0][0]+b[0][2])/2, (b[0][1]+b[0][3])/2]))[0]
        else:
            best_box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))  # fallback ke wajah terbesar
        x_min, y_min, x_max, y_max = map(int, best_box)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_rgb.shape[1], x_max), min(img_rgb.shape[0], y_max)
        face = img_rgb[y_min:y_max, x_min:x_max]
        return face  # dalam format NumPy RGB
    else:
        return None
