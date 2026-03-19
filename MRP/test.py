"""
test.py - Fixed inference script
"""
###
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import sys
from PIL import Image
import pandas as pd

IMG_SIZE = (160, 160)
MODEL_PATH = "cat_dog_classifier.h5"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        sys.exit(1)
    return tf.keras.models.load_model(MODEL_PATH)
###
def preprocess(path):
    try:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize(IMG_SIZE)

        # FIX: removed /255 (model already rescales)
        arr = np.array(img)

        return np.expand_dims(arr, axis=0)

    except Exception as e:
        print(f"Error: {e}")
        return None

def predict(model, path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    arr = preprocess(path)
    if arr is None:
        return None

    pred = model.predict(arr, verbose=0)[0][0]

    label = "Dog" if pred > 0.5 else "Cat"
    conf = pred * 100 if pred > 0.5 else (1 - pred) * 100
###
    print(f"{os.path.basename(path)} -> {label} ({conf:.2f}%)")

    return {
        "file": os.path.basename(path),
        "prediction": label,
        "confidence": conf,
        "raw_score": float(pred)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py image.jpg")
        sys.exit(1)

    model = load_model()
    res = predict(model, sys.argv[1])

    if res:
        pd.DataFrame([res]).to_csv(
            os.path.join(RESULTS_DIR, "single_prediction.csv"),
            index=False
        )
###