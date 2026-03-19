"""
train.py - Robust Cats vs Dogs training (handles corrupted images safely)
"""
###
# ----------------------------
# SUPPRESS WARNINGS
# ----------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------------
# IMPORTS
# ----------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
###
# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 12
DATA_DIR = "cats_and_dogs_data/PetImages"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("ROBUST TRAINING STARTED")
print("=" * 60)
###
# ----------------------------
# STRONG DATA CLEANING
# ----------------------------
def clean_dataset(data_dir):
    print("\nCleaning dataset (this may take a bit)...")
    removed = 0

    for cls in ["Cat", "Dog"]:
        cls_path = os.path.join(data_dir, cls)

        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)

            try:
                with Image.open(fpath) as img:
                    img.verify()
            except:
                try:
                    os.remove(fpath)
                    removed += 1
                except:
                    pass

    print(f"Removed {removed} corrupted images")

clean_dataset(DATA_DIR)
###
# ----------------------------
# LOAD DATASET
# ----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)
###
# ----------------------------
# SAFE PIPELINE (CRITICAL)
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ----------------------------
# AUGMENTATION
# ----------------------------
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])
###
# ----------------------------
# MODEL
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    data_aug,
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
###
# ----------------------------
# TRAIN
# ----------------------------
print("\nTraining...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("cat_dog_classifier.h5")
###
# ----------------------------
# PLOTS
# ----------------------------
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["train", "val"])
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "accuracy.png"))

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["train", "val"])
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "loss.png"))

# ----------------------------
# EVALUATION (SAFE)
# ----------------------------
print("\nEvaluating...")

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())
###
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Save predictions
pd.DataFrame({
    "true": y_true,
    "pred": y_pred
}).to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

# ----------------------------
# DONE
# ----------------------------
print("\nTraining complete.")
print("Check the 'results' folder for outputs.")
###