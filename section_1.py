
print("=== SECTION 1: Environment Setup & Folder Structure ===")

import os
import sys
import random
import numpy as np
import psutil
import gc
from datetime import datetime
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

# Google Colab Drive mount
try:
    from google.colab import drive
    IN_COLAB = True
    print("Environment: Google Colab detected.")
except Exception:
    IN_COLAB = False
    print("Environment: Not running in Colab.")

if IN_COLAB:
    DRIVE_MOUNT_POINT = "/content/drive"
    if not os.path.ismount(DRIVE_MOUNT_POINT):
        print("Mounting Google Drive at", DRIVE_MOUNT_POINT)
        drive.mount(DRIVE_MOUNT_POINT)
        print("Google Drive mounted.")
    else:
        print("Google Drive already mounted.")
else:
    DRIVE_MOUNT_POINT = None

# Directories in google drive
BASE_OUTPUT_DIR = os.path.join(DRIVE_MOUNT_POINT, "MyDrive", "v23_full_aug") if IN_COLAB else os.path.abspath("v23_full_aug_local")
DATASET_DIR = os.path.join(BASE_OUTPUT_DIR, "dataset")
RAW_DIR = DATASET_DIR
PROCESSED_DIR = os.path.join(BASE_OUTPUT_DIR, "processed")

FOLDER_PATHS = {
    "base": BASE_OUTPUT_DIR,
    "raw": RAW_DIR,
    "dataset": DATASET_DIR,
    "processed": PROCESSED_DIR,
    "models": os.path.join(BASE_OUTPUT_DIR, "models"),
    "viz": os.path.join(BASE_OUTPUT_DIR, "visualizations"),
    "metrics": os.path.join(BASE_OUTPUT_DIR, "metrics"),
    "clinical_metrics": os.path.join(BASE_OUTPUT_DIR, "clinical_metrics"),
    "aug_samples": os.path.join(BASE_OUTPUT_DIR, "aug_samples")
}

print("Creating directory structure...")
for name, path in FOLDER_PATHS.items():
    os.makedirs(path, exist_ok=True)
    print(f"Directory ready: {name} -> {path}")

# Classes names
CLASS_NAMES = [
    "Tinea Capitis",
    "Tinea Corporis",
    "Tinea Cruris",
    "Tinea Pedis",
    "Tinea Unguium"
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp", ".tif", ".tiff"]

SEED = 42
print(f"Setting random seeds for reproducibility (SEED={SEED})...")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUG_TEMP_DIR = os.path.join(FOLDER_PATHS["base"], "aug_temp")
os.makedirs(AUG_TEMP_DIR, exist_ok=True)

# Training config
BATCH_SIZE = 32
MAX_EPOCHS = 40
LEARNING_RATE = 1e-4
MODEL_NAMES = ["VGG16", "ResNet50V2"]


RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
RUN_MODEL_DIR = os.path.join(FOLDER_PATHS["models"], f"run_{RUN_TIMESTAMP}")
os.makedirs(RUN_MODEL_DIR, exist_ok=True)
print(f"Run timestamp: {RUN_TIMESTAMP}")
print(f"Run directory: {RUN_MODEL_DIR}")


print("Variable checks:")
print(f"  BASE_OUTPUT_DIR    = {BASE_OUTPUT_DIR}")
print(f"  CLASS_NAMES        = {CLASS_NAMES}")
print(f"  NUM_CLASSES        = {NUM_CLASSES}")
print(f"  SEED               = {SEED}")
print(f"  RUN_TIMESTAMP      = {RUN_TIMESTAMP}")
print(f"  MODEL_NAMES        = {MODEL_NAMES}")

print("=== SECTION 1 complete ===\n")

# Make globals available for next sections
GLOBAL_CONFIG = {
    "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
    "RAW_DIR": RAW_DIR,
    "DATASET_DIR": DATASET_DIR,
    "PROCESSED_DIR": PROCESSED_DIR,
    "FOLDER_PATHS": FOLDER_PATHS,
    "CLASS_NAMES": CLASS_NAMES,
    "NUM_CLASSES": NUM_CLASSES,
    "SEED": SEED,
    "MODEL_NAMES": MODEL_NAMES,
    "BATCH_SIZE": BATCH_SIZE,
    "MAX_EPOCHS": MAX_EPOCHS,
    "LEARNING_RATE": LEARNING_RATE,
    "RUN_TIMESTAMP": RUN_TIMESTAMP,
    "RUN_MODEL_DIR": RUN_MODEL_DIR
}
