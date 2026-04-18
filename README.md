# Ensemble Deep CNN for Human Skin Fungal Disease Detection and Classification
Ensemble deep learning for skin fungal disease classification using VGG16 and ResNet50V2. Combines models via weighted averaging and stacking to improve accuracy. Includes data preprocessing, training, and evaluation scripts for automated diagnosis.

This project presents an ensemble deep learning framework for automated classification of five common dermatophytosis types:

- Tinea capitis  
- Tinea pedis  
- Tinea unguium  
- Tinea corporis  
- Tinea cruris  

The system combines **VGG16** and **ResNet50V2** using **weighted averaging and stacking** to improve classification accuracy. It includes a full pipeline from preprocessing to real-time inference.

---

## 🚀 How to Run

Paste the code into **Google Colab (Pro recommended)** and execute **section by section**.

---

# 1. Initialization

## 1.1 Environment Detection and Google Drive Mount

```python
import os

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
```

---

## 1.2 Directory Structure Setup

```python
# Directories in Google Drive
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
```

---

## 📂 Folder Structure

```
v23_full_aug/
│
├── dataset/                # Raw dataset
├── processed/              # Cleaned + resized images
├── models/                 # Saved models
├── visualizations/         # Plots and Grad-CAM
├── metrics/                # Evaluation metrics
├── clinical_metrics/       # Statistical test outputs
├── aug_samples/            # Augmented samples
```

---

# 2. Image Preprocessing

## 2.1 Data Cleaning

To ensure dataset quality, the following automated techniques are applied:

- File integrity checks to remove corrupted images  
- Laplacian variance to detect and remove blurry images  
- Bilateral filtering to reduce noise  

```python
import cv2

THRESHOLD = 100

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < THRESHOLD
```

---

## 2.2 Adaptive Image Resizing + CLAHE

- Automatically determines optimal image size  
- Applies CLAHE for contrast enhancement  

```python
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
```

---

## 2.3 Data Augmentation + Class Balancing

### Techniques
- Random rotation  
- Shift and scaling  
- Horizontal and vertical flip  
- Random brightness and contrast  

### Controlled Repetition Strategy
- Prevents uneven repetition  
- Avoids overfitting to specific images  
- Ensures balanced augmentation  

---

## 2.4 Dataset Splitting

- Train: 70%  
- Validation: 15%  
- Test: 15%  

```python
from sklearn.model_selection import train_test_split
```

---

# 3. Model Definitions (VGG16 + ResNet50V2)

## 3.1 Custom Heads

Each model includes:

- GlobalAveragePooling2D  
- BatchNormalization  
- Dropout  
- ReLU activation  

```python
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
```

---

## 3.2 Base Models

```python
from tensorflow.keras.applications import VGG16, ResNet50V2
```

---

## 3.3 Ensemble Strategy

- Weighted averaging  
- Stacking  

---

# 4. Model Training Pipeline

## Hyperparameters

| Parameter | Value |
|----------|------|
| Epochs | Stage 1: 15 (Frozen) |
|         | Stage 2: 25 (Unfrozen) |
| Total Epochs | 40 |
| Batch Size | 32 |
| Learning Rate | Stage 1: 1e-4 |
|              | Stage 2: 2e-5 |
| Learning Rate Reduction | 5 |
| Early Stopping | 8 |
| Warmup Epochs | 3 |
| Optimizer | AdamW |

---

## Training Strategy

### Stage 1
- Freeze base models  
- Train custom head  

### Stage 2
- Unfreeze models  
- Fine-tune entire network  

---

# 5. Model Evaluation

## Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

## Ensemble Evaluation
- Compare individual vs ensemble performance  

## Statistical Testing

### McNemar’s Test
Used for statistical comparison of models.

```python
from statsmodels.stats.contingency_tables import mcnemar
```

---

# 6. Real-Time Demo

```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('ensemble_model.h5')

# preprocess image
# prediction
```

---

## 📌 Key Features

- Robust data cleaning pipeline  
- Adaptive preprocessing (CLAHE + resizing)  
- Balanced augmentation strategy  
- Ensemble learning (VGG16 + ResNet50V2)  
- Two-stage training to reduce overfitting  
- Statistical validation (McNemar’s test)  
- Ready for real-time deployment  

---

## 🔗 Related Repositories

This project is part of a complete end-to-end system including backend APIs and a mobile application:

- 🌐 **Flask Backend API**  
  Handles model inference, request processing, and deployment logic  
  👉 https://github.com/your-username/flask-fungal-api  

- 📱 **Flutter Mobile App**  
  User interface for real-time skin disease detection  
  👉 https://github.com/your-username/fungal-detection-app  

- 🧠 **Model Training (This Repository)**  
  Deep learning pipeline, preprocessing, training, and evaluation  

## 📊 Future Improvements

- Add more dermatological conditions  
- Integrate Grad-CAM for explainability  
---

## 🧠 Conclusion

This project demonstrates how ensemble deep learning improves medical image classification by combining strong preprocessing, balanced data augmentation, and multi-model learning.
