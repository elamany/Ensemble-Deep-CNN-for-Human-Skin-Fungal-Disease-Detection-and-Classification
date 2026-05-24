# Ensemble Deep CNN for Human Skin Fungal Disease Detection and Classification

This project presents an ensemble deep learning framework for automated classification of five common dermatophytosis types:

- Tinea capitis  
- Tinea pedis  
- Tinea unguium  
- Tinea corporis  
- Tinea cruris  

The system combines **VGG16** and **ResNet50V2** using **weighted averaging and stacking** to improve classification accuracy. It includes a full pipeline from preprocessing to real-time inference.

---
## 🚀 Data source
Download dataset from
Roboflow link
https://universe.roboflow.com/eczema-veixh/tinea/dataset/1
kaggle link **link**
https://www.kaggle.com/datasets/shubhamgoel27/dermnet

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
- Applies CLAHE for contrast enhancement 

```python
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img.verify()
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        img.close()
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        #remove blurry image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return None
        #denoise
        img_array = cv2.bilateralFilter(img_array, d=5, sigmaColor=50, sigmaSpace=50)
        #apply CLAHE to enhance low contrast features
        transform = A.Compose([A.CLAHE(clip_limit=1.5, tile_grid_size=(16, 16), always_apply=True)])
        img_array = transform(image=img_array)["image"]
        return img_array
    except Exception as e:
        return None
```

---

## 2.2 Adaptive Image Resizing

- Automatically determines optimal image size  
 

```python
def calculate_average_image_size(image_paths, sample_size=500):
    print("\n[2/5] Calculating optimal image size from cleaned dataset...")
    sampled_paths = image_paths[:min(sample_size, len(image_paths))]
    widths, heights = [], []

    for img_path in sampled_paths:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
        except:
            continue

    avg_width = int(np.mean(widths)) if widths else 320
    avg_height = int(np.mean(heights)) if heights else 320
    avg_size = max(avg_width, avg_height)

    MAX_IMAGE_SIZE = 320
    avg_size = min(avg_size, MAX_IMAGE_SIZE)

    # Select standard size
    standard_sizes = [120, 128, 224, 256, 320]
    selected_size = min(standard_sizes, key=lambda x: abs(x - avg_size))
    selected_size = min(selected_size, MAX_IMAGE_SIZE)

    print(f"  Selected standard size: {selected_size}×{selected_size}")
    return (selected_size, selected_size)
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


## 🧠 Conclusion

This project demonstrates how ensemble deep learning improves medical image classification by combining strong preprocessing, balanced data augmentation, and multi-model learning.
