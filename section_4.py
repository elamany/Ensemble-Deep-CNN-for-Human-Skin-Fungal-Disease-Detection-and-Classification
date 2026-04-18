# ==============================================================================
# SECTION 4: Training Pipelines
# ==============================================================================

print("=== SECTION 4: Training Pipelines ===")

import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import json
import pandas as pd
from datetime import datetime
import cv2
import albumentations as A
from PIL import Image

required = ["FOLDER_PATHS", "CLASS_NAMES", "SEED", "GLOBAL_CONFIG", "adamw_optimizer", "create_model"]
missing = [r for r in required if r not in globals()]
if missing:
    raise RuntimeError(f"Required globals missing: {missing}.")

# Get average size ---
average_size = GLOBAL_CONFIG.get('average_size', (320, 320))
print(f"\nTraining models at average image size: 256x25 ")

# --- 4. Hyperparameters ---
TRAINING_CONFIG = {
    "stage1_epochs": 15,        # Stage 1: Train head only (base model frozen)
    "stage2_epochs": 25,        # Stage 2: Fine-tune entire model
    "batch_size": 32,
    "early_stopping_patience": 8,
    "reduce_lr_patience": 5,
    "stage1_lr": 1e-4,          # Higher LR for head training
    "stage2_lr": 2e-5,          # Lower LR for fine-tuning
    "warmup": 3
}

print("\nTraining Configuration (v16):")
for key, value in TRAINING_CONFIG.items():
    print(f"  {key}: {value}")

# Visualization setup ---
def setup_graph_visualizations():
    """Configure Matplotlib for visualizations."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

setup_graph_visualizations()

# Data preparation using pre-split directories ---
def prepare_datasets_from_dirs():
    """Load pre-split datasets from PROCESSED_DIR (train, val, test)."""
    print("\nLoading pre-split datasets from PROCESSED_DIR ...")
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []

    for cls_idx, cls in enumerate(CLASS_NAMES):
        for split, img_list, lbl_list in [("train", train_images, train_labels),
                                          ("val", val_images, val_labels),
                                          ("test", test_images, test_labels)]:
            split_dir = os.path.join(FOLDER_PATHS["processed"], split, cls)
            if os.path.isdir(split_dir):
                for fname in os.listdir(split_dir):
                    if any(fname.lower().endswith(ext) for ext in IMAGE_EXTS):
                        img_list.append(os.path.join(split_dir, fname))
                        lbl_list.append(cls_idx)

    if not train_images or not val_images or not test_images:
        raise RuntimeError(f"Empty dataset detected: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

    print(f"Dataset loaded from directories :")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")

    print("\nClass distribution in splits (v16):")
    for split_name, labels in [("Training", train_labels), ("Validation", val_labels), ("Test", test_labels)]:
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        print(f"  {split_name}: {dict(zip(CLASS_NAMES, counts))}")

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = prepare_datasets_from_dirs()

batch_size = TRAINING_CONFIG["batch_size"]
print(f"\nCreating data generators with batch size: {batch_size}")

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_paths, labels, batch_size, target_size, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(image_paths))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_size_actual = len(batch_indices)
        batch_images = np.zeros((batch_size_actual, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        batch_labels = np.zeros((batch_size_actual,), dtype=np.int32)

        for i, batch_idx in enumerate(batch_indices):
            try:
                img_path = self.image_paths[batch_idx]
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)

                # Ensure correct shape
                img_out = transformed['image']
                if img_out.shape != (self.target_size[0], self.target_size[1], 3):
                    # Fix shape inconsistencies
                    img_out = cv2.resize(img_out, (self.target_size[1], self.target_size[0]))
                    if len(img_out.shape) == 2:
                        img_out = np.stack([img_out] * 3, axis=-1)
                    elif img_out.shape[2] == 4:
                        img_out = img_out[:, :, :3]

                batch_images[i] = img_out.astype(np.float32)
                batch_labels[i] = self.labels[batch_idx]
            except Exception as e:
                print(f"--  Error loading preprocessed image {img_path}: {e}")
                # Fallback to zero image
                batch_images[i] = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
                batch_labels[i] = self.labels[batch_idx]

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Create generators
train_generator = DataGenerator(
    train_images, train_labels, batch_size=batch_size, target_size=average_size,shuffle=True
)

val_generator = DataGenerator(
    val_images, val_labels, batch_size=batch_size, target_size=average_size, shuffle=False
)

test_generator = DataGenerator(
    test_images, test_labels, batch_size=batch_size, target_size=average_size,
    augment=False, shuffle=False
)

# Training callbacks ---
def create_training_callbacks(model_name):
    """Create callbacks for training with detailed logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(FOLDER_PATHS["models"], f"{model_name}_{timestamp}_v16")
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=TRAINING_CONFIG["early_stopping_patience"],
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=TRAINING_CONFIG["reduce_lr_patience"],
            min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, f"best_{model_name}_{average_size[0]}x{average_size[1]}_v16.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(model_dir, f"training_history_{model_name}_{average_size[0]}x{average_size[1]}_v16.csv")
        ),
        tf.keras.callbacks.TerminateOnNaN()  # Prevent training from continuing with NaN values
    ]

    callbacks.append(tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: TRAINING_CONFIG["stage1_lr"] * min(1, epoch/TRAINING_CONFIG['warmup']) if epoch < TRAINING_CONFIG['warmup'] else TRAINING_CONFIG["stage1_lr"]
    ))

    return callbacks, model_dir

# 10. Training visualization functions ---
def plot_learning_curves(history, model_name, save_path):
    """Plot training and validation accuracy/loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #fig.suptitle(f'Learning Curves: {model_name} ({average_size[0]}x{average_size[1]}) (v16)', fontsize=16, fontweight='bold')

    # Accuracy plot
    if 'accuracy' in history.history:
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#1f77b4')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#ff7f0e')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Loss plot
    if 'loss' in history.history:
        axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#1f77b4')
        axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved: {save_path}")

# Model training function ---
def train_model_progressive(model, base_model, model_name):
    print(f"\n{'#'*60}")
    print(f"-------> TRAINING {model_name} MODEL on 256x256 (v16)")
    print(f"{'#'*60}")

    start_time = time.time()

    print("   Stage 1: Training classification head (base model frozen) (v16)")
    base_model.trainable = False

    model.compile(
        optimizer=adamw_optimizer(learning_rate=TRAINING_CONFIG["stage1_lr"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks, model_dir = create_training_callbacks(model_name)

    history_stage1 = model.fit(
        train_generator,
        epochs=TRAINING_CONFIG["stage1_epochs"],
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    best_val_acc_stage1 = max(history_stage1.history['val_accuracy'])
    print(f"   Best Val Acc in Stage 1 : {best_val_acc_stage1:.4f}")

    # Plot Stage 1 learning curves
    plot_learning_curves(history_stage1, f"{model_name}_Stage1_v16",
                        os.path.join(FOLDER_PATHS["viz"], f"learning_curves_{model_name}_stage1_{average_size[0]}x{average_size[1]}_v16.png"))


    print("   Stage 2: Fine-tuning entire model (base model unfrozen) ")
    base_model.trainable = True

    callbacks_stage2 = callbacks.copy()
    callbacks_stage2.append(tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: TRAINING_CONFIG["stage2_lr"] * min(1, epoch/TRAINING_CONFIG['warmup']) if epoch < TRAINING_CONFIG['warmup'] else TRAINING_CONFIG["stage2_lr"]
    ))

    model.compile(
        optimizer=adamw_optimizer(learning_rate=TRAINING_CONFIG["stage2_lr"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_stage2 = model.fit(
        train_generator,
        epochs=TRAINING_CONFIG["stage2_epochs"],
        validation_data=val_generator,
        callbacks=callbacks_stage2,
        verbose=1
    )

    best_val_acc_stage2 = max(history_stage2.history['val_accuracy'])
    print(f"   Best Val Acc in Stage 2 : {best_val_acc_stage2:.4f}")

    # Plot Stage 2 learning curves
    plot_learning_curves(history_stage2, f"{model_name}_Stage2_v16",
                        os.path.join(FOLDER_PATHS["viz"], f"learning_curves_{model_name}_stage2_{average_size[0]}x{average_size[1]}_v16.png"))

    # Combine histories for final plot
    combined_history = tf.keras.callbacks.History()
    combined_history.history = {}
    for key in history_stage1.history.keys():
        combined_history.history[key] = history_stage1.history[key] + history_stage2.history[key]

    # Plot combined learning curves
    plot_learning_curves(combined_history, f"{model_name}_Combined_v16",
                        os.path.join(FOLDER_PATHS["viz"], f"learning_curves_{model_name}_combined_{average_size[0]}x{average_size[1]}_v16.png"))

    # Save final model
    final_model_path = os.path.join(model_dir, f"final_{model_name}_{average_size[0]}x{average_size[1]}_v16.keras")
    model.save(final_model_path)
    print(f"Final model saved to : {final_model_path}")

    # Calculate final training accuracy
    final_train_acc = combined_history.history['accuracy'][-1]
    best_val_acc = max(best_val_acc_stage1, best_val_acc_stage2)

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes (v16)")

    return model, combined_history, model_dir, best_val_acc, final_train_acc

# --- 12. Train models ---
print("\n" + "="*80)
print(f"--- STARTING MODEL TRAINING AT RESOLUTION: {average_size[0]}×{average_size[1]} (v16) ---")
print("="*80)

backbones = ["VGG16", "ResNet50V2"]
trained_models = {}
model_histories = {}
model_dirs = {}
validation_accuracies = {}
training_accuracies = {}

for model_name in backbones:
    try:
        # Create model
        model, base_model = create_model(
            base_model_name=model_name,
            input_shape=(average_size[0], average_size[1], 3),
            num_classes=len(CLASS_NAMES)
        )

        # Train model
        trained_model, history, model_dir, best_val_acc, final_train_acc = train_model_progressive(
            model, base_model, model_name
        )

        # Store results
        trained_models[model_name] = trained_model
        model_histories[model_name] = history
        model_dirs[model_name] = model_dir
        validation_accuracies[model_name] = best_val_acc
        training_accuracies[model_name] = final_train_acc

        print(f"\n {model_name} training completed successfully ")
        print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"  Final Training Accuracy: {final_train_acc:.4f}")

        # Clear session to free memory between models
        tf.keras.backend.clear_session()

    except Exception as e:
        print(f"\n Training failed for {model_name}: {e}")
        continue

# --- 13. Save training results ---
training_results_path = os.path.join(FOLDER_PATHS["clinical_metrics"], "training_results_v16.json")
training_results = {
    "resolution": average_size,
    "models_trained": list(trained_models.keys()),
    "validation_accuracies": {name: float(acc) for name, acc in validation_accuracies.items()},
    "training_accuracies": {name: float(acc) for name, acc in training_accuracies.items()},
    "model_dirs": model_dirs,
    "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
    "config": TRAINING_CONFIG
}

# Convert numpy types to Python native types for JSON serialization
def convert_to_python_types(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj

training_results = convert_to_python_types(training_results)
with open(training_results_path, 'w') as f:
    json.dump(training_results, f, indent=4)
print(f"\nTraining results saved: {training_results_path}")

# --- 14. Final summary ---
print("\n" + "="*80)
print("TRAINING SUMMARY (v16)")
print("="*80)
print(f"Resolution used: 256x256 pixels")
print(f"Models trained: {', '.join(trained_models.keys())}")
print("\nBest Validation Accuracies:")
for name, acc in validation_accuracies.items():
    print(f"  {name}: {acc:.4f}")

# Make results available for next sections
GLOBAL_CONFIG['trained_models'] = trained_models
GLOBAL_CONFIG['model_histories'] = model_histories
GLOBAL_CONFIG['model_dirs'] = model_dirs
GLOBAL_CONFIG['validation_accuracies'] = validation_accuracies
GLOBAL_CONFIG['training_accuracies'] = training_accuracies
GLOBAL_CONFIG['test_generator'] = test_generator
GLOBAL_CONFIG['val_generator'] = val_generator
GLOBAL_CONFIG['test_images'] = test_images
GLOBAL_CONFIG['test_labels'] = test_labels

print("\n=== SECTION 4 completed: Model training finished ===\n")
