# ==============================================================================
# SECTION 6: Real-Time Tinea Classification Demo
# ==============================================================================

print("=== SECTION 6: Real-Time Tinea Classification Demo ===")

# --- 1. Import necessary libraries ---
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import json
import warnings
from sklearn.linear_model import LogisticRegression
from urllib.request import urlopen
from io import BytesIO
import albumentations as A
import cv2

# Suppress warnings for cleaner demo
warnings.filterwarnings("ignore")

# --- 2. Environment setup ---
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
    DRIVE_MOUNT_POINT = None

# Set up directory paths
BASE_OUTPUT_DIR = os.path.join(DRIVE_MOUNT_POINT, "MyDrive", "v23_full_aug") if IN_COLAB else os.path.abspath("v23_full_aug_local")
MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models")
CLINICAL_METRICS_DIR = os.path.join(BASE_OUTPUT_DIR, "clinical_metrics")
VIZ_DIR = os.path.join(BASE_OUTPUT_DIR, "visualizations")

print("\nDirectory structure:")
print(f"  Models Directory: {MODELS_DIR}")
print(f"  Clinical Metrics Directory: {CLINICAL_METRICS_DIR}")

# --- Class Constants ---
CLASS_NAMES = [
    "Tinea Capitis",
    "Tinea Corporis",
    "Tinea Cruris",
    "Tinea Pedis",
    "Tinea Unguium"
]
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
resolution=[320,320]

# --- Helper functions ---
def load_best_ensemble_config():
    """Load the best ensemble configuration"""
    try:
        # Find latest ensemble config
        ensemble_files = [f for f in os.listdir(CLINICAL_METRICS_DIR)
                         if f.startswith('ensemble_config_') and f.endswith('.json')]
        if not ensemble_files:
            print("No ensemble configuration found")
            return None

        ensemble_files.sort(reverse=True)
        latest_ensemble = ensemble_files[0]
        ensemble_path = os.path.join(CLINICAL_METRICS_DIR, latest_ensemble)

        print(f"Loading latest ensemble config: {latest_ensemble}")
        with open(ensemble_path, 'r') as f:
            return json.load(f)

    except Exception as e:
        print(f"Error loading ensemble configuration: {e}")
        return None

def find_model_directory(model_name):
    """Find the directory that starts with the model name in the models directory"""
    try:
        # List all directories in the models directory
        all_items = os.listdir(MODELS_DIR)
        model_dirs = [d for d in all_items
                     if os.path.isdir(os.path.join(MODELS_DIR, d)) and d.startswith(model_name)]

        if not model_dirs:
            print(f"No directories found starting with '{model_name}' in {MODELS_DIR}")
            print(f"   Available directories: {', '.join([d for d in all_items if os.path.isdir(os.path.join(MODELS_DIR, d))])}")
            return None

        # Sort by timestamp (newest first)
        model_dirs.sort(reverse=True)
        latest_dir = model_dirs[0]
        dir_path = os.path.join(MODELS_DIR, latest_dir)
        print(f"Found {model_name} directory: {latest_dir}")
        return dir_path

    except Exception as e:
        print(f"Error finding {model_name} directory: {e}")
        return None

def find_model_file(model_dir, model_name, resolution_label):
    try:
        # List all .keras files in the model directory
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]

        if not model_files:
            print(f"No .keras files found in {model_dir}")
            return None

        # Look for files with the pattern "final_{model_name}_{resolution_label}.keras"
        pattern = f"final_{model_name}_{resolution_label}.keras"
        matching_files = [f for f in model_files if f.startswith(pattern.split('.')[0])]

        if not matching_files:
            # Try broader pattern
            pattern = f"{model_name}_{resolution_label}.keras"
            matching_files = [f for f in model_files if f.startswith(pattern.split('.')[0])]

        if not matching_files:
            print(f"No matching model files found for {model_name} at {resolution_label} in {model_dir}")
            print(f"   Searched for files starting with: {pattern.split('.')[0]}")
            print(f"   Available files: {', '.join(model_files)}")
            return None

        # Get the most recent matching file
        matching_files.sort(reverse=True)
        return matching_files[0]

    except Exception as e:
        print(f"Error finding model file in {model_dir}: {e}")
        return None

def load_models_for_ensemble(ensemble_config):
    """Load all models required for the ensemble - DYNAMIC DIRECTORY HANDLING"""
    try:
        models_in_ensemble = ensemble_config['models_in_ensemble']
        resolution = ensemble_config['resolution']
        resolution_label = f"{resolution[0]}x{resolution[1]}"

        print(f"\nLoading models for ensemble at {resolution_label}...")
        loaded_models = {}
        model_directories = {}

        # First find all model directories
        for model_name in models_in_ensemble:
            model_dir = find_model_directory(model_name)
            if model_dir is None:
                continue
            model_directories[model_name] = model_dir

        # Now load models from their directories
        for model_name, model_dir in model_directories.items():
            model_file = find_model_file(model_dir, model_name, resolution_label)
            if model_file is None:
                continue

            model_path = os.path.join(model_dir, model_file)

            print(f"   Loading {model_name} from: {model_file}")
            model = load_model(model_path, compile=False)
            loaded_models[model_name] = model
            print(f"   {model_name} loaded successfully")

        if len(loaded_models) == 0:
            print("Failed to load any models for ensemble")
            return None

        return loaded_models

    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_stacking_meta_model(ensemble_config):
    """Load stacking meta-model parameters"""
    try:
        if 'stacking' not in ensemble_config:
            print("  No stacking parameters found in ensemble config")
            return None

        stacking_params = ensemble_config['stacking']
        meta_model_params = stacking_params.get('meta_model_params', {})

        if not meta_model_params:
            print("  No meta-model parameters found")
            return None

        # Reconstruct logistic regression meta-model
        meta_model = LogisticRegression(max_iter=1000, random_state=SEED)

        # Create dummy data to initialize model
        dummy_X = np.zeros((2, NUM_CLASSES * len(ensemble_config['models_in_ensemble'])))
        dummy_y = [0, 1]
        meta_model.fit(dummy_X, dummy_y)

        # Load real parameters
        meta_model.coef_ = np.array(meta_model_params['coefficients'])
        meta_model.intercept_ = np.array(meta_model_params['intercept'])
        meta_model.classes_ = np.array(meta_model_params['classes'])

        return meta_model

    except Exception as e:
        print(f"Error loading stacking meta-model: {e}")
        return None

def preprocess_images_for_inference(img, target_size):
    try:
        img = img.convert("RGB")
        x = np.array(img)
        transform = A.Compose([A.Resize(height=256, width=256),A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        transformed = transform(image=x)
        processed_img = transformed['image']
        return np.expand_dims(processed_img, axis=0), processed_img
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None

def run_ensemble_prediction(models, ensemble_config, img_array):
    """Run ensemble prediction using the configured method"""
    try:
        best_ensemble = ensemble_config['best_ensemble']
        resolution = [256,256] #ensemble_config['resolution']

        # Get predictions from each model
        predictions = []

        for model_name, model in models.items():
            pred = model.predict(img_array, verbose=0)[0]
            predictions.append(pred)

        # Weighted Average Ensemble
        if best_ensemble == "Weighted Average":
            weights = ensemble_config['weighted_average']['weights']

            weighted_pred = np.zeros((NUM_CLASSES,))
            for i, (model_name, pred) in enumerate(zip(ensemble_config['models_in_ensemble'], predictions)):
                weight = weights.get(model_name, 1.0 / len(models))
                weighted_pred += weight * pred

            final_pred = weighted_pred
            confidence = np.max(final_pred)
            pred_class = np.argmax(final_pred)
            return CLASS_NAMES[pred_class], confidence * 100, final_pred, best_ensemble

        # Stacking Ensemble
        elif best_ensemble == "Stacking":
            # Load meta-model
            meta_model = load_stacking_meta_model(ensemble_config)
            if meta_model is None:
                print("  Falling back to Weighted Average (stacking meta-model not available)")
                return run_weighted_average_fallback(models, ensemble_config, img_array)

            # Prepare features for meta-model
            test_features = np.concatenate(predictions).reshape(1, -1)
            stacking_proba = meta_model.predict_proba(test_features)[0]

            confidence = np.max(stacking_proba)
            pred_class = np.argmax(stacking_proba)
            return CLASS_NAMES[pred_class], confidence * 100, stacking_proba, best_ensemble

        else:
            print(f"  Unknown ensemble method: {best_ensemble}. Using Weighted Average as fallback.")
            return run_weighted_average_fallback(models, ensemble_config, img_array)

    except Exception as e:
        print(f"Ensemble prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to simple average
        avg_pred = np.mean(predictions, axis=0)
        confidence = np.max(avg_pred)
        pred_class = np.argmax(avg_pred)
        return CLASS_NAMES[pred_class], confidence * 100, avg_pred, "Simple Average"

def run_weighted_average_fallback(models, ensemble_config, img_array):
    """Fallback to weighted average if other methods fail"""
    try:
        weights = ensemble_config.get('weighted_average', {}).get('weights', {})
        if not weights:
            # Use uniform weights if no weights available
            weights = {name: 1.0/len(models) for name in models.keys()}

        predictions = []

        for model_name, model in models.items():
            pred = model.predict(img_array, verbose=0)[0]
            predictions.append(pred)

        weighted_pred = np.zeros((NUM_CLASSES,))
        for i, (model_name, pred) in enumerate(zip(ensemble_config['models_in_ensemble'], predictions)):
            weight = weights.get(model_name, 1.0 / len(models))
            weighted_pred += weight * pred

        final_pred = weighted_pred
        confidence = np.max(final_pred)
        pred_class = np.argmax(final_pred)
        return CLASS_NAMES[pred_class], confidence * 100, final_pred, "Weighted Average (Fallback)"

    except Exception as e:
        print(f"Weighted average fallback error: {e}")
        avg_pred = np.mean(predictions, axis=0)
        confidence = np.max(avg_pred)
        pred_class = np.argmax(avg_pred)
        return CLASS_NAMES[pred_class], confidence * 100, avg_pred, "Simple Average"

def display_image(img):
    """Display image in a pixel box for clear visualization"""
    plt.figure(figsize=(3.5, 3.5), dpi=100)  # 3.5 inches * 100 DPI = 350 pixels
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image', fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.show()

def preprocess_image_for_inference(img, target_size):
    try:
        img = img.convert("RGB")
        x = np.array(img)
        #resize and normalize
        transform = A.Compose([
            A.Resize(height=320, width=320),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        transformed = transform(image=x)
        processed_img = transformed['image']
        return np.expand_dims(processed_img, axis=0), processed_img

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None

def main_demo():
    """Main interactive demo function"""
    print("\n" + "="*60)
    print("REAL TIME TINEA CLASSIFICATION DEMO (v16)")
    print("="*60)

    # Step 1: Load ensemble configuration
    print("\nStep 1: Loading best ensemble configuration...")

    ensemble_config = load_best_ensemble_config()
    if ensemble_config is None:
        print("Cannot proceed without ensemble configuration")
        return

    # List all directories in the models directory
    print("\nDirectory structure in models directory:")
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path):
            print(f"   {item}/")
        else:
            print(f"   {item}")

    print(f"Loaded ensemble configuration. Resolution: {ensemble_config['resolution']}")
    print(f"   Best ensemble: {ensemble_config['best_ensemble']}")
    print(f"   Models in ensemble: {ensemble_config['models_in_ensemble']}")

    models = load_models_for_ensemble(ensemble_config)
    if models is None:
        print("Cannot proceed without loaded models")
        return

    resolution = ensemble_config['resolution']
    best_ensemble = ensemble_config['best_ensemble']

    print(f"\nSystem ready for inference at {resolution[0]}x{resolution[1]} resolution")
    print(f"   Using {best_ensemble} ensemble with {len(models)} models")

    print(f"\nChoose Input Method")
    print("  [1] Upload local image file")
    print("  [2] Enter public image URL")
    input_choice = input("\nEnter choice (1 or 2): ").strip()

    # Load image
    img_path = None
    if input_choice == "1":
        try:
            if IN_COLAB:
                from google.colab import files
                print("Upload an image...")
                uploaded = files.upload()
                img_path = list(uploaded.keys())[0]
                img = Image.open(img_path)
            else:
                path = input("Enter local image path: ").strip()
                img_path = path
                img = Image.open(path)
            print(f"Image loaded successfully: {img.size}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return
    elif input_choice == "2":
        try:
            url = input("Enter image URL: ").strip()
            img_path = url
            print("Downloading image...")
            img = Image.open(BytesIO(urlopen(url).read()))
            print(f"Image downloaded successfully: {img.size}")
        except Exception as e:
            print(f"Error downloading image: {e}")
            return
    else:
        print("Invalid choice")
        return

    display_image(img)
    img_array, processed_img = preprocess_images_for_inference(img, resolution)
    if img_array is None:
        print("Preprocessing failed")
        return

    print(f"\n Running inference with {best_ensemble} ensemble...")
    try:
        pred_class, conf, probs, actual_ensemble = run_ensemble_prediction(
            models, ensemble_config, img_array
        )
        print(f"Inference completed successfully")
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    if conf < 70:
        display_class = "Low-confidence / Unknown"
    else:
        display_class = pred_class

    # Step 7: Display results in console
    print("\n" + "="*60)
    print("PREDICTION RESULTS ")
    print("="*60)
    print(f"Ensemble Method: {actual_ensemble}")
    print(f"Image Resolution: {resolution[0]}x{resolution[1]}")
    print(f"Prediction: {display_class}")
    print(f"Confidence: {conf:.1f}%")

    if conf < 70:
        print("\nWarning: Low confidence prediction.")

    print("\nClass Probabilities:")
    print("-" * 40)

    # Sort probabilities by value (highest to lowest)
    sorted_indices = np.argsort(probs)[::-1]
    for i, idx in enumerate(sorted_indices):
        cls = CLASS_NAMES[idx]
        p = probs[idx] * 100
        star = ""
        if cls == pred_class:
            star = "*"
        print(f"{i+1}. {cls:<18} | {p:5.1f}% {star}")

    if conf >= 70:
        print("* = Predicted class")

    print("-" * 40)
    if 'per_model_accuracies' in ensemble_config:
        for model_name, accuracy in ensemble_config['per_model_accuracies'].items():
            print(f"  {model_name}: {accuracy:.4f}")

    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main_demo()
    except KeyboardInterrupt:
        print("\n\nExited.")
    except Exception as e:
        print(f"\nUnexpected error during demo: {e}")
        import traceback
        traceback.print_exc()

print("\n=== SECTION 6 completed: Real-time tinea classification demo finished ===\n")
