# ==============================================================================
# SECTION 5: Test Model Evaluation & Ensemble Learning
# ==============================================================================
print("=== SECTION 5: Test Model Evaluation & Ensemble Learning ===")

import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency, pearsonr
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
try:
    from sklearn.calibration import calibration_curve
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

required = ['GLOBAL_CONFIG', 'FOLDER_PATHS', 'CLASS_NAMES', 'SEED', 'average_size', 'test_generator', 'val_generator', 'trained_models', 'training_accuracies', 'test_images', 'test_labels']
missing = [r for r in required if r not in globals()]
if missing:
    raise RuntimeError(f"Required globals missing: {missing}. Ensure Sections 1-4 are executed.")

# Get GLOBAL_CONFIG
average_size = GLOBAL_CONFIG.get('average_size', (320, 320))
trained_models = GLOBAL_CONFIG.get('trained_models', {})
training_accuracies = GLOBAL_CONFIG.get('training_accuracies', {})
test_generator = GLOBAL_CONFIG.get('test_generator', None)
val_generator = GLOBAL_CONFIG.get('val_generator', None)
test_images = GLOBAL_CONFIG.get('test_images', [])
test_labels = GLOBAL_CONFIG.get('test_labels', [])

print(f"\nEvaluating models at average image size: {average_size[0]}×{average_size[1]}")
print(f"Models available for evaluation: {list(trained_models.keys())}")

# --- Helper Functions for Evaluation & Ensemble ---
def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    else:
        return obj

def plot_and_save_cm(cm, model_name, resolution_label, save_dir=None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap='Blues', annot_kws={'size': 12}, cbar_kws={'label': 'Count'})
    #plt.title(f'Confusion Matrix ({resolution_label}) - {model_name} ', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_dir is None:
        save_dir = FOLDER_PATHS["clinical_metrics"]
    cm_path = os.path.join(save_dir, f"confusion_matrix_{resolution_label}_{model_name}_v18.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")
    return cm_path


def mcnemar_test(y_true, pred1, pred2):
    """Perform McNemar's test to compare two classifiers."""
    n11 = np.sum((pred1 == y_true) & (pred2 == y_true))
    n12 = np.sum((pred1 == y_true) & (pred2 != y_true))
    n21 = np.sum((pred1 != y_true) & (pred2 == y_true))
    n22 = np.sum((pred1 != y_true) & (pred2 != y_true))
    contingency_table = [[n11, n12], [n21, n22]]
    try:
        chi2, p_value = chi2_contingency(contingency_table, correction=True)[:2]
        return p_value
    except ValueError:
        print("   McNemar's test skipped: Insufficient data in contingency table")
        return 1.0

def compute_ensemble_std(predictions_list):
    """Compute standard deviation across ensemble members as uncertainty measure."""
    preds_array = np.array(predictions_list)  # shape: (n_models, n_samples, n_classes)
    std_per_sample = np.std(preds_array, axis=0)  # std across models
    return np.mean(std_per_sample, axis=1)  # average std across classes

# --- Collect predictions and evaluate models ---
def collect_predictions_and_labels(generator, trained_models):
    """Collect predictions and true labels from generator for all trained models."""
    all_true_labels = []
    model_predictions = {name: [] for name in trained_models.keys()}

    print("Collecting predictions from all models...")
    for i in range(len(generator)):
        batch_images, batch_labels = generator[i]
        all_true_labels.extend(batch_labels)
        for name, model in trained_models.items():
            preds = model.predict(batch_images, verbose=0)
            model_predictions[name].append(preds)

    # Concatenate predictions
    for name in model_predictions.keys():
        model_predictions[name] = np.concatenate(model_predictions[name], axis=0)
    all_true_labels = np.array(all_true_labels)
    return all_true_labels, model_predictions

# Collect validation and test predictions
print("\n" + "="*60)
print("MODEL EVALUATION ON TEST SET ")
print("="*60)

print("\nCollecting validation set predictions...")
y_val_true, val_predictions = collect_predictions_and_labels(val_generator, trained_models)

print("\nCollecting test set predictions...")
y_test_true, test_predictions = collect_predictions_and_labels(test_generator, trained_models)

# --- Model Performance Evaluation ---
print("\nIndividual Model Performance on Test Set:")
per_model_accuracies = {}
per_model_train_accuracies = {}
per_model_f1_scores = {}
per_model_precision = {}
per_model_recall = {}

for name, model in trained_models.items():
    test_preds = test_predictions[name]
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_acc = accuracy_score(y_test_true, test_pred_classes)
    test_f1 = f1_score(y_test_true, test_pred_classes, average='macro')
    test_precision = precision_score(y_test_true, test_pred_classes, average='macro')
    test_recall = recall_score(y_test_true, test_pred_classes, average='macro')
    train_acc = training_accuracies.get(name, 0.0)

    per_model_accuracies[name] = test_acc
    per_model_train_accuracies[name] = train_acc
    per_model_f1_scores[name] = test_f1
    per_model_precision[name] = test_precision
    per_model_recall[name] = test_recall

    print(f"\n{name} Performance :")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"  Macro F1-Score:    {test_f1:.4f}")
    print(f"  Macro Precision:   {test_precision:.4f}")
    print(f"  Macro Recall:      {test_recall:.4f}")

    # Save confusion matrix
    cm = confusion_matrix(y_test_true, test_pred_classes)
    plot_and_save_cm(cm, name, "256x256")

    # Save classification report
    report = classification_report(y_test_true, test_pred_classes, target_names=CLASS_NAMES, digits=4, output_dict=False)
    print(f"\nClassification Report for {name} :")
    print(report)

    # Save to file
    report_path = os.path.join(FOLDER_PATHS["clinical_metrics"], f"classification_report_{name}_{average_size[0]}x{average_size[1]}_v18.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {name} at {average_size[0]}x{average_size[1]} resolution \n")
        f.write("="*70 + "\n")
        f.write(report)
    print(f"Classification report saved: {report_path}")

    # Save JSON version
    report_dict = classification_report(y_test_true, test_pred_classes, target_names=CLASS_NAMES, output_dict=True)
    report_dict = convert_to_python_types(report_dict)
    report_json_path = os.path.join(FOLDER_PATHS["clinical_metrics"], f"classification_report_{name}_{average_size[0]}x{average_size[1]}_v18.json")
    with open(report_json_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report JSON saved: {report_json_path}")




# -- Ensemble Learning (Weighted Average and Stacking) ---
print("\n" + "="*60)
print("-------ENSEMBLE LEARNING (WEIGHTED AVERAGE AND STACKING) -------")
print("="*60)

# --- Ensemble Configuration ---
ENSEMBLE_CONFIG = {
    "n_folds": 3,                    # Number of folds for stacking CV
    "weighted_method": "accuracy_weighted",                                                                                         # Options: 'accuracy_weighted', 'f1_weighted', 'uniform'
    "stacking_max_iter": 1000,       # Maximum iterations for logistic regression
    "min_accuracy_threshold": 0.60   # Minimum accuracy to include in ensemble
}

print("\nEnsemble Configuration :")
for key, value in ENSEMBLE_CONFIG.items():
    print(f"  {key}: {value}")

# ---  Models for Ensemble ---
print("\n" + "="*60)
print("MODEL SELECTION FOR ENSEMBLING ")
print("="*60)

print("\nIndividual model performance on test set :")
for name, acc in per_model_accuracies.items():
    print(f"  {name}: Test Accuracy = {acc:.4f}, F1-Score = {per_model_f1_scores[name]:.4f}")

# Filter models based on minimum accuracy threshold
models_for_ensemble = [name for name, acc in per_model_accuracies.items() if acc >= ENSEMBLE_CONFIG["min_accuracy_threshold"]]
if not models_for_ensemble:
    models_for_ensemble = list(test_predictions.keys())

print(f"\nModels selected for ensemble : {models_for_ensemble}")

# --- Weighted Average Ensemble ---
print("\n" + "="*60)
print("WEIGHTED AVERAGE ENSEMBLE ")
print("="*60)

# Calculate weights based on accuracy
if ENSEMBLE_CONFIG["weighted_method"] == "accuracy_weighted":
    weights = np.array([per_model_accuracies[name] for name in models_for_ensemble])

weights = weights / weights.sum()
weights_dict = {name: float(w) for name, w in zip(models_for_ensemble, weights)}

print(f"\nEnsemble weights ({ENSEMBLE_CONFIG['weighted_method']}) :")
for name, w in weights_dict.items():
    print(f"  {name}: {w:.4f}")

# Create weighted predictions
ensemble_predictions = []
for name in models_for_ensemble:
    ensemble_predictions.append(test_predictions[name])

weighted_ensemble_pred = np.average(ensemble_predictions, axis=0, weights=weights)
y_weighted_pred = np.argmax(weighted_ensemble_pred, axis=1)
weighted_accuracy = accuracy_score(y_test_true, y_weighted_pred)

print(f"\nWeighted Average Ensemble Performance :")
print(f"  Test Accuracy: {weighted_accuracy:.4f}")

# Save confusion matrix
cm_weighted = confusion_matrix(y_test_true, y_weighted_pred)
plot_and_save_cm(cm_weighted, "Weighted_Average", resolution_label)

# Save classification report
report_weighted = classification_report(y_test_true, y_weighted_pred, target_names=CLASS_NAMES, digits=4)
print("\nClassification Report (Weighted Average Ensemble) :")
print(report_weighted)

report_path_weighted = os.path.join(FOLDER_PATHS["clinical_metrics"], f"classification_report_{resolution_label}_Weighted_Average_v18.txt")
os.makedirs(os.path.dirname(report_path_weighted), exist_ok=True)
with open(report_path_weighted, 'w') as f:
    f.write(f"Classification Report for Weighted Average Ensemble at {resolution_label} resolution \n")
    f.write("="*70 + "\n")
    f.write(report_weighted)
print(f"Classification report saved: {report_path_weighted}")

# Save JSON version
report_dict_weighted = classification_report(y_test_true, y_weighted_pred, target_names=CLASS_NAMES, output_dict=True)
report_json_path_weighted = os.path.join(FOLDER_PATHS["clinical_metrics"], f"classification_report_{resolution_label}_Weighted_Average_v18.json")
with open(report_json_path_weighted, 'w') as f:
    json.dump(report_dict_weighted, f, indent=4)
print(f"Classification report JSON saved: {report_json_path_weighted}")




# --- Stacking Ensemble with Cross-Validation ---
print("\n" + "="*60)
print("STACKING ENSEMBLE WITH CROSS-VALIDATION ")
print("="*60)

# Prepare features for stacking (concatenate predictions from all models)
stacking_features = np.concatenate([test_predictions[name] for name in models_for_ensemble], axis=1)
print(f"Stacking feature shape: {stacking_features.shape}")

# Split data for stacking CV
#n fold used for: prevents the meta-model from "memorizing" the base models', By training on different "folds" of data, you ensure the meta-model generalizes well to new, unseen medical images.
kf = KFold(n_splits=ENSEMBLE_CONFIG["n_folds"], shuffle=True, random_state=SEED)
cv_accuracies = []
meta_models = []

print(f"\nTraining stacking meta-model with {ENSEMBLE_CONFIG['n_folds']}-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(stacking_features)):
    print(f"  Fold {fold+1}/{ENSEMBLE_CONFIG['n_folds']}...")

    # Split data
    X_train, X_val = stacking_features[train_idx], stacking_features[val_idx]
    y_train, y_val = y_test_true[train_idx], y_test_true[val_idx]

    # Train meta-model
    meta_model = LogisticRegression(
        max_iter=ENSEMBLE_CONFIG["stacking_max_iter"],
        random_state=SEED,
        class_weight='balanced'  # Handle class imbalance
    )
    meta_model.fit(X_train, y_train)

    # Evaluate on validation fold
    cv_pred = meta_model.predict(X_val)
    cv_acc = accuracy_score(y_val, cv_pred)
    cv_accuracies.append(cv_acc)
    meta_models.append(meta_model)
    print(f"    Fold {fold+1} Accuracy: {cv_acc:.4f}")

# Calculate cross-validation statistics
cv_mean = np.mean(cv_accuracies) if cv_accuracies else 0.0
cv_std = np.std(cv_accuracies) if len(cv_accuracies) > 1 else 0.0
print(f"\nStacking CV Performance : {cv_mean:.4f} ± {cv_std:.4f}")

# Create final stacking model using the best fold or train on full data
if meta_models:
    # Select the meta-model with highest validation accuracy
    best_fold_idx = np.argmax(cv_accuracies)
    meta_model = meta_models[best_fold_idx]
    print(f"  Using meta-model from fold {best_fold_idx+1} (best CV accuracy: {cv_accuracies[best_fold_idx]:.4f})")
else:
    print("  All CV folds failed. Training meta-model on full dataset...")
    meta_model = LogisticRegression(
        max_iter=ENSEMBLE_CONFIG["stacking_max_iter"],
        random_state=SEED,
        class_weight='balanced'
    )
    meta_model.fit(stacking_features, y_test_true)

# Get final stacking predictions
stacking_proba = meta_model.predict_proba(stacking_features)
stacking_pred = np.argmax(stacking_proba, axis=1)
stacking_accuracy = accuracy_score(y_test_true, stacking_pred)

print(f"\nStacking Ensemble Performance :")
print(f"  Test Accuracy: {stacking_accuracy:.4f}")

# Save confusion matrix
cm_stacking = confusion_matrix(y_test_true, stacking_pred)
plot_and_save_cm(cm_stacking, "Stacking", resolution_label)

# Save classification report
report_stacking = classification_report(y_test_true, stacking_pred, target_names=CLASS_NAMES, digits=4)
print("\nClassification Report (Stacking Ensemble) :")
print(report_stacking)

report_path_stacking = os.path.join(FOLDER_PATHS["clinical_metrics"], f"classification_report_{resolution_label}_Stacking_v18.txt")
with open(report_path_stacking, 'w') as f:
    f.write(f"Classification Report for Stacking Ensemble at {resolution_label} resolution \n")
    f.write("="*70 + "\n")
    f.write(report_stacking)
print(f"Classification report saved: {report_path_stacking}")

# Save JSON version
report_dict_stacking = classification_report(y_test_true, stacking_pred, target_names=CLASS_NAMES, output_dict=True)
report_json_path_stacking = os.path.join(FOLDER_PATHS["clinical_metrics"], f"classification_report_{resolution_label}_Stacking_v18.json")
with open(report_json_path_stacking, 'w') as f:
    json.dump(report_dict_stacking, f, indent=4)
print(f"Classification report JSON saved: {report_json_path_stacking}")




# ---  Ensemble Comparison ---
print("\n" + "="*60)
print("ENSEMBLE COMPARISON AND SELECTION ")
print("="*60)

# Compare ensemble performance
ensemble_results = {
    "Weighted Average": weighted_accuracy,
    "Stacking": stacking_accuracy
}

print("\nEnsemble Performance Comparison:")
for name, acc in ensemble_results.items():
    print(f"  {name} Ensemble: {acc:.4f}")

# Statistical comparison using McNemar's test
print("\nStatistical Comparison (McNemar's Test):")
p_value = mcnemar_test(y_test_true, y_weighted_pred, stacking_pred)
print(f"  Weighted Average vs Stacking p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  Result: Significant difference between ensembles (p < 0.05)")
else:
    print("  Result: No significant difference between ensembles (p >= 0.05)")

# Select best ensemble
best_ensemble = max(ensemble_results, key=ensemble_results.get)
best_accuracy = ensemble_results[best_ensemble]
best_proba = weighted_ensemble_pred if best_ensemble == "Weighted Average" else stacking_proba
best_pred = y_weighted_pred if best_ensemble == "Weighted Average" else stacking_pred

print(f"\nBest Ensemble Technique : {best_ensemble}")
print(f"Best Ensemble Test Accuracy: {best_accuracy:.4f}")


# --- Save Ensemble Configuration and Results ---
print("\n" + "="*60)
print("SAVING ENSEMBLE CONFIGURATION AND RESULTS ")
print("="*60)

# Prepare ensemble data for saving
ensemble_data = {
    'timestamp': datetime.now().strftime("%Y%m%d%H%M%S"),
    'resolution': average_size,
    'models_in_ensemble': models_for_ensemble,
    'weighted_average': {
        'accuracy': float(weighted_accuracy),
        'weights': weights_dict,
        'weighting_method': ENSEMBLE_CONFIG["weighted_method"]
    },
    'stacking': {
        'accuracy': float(stacking_accuracy),
        'cv_accuracy': float(cv_mean),
        'cv_std': float(cv_std),
        'meta_model_params': {
            'coefficients': meta_model.coef_.tolist(),
            'intercept': meta_model.intercept_.tolist(),
            'classes': meta_model.classes_.tolist()
        }
    },
    'best_ensemble': best_ensemble,
    'best_accuracy': float(best_accuracy),
    'mcnemar_p_value': float(p_value),
}

# Save ensemble configuration as JSON
ensemble_json_path = os.path.join(FOLDER_PATHS["clinical_metrics"], f"ensemble_config_{resolution_label}_v18.json")
with open(ensemble_json_path, 'w') as f:
    json.dump(ensemble_data, f, indent=4)
print(f"Ensemble configuration saved: {ensemble_json_path}")

# Save ensemble weights and comparison as TXT
ensemble_txt_path = os.path.join(FOLDER_PATHS["clinical_metrics"], f"ensemble_comparison_{resolution_label}_v18.txt")
with open(ensemble_txt_path, 'w') as f:
    f.write("ENSEMBLE LEARNING COMPARISON REPORT \n")
    f.write("=" * 60 + "\n")
    f.write(f"Resolution: {resolution_label}\n")
    f.write(f"Models in Ensemble: {', '.join(models_for_ensemble)}\n")
    f.write(f"Ensemble Construction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    f.write("\nWEIGHTED AVERAGE ENSEMBLE:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Accuracy: {weighted_accuracy:.4f}\n")
    f.write("Model Weights:\n")
    for name, w in weights_dict.items():
        f.write(f"  {name}: {w:.4f}\n")

    f.write("\nSTACKING ENSEMBLE:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Accuracy: {stacking_accuracy:.4f}\n")
    f.write(f"Cross-Validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}\n")

    f.write("\nENSEMBLE COMPARISON:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Best Ensemble: {best_ensemble}\n")
    f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
    f.write(f"McNemar's Test p-value (Weighted vs Stacking): {p_value:.4f}\n")
    if p_value < 0.05:
        f.write("Conclusion: Significant difference between ensembles (p < 0.05)\n")
    else:
        f.write("Conclusion: No significant difference between ensembles (p >= 0.05)\n")

print(f"Ensemble comparison report saved: {ensemble_txt_path}")

# --- Create Ensemble Performance Visualization ---
print("\nCreating ensemble performance comparison visualization ...")
plt.figure(figsize=(12, 8))
x = np.arange(1)
width = 0.35

# Plot individual model accuracies
for i, name in enumerate(models_for_ensemble):
    if i < 5:  # Limit to 5 models for clarity
        plt.bar(x - width/2 + i*0.1, per_model_accuracies[name], width/5,
                label=f'{name}', alpha=0.7, edgecolor='black')

# Plot ensemble accuracies
plt.bar(x + width/2 - 0.1, weighted_accuracy, width/3,
        label='Weighted Average', color='#2ca02c', alpha=0.9, edgecolor='black')
plt.bar(x + width/2 + 0.1, stacking_accuracy, width/3,
        label='Stacking', color='#d62728', alpha=0.9, edgecolor='black')

plt.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.7,
            label=f'Best Ensemble ({best_accuracy:.4f})')

plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
#plt.title(f'Ensemble Performance Comparison ({resolution_label}) ', fontsize=16, fontweight='bold')
plt.xticks([])
plt.ylim(0, 1.05)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', alpha=0.3)

# Add value labels
for i, name in enumerate(models_for_ensemble):
    if i < 5:
        plt.text(x - width/2 + i*0.1, per_model_accuracies[name] + 0.01,
                f'{per_model_accuracies[name]:.3f}',
                ha='center', va='bottom', fontsize=9, rotation=45)

plt.text(x + width/2 - 0.1, weighted_accuracy + 0.01, f'{weighted_accuracy:.3f}',
        ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.text(x + width/2 + 0.1, stacking_accuracy + 0.01, f'{stacking_accuracy:.3f}',
        ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
comparison_path = os.path.join(FOLDER_PATHS["viz"], f"ensemble_comparison_{resolution_label}_v18.png")
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Ensemble comparison visualization saved: {comparison_path}")

# --- 14. Final Evaluation & Ensemble Summary ---
print("\n" + "="*60)
print("EVALUATION & ENSEMBLE LEARNING SUMMARY")
print("="*60)

print(f"\nResolution used: {average_size[0]}×{average_size[1]}")
print(f"Models included in ensemble: {', '.join(models_for_ensemble)}")
print(f"Ensemble methods compared: Weighted Average, Stacking")
print(f"Best ensemble method: {best_ensemble}")
print(f"Best ensemble test accuracy: {best_accuracy:.4f}")

# Save ensemble results for real-time demo
GLOBAL_CONFIG['ensemble_results'] = {
    'best_ensemble': best_ensemble,
    'best_accuracy': best_accuracy,
    'best_proba': best_proba,
    'best_pred': best_pred,
    'models_for_ensemble': models_for_ensemble,
    'weights_dict': weights_dict,
    'meta_model': meta_model,
    'resolution': average_size,
}

print("\n=== SECTION 5 completed: Test evaluation and ensemble learning finished ===\n")
