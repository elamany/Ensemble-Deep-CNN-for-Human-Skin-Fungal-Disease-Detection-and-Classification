
print("=== SECTION 2: Data Cleaning + Average Size Calculation + Augmentation + Splitting ===")

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
import cv2
import pandas as pd
from collections import Counter
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

required = ["FOLDER_PATHS", "CLASS_NAMES", "SEED", "PROCESSED_DIR", "IMAGE_EXTS", "GLOBAL_CONFIG"]
missing = [r for r in required if r not in globals()]
if missing:
    raise RuntimeError(f"Required globals variable missing: {missing}.")

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

def calculate_imbalance_ratios(counts_dict, stage_name):
    """Calculate class imbalance ratio"""
    total = sum(counts_dict.values())
    ratios = {}
    print(f"\nClass Distribution - {stage_name}:")
    for cls in CLASS_NAMES:
        count = counts_dict.get(cls, 0)
        ratio = count / total if total > 0 else 0
        ratios[cls] = ratio
        print(f"  {cls}: {count} images ({ratio:.1%})")

    non_zero = [c for c in counts_dict.values() if c > 0]
    if non_zero:
        imbalance_factor = max(non_zero) / min(non_zero)
        print(f"  Imbalance Factor: {imbalance_factor:.1f}x (max/min)")
    return ratios

def collect_and_clean_data():
    """Collect raw data and perform data cleaning"""
    all_images = []
    all_labels = []
    original_counts = {cls: 0 for cls in CLASS_NAMES}
    preprocessed_counts = {cls: 0 for cls in CLASS_NAMES}
    removed_images = []
    removed_by_class = {cls: 0 for cls in CLASS_NAMES}

    print("\n[1/5] Scanning raw dataset...")
    for cls_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(FOLDER_PATHS["raw"], cls)
        if not os.path.isdir(cls_dir):
            print(f"-- Missing directory for {cls}: {cls_dir}")
            continue

        images = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        original_counts[cls] = len(images)
        print(f"  {cls}: {len(images)} raw images")

        for fname in images:
            img_path = os.path.join(cls_dir, fname)
            img = preprocess_image(img_path) #data cleaning
            if img is not None:
                all_images.append(img_path)
                all_labels.append(cls_idx)
                preprocessed_counts[cls] += 1
            else:
                removed_images.append(img_path)
                removed_by_class[cls] += 1

    # Generate image removal report
    if removed_images:
        report_dir = os.path.join(FOLDER_PATHS["clinical_metrics"], "data_cleaning_report")
        os.makedirs(report_dir, exist_ok=True)

        with open(os.path.join(report_dir, "removal_report.txt"), 'w') as f:
            f.write(f"IMAGES REMOVED DURING CLEANING\n{'='*50}\n")
            f.write(f"Total removed: {len(removed_images)}\n")
            f.write(f"Removals by class:\n")
            for cls, count in removed_by_class.items():
                f.write(f"- {cls}: {count}\n")

        plt.figure(figsize=(10, 6))
        bars = plt.bar(CLASS_NAMES, [removed_by_class[cls] for cls in CLASS_NAMES],
                      color=plt.cm.viridis(np.linspace(0, 1, len(CLASS_NAMES))))
        plt.title('Images Removed During Cleaning', fontsize=14, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Removed Images')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "removals_by_class.png"), dpi=300)
        plt.close()

    return all_images, all_labels, original_counts, preprocessed_counts, removed_images

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

def augumetation_preprocess(target_size,images, labels, preprocessed_counts):
    print("\n[3/5] Augmentation dataset...")

    # Calculate target size
    min_count = min(preprocessed_counts.values())
    if min_count == 0:
        raise ValueError("Cannot augment - No image found")

    TARGET_PER_CLASS = 7 * min_count
    print(f"  Target per class: {TARGET_PER_CLASS} images (7 × min class size of {min_count})")

    augmentation_pipeline = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.ShiftScaleRotate(shift_limit=0.05,
                           scale_limit=0.15,
                           rotate_limit=12,
                          border_mode=cv2.BORDER_REFLECT, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.GridDistortion(num_steps=3, distort_limit=0.2, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ], p=0.9)


    augmented_images = images.copy()
    augmented_labels = labels.copy()
    augment_added = {cls: 0 for cls in CLASS_NAMES}
    class_images = {cls: [] for cls in range(len(CLASS_NAMES))}

    # Organize by class
    for img_path, label in zip(images, labels):
        class_images[label].append(img_path)

    # Augment minority classes
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        current_count = len(class_images[cls_idx])

        if current_count < TARGET_PER_CLASS:
            needed_to_add_augmentation = TARGET_PER_CLASS - current_count
            augment_added[cls_name] = needed_to_add_augmentation
            print(f"  {cls_name}: {current_count} -> {TARGET_PER_CLASS} (+{needed_to_add_augmentation} augmented)")

            # controlled repetition
            sources = []
            repeats = max(1, needed_to_add_augmentation // (2 * current_count) + 1)  #ensures every original image is used multiple times at the start.
            for img_path in class_images[cls_idx]:
                sources.extend([img_path] * min(repeats, needed_to_add_augmentation - len(sources)))
            while len(sources) < needed_to_add_augmentation:
                sources.extend(class_images[cls_idx])
            sources = sources[:needed_to_add_augmentation]

            # Generate augmentations
            for i, img_path in enumerate(sources):
                try:
                    img = np.array(Image.open(img_path).convert('RGB'))
                    augmented = augmentation_pipeline(image=img)['image']# augment apply
                    aug_img = Image.fromarray(augmented.astype(np.uint8))
                    augmented_images.append(('augmented', img_path, aug_img))
                    augmented_labels.append(cls_idx)
                except Exception as e:
                    print(f"    Augmentation failed for {os.path.basename(img_path)}: {str(e)[:50]}")
        else:
            print(f"  {cls_name}: {current_count} (no augmentation needed)")

    final_counts = Counter(augmented_labels)
    print("\nClass distribution after augmentation :")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        print(f"  {cls_name}: {final_counts[cls_idx]} images")

    return augmented_images, augmented_labels, augment_added, TARGET_PER_CLASS

def split_dataset(images, labels, test_size=0.15, val_size=0.15):
    print(f"\n[4/5]  70/15/15 ratio split...")

    # separate test set
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=SEED
    )

    # separate validation from training
    val_size_adj = val_size / (1 - test_size)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=val_size_adj,
        stratify=train_val_labels, random_state=SEED
    )

    # Report splits
    total = len(images)
    print(f"\nFinal dataset splits:")
    print(f"  Training:   {len(train_images)} ({len(train_images)/total:.1%})")
    print(f"  Validation: {len(val_images)} ({len(val_images)/total:.1%})")
    print(f"  Test:       {len(test_images)} ({len(test_images)/total:.1%})")

    # Class distribution per split
    splits = {
        "Training": train_labels,
        "Validation": val_labels,
        "Test": test_labels
    }

    print("\nClass distribution per split:")
    for split_name, split_labels in splits.items():
        counts = np.bincount(split_labels, minlength=len(CLASS_NAMES))
        percentages = [f"{c/sum(counts):.1%}" for c in counts]
        print(f"  {split_name}:")
        for cls, count, pct in zip(CLASS_NAMES, counts, percentages):
            print(f"    {cls}: {count} images ({pct})")

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

def save_processed_images(images, labels, split_name):
    """Save images to Google Drive"""
    split_dir = os.path.join(PROCESSED_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Create class subdirectories
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    saved = 0
    errors = 0
    aug_counters = {}

    print(f"\n[5/5] Saving {len(images)} {split_name} images to {split_dir}...")

    # Process in smaller batches
    BATCH_SIZE = 50
    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(images))
        print(f"  Processing batch {batch_start//BATCH_SIZE + 1}/{(len(images)+BATCH_SIZE-1)//BATCH_SIZE} "
              f"({batch_start} to {batch_end-1})")

        for i in range(batch_start, batch_end):
            img_ref, label = images[i], labels[i]
            cls_name = CLASS_NAMES[label]
            dest_dir = os.path.join(split_dir, cls_name)

            try:
                if isinstance(img_ref, tuple) and img_ref[0] == 'augmented': # Handle augmented image
                    _, orig_path, aug_img = img_ref
                    base = os.path.splitext(os.path.basename(orig_path))[0]

                    # Create unique counter per original image per split
                    key = (cls_name, base)
                    aug_counters[key] = aug_counters.get(key, 0) + 1

                    # create unique filename
                    safe_base = base[:30].replace(" ", "_").replace(".", "_")
                    dest_path = os.path.join(
                        dest_dir,
                        f"{safe_base}_aug{aug_counters[key]}_{split_name[:3]}.png"
                    )


                    aug_img.save(dest_path, quality=95, optimize=True)
                    aug_img.close()
                else: # Handle original image
                    orig_fname = os.path.basename(img_ref)
                    dest_path = os.path.join(dest_dir, orig_fname)
                    shutil.copy2(img_ref, dest_path)

                saved += 1
                if saved % 100 == 0:
                    print(f"    -- {saved}/{len(images)} images saved ({errors} errors)")

            except Exception as e:
                continue

    print(f"   COMPLETED: {saved} images saved to {split_dir} (errors: {errors})")
    return saved

def generate_summary(original_counts, preprocessed_counts, entire_counts,
                    augment_added, train_counts, val_counts, test_counts,
                    avg_size, target_per_class):
    """Generate report"""
    summary_dir = os.path.join(FOLDER_PATHS["clinical_metrics"], "dataset_summary_v19")
    os.makedirs(summary_dir, exist_ok=True)


    df = pd.DataFrame({
        "Stage": ["Raw", "Preprocessed", "Augmented (Entire)", "Training", "Validation", "Test"],
        "Total": [
            sum(original_counts.values()),
            sum(preprocessed_counts.values()),
            sum(entire_counts.values()),
            sum(train_counts.values()),
            sum(val_counts.values()),
            sum(test_counts.values())
        ],
        **{cls: [
            original_counts.get(cls, 0),
            preprocessed_counts.get(cls, 0),
            entire_counts.get(cls, 0),
            train_counts.get(cls, 0),
            val_counts.get(cls, 0),
            test_counts.get(cls, 0)
        ] for cls in CLASS_NAMES}
    })
    df.to_csv(os.path.join(summary_dir, "dataset_counts.csv"), index=False)


    print(f"\nSummary saved to: {summary_dir}")
    return summary_dir

print("\n" + "="*70)
print("DATASET PROCESSING PIPELINE")
print("="*70)

# 1: Load images and data clean
all_images, all_labels, raw_counts, cleaned_counts, _ = collect_and_clean_data()
if not all_images:
    raise RuntimeError("No valid images after cleaning - check raw data quality")

# Report cleaning results
print("\nCLEANING RESULTS:")
print(f"  Raw images: {sum(raw_counts.values())}")
print(f"  Valid images after cleaning: {sum(cleaned_counts.values())}")
print(f"  Removal rate: {(1 - sum(cleaned_counts.values())/sum(raw_counts.values())):.1%}")
calculate_imbalance_ratios(raw_counts, "Raw Dataset")
calculate_imbalance_ratios(cleaned_counts, "After Cleaning")

# 2: Calculate Image Size
avg_size = calculate_average_image_size(all_images)

# 3: Augment Dataset
augmented_images, augmented_labels, augment_added, target_per_class = augumetation_preprocess(
    avg_size,all_images, all_labels, cleaned_counts
)
entire_counts = Counter(augmented_labels)
entire_counts_dict = {CLASS_NAMES[i]: count for i, count in entire_counts.items()}
calculate_imbalance_ratios(entire_counts_dict, "After 7x Augmentation")

# 4: Split Augmented Dataset
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = split_dataset(
    augmented_images, augmented_labels
)

# 5: Save Processed Data
for split_name, images, labels in [
    ("train", train_images, train_labels),
    ("val", val_images, val_labels),
    ("test", test_images, test_labels)
]:
    save_processed_images(images, labels, split_name)

# 6: Generate Report
summary_dir = generate_summary(
    raw_counts,
    cleaned_counts,
    entire_counts_dict,
    augment_added,
    {CLASS_NAMES[i]: count for i, count in Counter(train_labels).items()},
    {CLASS_NAMES[i]: count for i, count in Counter(val_labels).items()},
    {CLASS_NAMES[i]: count for i, count in Counter(test_labels).items()},
    avg_size,
    target_per_class
)

size_stats = {
    "average_size": avg_size,
    "width_stats": {
        "min": min(widths),
        "max": max(widths),
        "mean": sum(widths) / len(widths),
        "median": sorted(widths)[len(widths)//2]
    },
    "height_stats": {
        "min": min(heights),
        "max": max(heights),
        "mean": sum(heights) / len(heights),
        "median": sorted(heights)[len(heights)//2]
    },
    "sample_size": len(widths)
}

os.path.join(summary_dir, "image_size_statistics.json")

# FINAL VERIFICATION
print("\n" + "="*70)
print("FINAL DATASET VERIFICATION")
print("="*70)
total_final = len(train_images) + len(val_images) + len(test_images)
print(f"Total images after processing: {total_final}")
print(f"Average image size: {avg_size[0]}×{avg_size[1]}")
print(f"Saved to: {PROCESSED_DIR}")


# Update global config
GLOBAL_CONFIG.update({
    'train_size': len(train_images),
    'val_size': len(val_images),
    'test_size': len(test_images),
    'class_names': CLASS_NAMES,
    'average_size':avg_size,
    'target_per_class': target_per_class,
    'augmentation_done': True
})

print("\n=== SECTION 2 COMPLETED: dataset cleaned, 7x balanced, and split ===")
