# ==============================================================================
# SECTION 3: Model Definitions (VGG16 + ResNet50V2)
# ==============================================================================

print("=== SECTION 3: Model Definitions (VGG16 + ResNet50V2) ===")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16, ResNet50V2

# --- AdamW optimizer ---
def adamw_optimizer(**kwargs):
    try:
        optimizer = tf.keras.optimizers.AdamW(**kwargs)
        print("Using tf.keras.optimizers.AdamW (built-in, TF >= 2.10)")
        return optimizer
    except AttributeError:
        print("tf.keras.optimizers.AdamW not found (TF < 2.10). Falling back to Adam.")
        kwargs.pop('weight_decay', None)
        return tf.keras.optimizers.Adam(**kwargs)

average_size = GLOBAL_CONFIG.get('average_size', (320, 320))
print(f"\nUsing average image size from data: 256")

# --- Model creation function ---
def create_model(base_model_name="VGG16", input_shape=(320, 320, 3), num_classes=5, dropout_rate=0.5):
    print(f"\nCreating {base_model_name} model with input shape {input_shape}")
    inputs = layers.Input(shape=input_shape)

    if base_model_name == "VGG16":
        base_model = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    elif base_model_name == "ResNet50V2":
        base_model = ResNet50V2(include_top=False, weights="imagenet", input_tensor=inputs)
    else:
        raise ValueError("Only VGG16 and ResNet50V2 supported")

    base_model.trainable = False

    # ——— HEAD  ———
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs, name=f"{base_model_name}_tinea_model")

    # Stats
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"  Trainable params  : {trainable:,}")
    print(f"  Non-trainable     : {non_trainable:,}")
    print(f"  Total params      : {trainable + non_trainable:,}")

    return model, base_model


# --- save global variable ---
GLOBALS_TO_EXPORT = {
    'adamw_optimizer': adamw_optimizer,
    'create_model': create_model,
    'average_size': average_size
}

# Save model configuration for later sections
model_config_path = os.path.join(FOLDER_PATHS["clinical_metrics"], "model_configuration.json")
model_config = {
    "backbones": GLOBAL_CONFIG["MODEL_NAMES"],
    "input_shape": [average_size[0], average_size[1], 3],
    "num_classes": len(CLASS_NAMES),
    "dropout_rate": 0.5,
    "class_names": CLASS_NAMES,
    "timestamp": datetime.now().strftime("%Y%m%d%H%M%S")
}
with open(model_config_path, 'w') as f:
    json.dump(model_config, f, indent=4)
print(f"\nModel configuration saved: {model_config_path}")

print("\n=== SECTION 3 completed: Model definitions with medical-specific customizations ===\n")
