"""
Training pipeline for the pneumonia detection model.

Usage
-----
    python -m pneumonia_detection.train

Two-stage training strategy
----------------------------
Stage 1 – Feature extraction
    The ResNet50 base is frozen.  Only the custom classification head is
    trained until the validation loss stops improving.

Stage 2 – Fine-tuning
    The top layers of the base (from config.FINE_TUNE_AT onwards) are
    unfrozen and the whole network is re-trained with a much smaller
    learning rate.
"""

import os

import tensorflow as tf

from pneumonia_detection import config
from pneumonia_detection.model import build_model
from pneumonia_detection.preprocess import (
    compute_class_weights,
    get_train_generator,
    get_val_generator,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _get_callbacks(model_path: str, phase: str) -> list:
    checkpoint_path = model_path.replace(".keras", f"_{phase}.keras")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=config.VERBOSE,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=config.VERBOSE,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=config.VERBOSE,
        ),
    ]


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> tf.keras.Model:
    """Run the full two-stage training and return the trained model."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    train_gen = get_train_generator()
    val_gen = get_val_generator()
    class_weights = compute_class_weights(config.TRAIN_DIR)

    # ------------------------------------------------------------------
    # Stage 1: Feature extraction (frozen base)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Stage 1 — Feature extraction (frozen base)")
    print("=" * 60)

    model = build_model(trainable_base=False)
    model.summary()

    model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=_get_callbacks(config.MODEL_PATH, phase="stage1"),
        verbose=config.VERBOSE,
    )

    # ------------------------------------------------------------------
    # Stage 2: Fine-tuning (partial base unfrozen)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Stage 2 — Fine-tuning (top base layers unfrozen)")
    print("=" * 60)

    model = build_model(trainable_base=True)
    # Load the best weights from stage 1
    stage1_path = config.MODEL_PATH.replace(".keras", "_stage1.keras")
    if os.path.exists(stage1_path):
        model.load_weights(stage1_path)
    else:
        import warnings
        warnings.warn(
            f"Stage 1 checkpoint not found at {stage1_path!r}. "
            "Fine-tuning will start from randomly initialised weights.",
            UserWarning,
            stacklevel=2,
        )

    model.fit(
        train_gen,
        epochs=config.FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=_get_callbacks(config.MODEL_PATH, phase="stage2"),
        verbose=config.VERBOSE,
    )

    # Save the final model
    model.save(config.MODEL_PATH)
    print(f"\nFinal model saved to: {config.MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
