"""
Data loading, preprocessing, and augmentation utilities.

Expected directory layout (Kaggle chest-xray dataset):
    data/chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pneumonia_detection import config


# ---------------------------------------------------------------------------
# Class-weight helper (handles class imbalance)
# ---------------------------------------------------------------------------

def compute_class_weights(train_dir: str) -> dict:
    """Return {0: w_normal, 1: w_pneumonia} weights to offset class imbalance."""
    normal_count = len(os.listdir(os.path.join(train_dir, "NORMAL")))
    pneumonia_count = len(os.listdir(os.path.join(train_dir, "PNEUMONIA")))
    total = normal_count + pneumonia_count
    # sklearn-style balanced weights: n_samples / (n_classes * n_samples_class)
    weight_normal = total / (2.0 * normal_count)
    weight_pneumonia = total / (2.0 * pneumonia_count)
    return {0: weight_normal, 1: weight_pneumonia}


# ---------------------------------------------------------------------------
# Generator builders
# ---------------------------------------------------------------------------

def _rescale_only() -> ImageDataGenerator:
    return ImageDataGenerator(rescale=1.0 / 255.0)


def _augmented() -> ImageDataGenerator:
    return ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        fill_mode=config.FILL_MODE,
    )


def _flow_from_dir(
    generator: ImageDataGenerator,
    directory: str,
    shuffle: bool = True,
) -> tf.keras.preprocessing.image.DirectoryIterator:
    return generator.flow_from_directory(
        directory,
        target_size=config.IMAGE_SIZE,
        color_mode="rgb",
        class_mode="binary",
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.RANDOM_SEED,
    )


def get_train_generator():
    """Augmented generator for the training split."""
    return _flow_from_dir(_augmented(), config.TRAIN_DIR, shuffle=True)


def get_val_generator():
    """Plain rescale generator for the validation split."""
    return _flow_from_dir(_rescale_only(), config.VAL_DIR, shuffle=False)


def get_test_generator():
    """Plain rescale generator for the test split (no shuffle, for metrics)."""
    return _flow_from_dir(_rescale_only(), config.TEST_DIR, shuffle=False)


# ---------------------------------------------------------------------------
# Single-image preprocessing (used by predict.py and the Streamlit app)
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load a single image from *image_path*, resize it to the model's expected
    input dimensions, rescale pixel values to [0, 1], and return a
    (1, H, W, C) NumPy array ready to pass to model.predict().
    """
    img = tf.keras.utils.load_img(
        image_path,
        color_mode="rgb",
        target_size=config.IMAGE_SIZE,
    )
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)
