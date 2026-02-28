"""
Model architecture — transfer learning with ResNet50.

Architecture summary
--------------------
Base : ResNet50 (ImageNet weights, top removed)
Head : GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)

Training is done in two stages:
  1. Feature extraction  — base frozen, only the custom head is trained.
  2. Fine-tuning         — top layers of the base are unfrozen and retrained
                           with a very small learning rate.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from pneumonia_detection import config


def build_model(trainable_base: bool = False, weights: str | None = "imagenet") -> tf.keras.Model:
    """
    Build and return the compiled ResNet50-based classifier.

    Parameters
    ----------
    trainable_base : bool
        If *False* (default) the ResNet50 base is frozen (feature-extraction
        phase).  Set to *True* for the fine-tuning phase.
    weights : str or None
        Weights to initialise ResNet50 with.  Use ``"imagenet"`` (default)
        for transfer learning or ``None`` to initialise randomly (useful in
        tests where network access is unavailable).

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training or inference.
    """
    # 1. Load ResNet50 without the classification top
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=config.INPUT_SHAPE,
    )

    # 2. Freeze / unfreeze the base
    if trainable_base:
        # Unfreeze layers from FINE_TUNE_AT onwards
        for layer in base.layers[:config.FINE_TUNE_AT]:
            layer.trainable = False
        for layer in base.layers[config.FINE_TUNE_AT:]:
            layer.trainable = True
    else:
        base.trainable = False

    # 3. Build the classification head
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE)
    x = base(inputs, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="pneumonia_detector")

    # 4. Compile
    lr = config.FINE_TUNE_LR if trainable_base else config.LEARNING_RATE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
