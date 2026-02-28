"""
Single-image inference for the pneumonia detection model.

Usage
-----
    python -m pneumonia_detection.predict --image path/to/xray.jpg

    # or from Python
    from pneumonia_detection.predict import predict_image
    result = predict_image("path/to/xray.jpg")
    print(result)  # {'label': 'PNEUMONIA', 'confidence': 0.92, 'probability': 0.92}
"""

import argparse

import numpy as np
import tensorflow as tf

from pneumonia_detection import config
from pneumonia_detection.preprocess import preprocess_image


def predict_image(image_path: str, model: tf.keras.Model | None = None) -> dict:
    """
    Run the trained model on a single chest X-ray image.

    Parameters
    ----------
    image_path : str
        Path to a JPEG / PNG chest X-ray image.
    model : tf.keras.Model, optional
        Pre-loaded model instance.  When *None* the saved model is loaded
        from *config.MODEL_PATH* on every call (convenient for scripts,
        but inefficient inside a loop).

    Returns
    -------
    dict
        ``label``       – "NORMAL" or "PNEUMONIA"
        ``confidence``  – confidence of the predicted class (0–1)
        ``probability`` – raw sigmoid output (probability of PNEUMONIA)
    """
    if model is None:
        model = tf.keras.models.load_model(config.MODEL_PATH)

    img_array = preprocess_image(image_path)
    probability: float = float(model.predict(img_array, verbose=0)[0][0])
    label = config.CLASSES[int(probability >= 0.5)]
    confidence = probability if label == "PNEUMONIA" else 1.0 - probability

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probability": round(probability, 4),
    }


def predict_batch(image_paths: list[str], model: tf.keras.Model | None = None) -> list[dict]:
    """
    Run inference on a list of image paths.

    Parameters
    ----------
    image_paths : list[str]
        Paths to chest X-ray images.
    model : tf.keras.Model, optional
        Pre-loaded model instance.

    Returns
    -------
    list[dict]
        One result dictionary per image (same schema as :func:`predict_image`).
    """
    if model is None:
        model = tf.keras.models.load_model(config.MODEL_PATH)

    batch = np.concatenate([preprocess_image(p) for p in image_paths], axis=0)
    probabilities = model.predict(batch, verbose=0).ravel()
    results = []
    for prob in probabilities:
        label = config.CLASSES[int(prob >= 0.5)]
        confidence = float(prob) if label == "PNEUMONIA" else 1.0 - float(prob)
        results.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "probability": round(float(prob), 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict pneumonia from a chest X-ray image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a chest X-ray image file (JPEG / PNG).",
    )
    parser.add_argument(
        "--model",
        default=config.MODEL_PATH,
        help="Path to the saved Keras model (default: config.MODEL_PATH).",
    )
    args = parser.parse_args()

    loaded_model = tf.keras.models.load_model(args.model)
    result = predict_image(args.image, model=loaded_model)
    print(f"\nImage   : {args.image}")
    print(f"Label   : {result['label']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print(f"P(PNEUMONIA): {result['probability']:.4f}")
