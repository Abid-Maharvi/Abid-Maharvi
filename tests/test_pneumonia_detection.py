"""
Unit tests for the pneumonia detection package.

These tests verify the core building blocks without requiring the full
Kaggle dataset or a pre-trained model file on disk.
"""

import importlib
import os
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so imports work regardless of how
# pytest is invoked.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ===========================================================================
# config
# ===========================================================================

class TestConfig:
    def test_classes(self):
        from pneumonia_detection import config
        assert config.CLASSES == ["NORMAL", "PNEUMONIA"]
        assert config.NUM_CLASSES == 2

    def test_image_shape(self):
        from pneumonia_detection import config
        assert config.IMAGE_SIZE == (224, 224)
        assert config.INPUT_SHAPE == (224, 224, 3)

    def test_hyperparams_positive(self):
        from pneumonia_detection import config
        assert config.BATCH_SIZE > 0
        assert config.EPOCHS > 0
        assert config.LEARNING_RATE > 0
        assert config.FINE_TUNE_LR > 0
        assert config.FINE_TUNE_LR < config.LEARNING_RATE

    def test_fine_tune_at_within_resnet50_bounds(self):
        from pneumonia_detection import config
        # ResNet50 has 175 layers; FINE_TUNE_AT must be in [0, 175)
        assert 0 <= config.FINE_TUNE_AT < 175


# ===========================================================================
# preprocess — preprocess_image
# ===========================================================================

class TestPreprocessImage:
    """Tests for the single-image preprocessing utility."""

    def _make_dummy_image(self, tmp_path: str) -> str:
        """Write a small white JPEG to *tmp_path* and return its path."""
        from PIL import Image
        img = Image.fromarray(
            np.full((64, 64, 3), 200, dtype=np.uint8), mode="RGB"
        )
        path = os.path.join(tmp_path, "dummy_xray.jpg")
        img.save(path)
        return path

    def test_output_shape(self, tmp_path):
        from pneumonia_detection import config
        from pneumonia_detection.preprocess import preprocess_image

        img_path = self._make_dummy_image(str(tmp_path))
        arr = preprocess_image(img_path)
        assert arr.shape == (1, *config.IMAGE_SIZE, 3)

    def test_pixel_range(self, tmp_path):
        from pneumonia_detection.preprocess import preprocess_image

        img_path = self._make_dummy_image(str(tmp_path))
        arr = preprocess_image(img_path)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_dtype_float(self, tmp_path):
        from pneumonia_detection.preprocess import preprocess_image

        img_path = self._make_dummy_image(str(tmp_path))
        arr = preprocess_image(img_path)
        assert np.issubdtype(arr.dtype, np.floating)


# ===========================================================================
# preprocess — compute_class_weights
# ===========================================================================

class TestComputeClassWeights:
    """Tests for the class-weight helper used to offset label imbalance."""

    def _make_dir_with_files(self, base: str, class_name: str, n: int) -> str:
        path = os.path.join(base, class_name)
        os.makedirs(path, exist_ok=True)
        for i in range(n):
            open(os.path.join(path, f"img_{i}.jpg"), "w").close()
        return path

    def test_balanced_equal_weights(self, tmp_path):
        from pneumonia_detection.preprocess import compute_class_weights

        base = str(tmp_path)
        self._make_dir_with_files(base, "NORMAL", 10)
        self._make_dir_with_files(base, "PNEUMONIA", 10)
        weights = compute_class_weights(base)
        assert abs(weights[0] - weights[1]) < 1e-9

    def test_imbalanced_higher_weight_for_minority(self, tmp_path):
        from pneumonia_detection.preprocess import compute_class_weights

        base = str(tmp_path)
        self._make_dir_with_files(base, "NORMAL", 5)
        self._make_dir_with_files(base, "PNEUMONIA", 15)
        weights = compute_class_weights(base)
        # NORMAL (minority) should have a higher weight
        assert weights[0] > weights[1]

    def test_weights_are_positive(self, tmp_path):
        from pneumonia_detection.preprocess import compute_class_weights

        base = str(tmp_path)
        self._make_dir_with_files(base, "NORMAL", 8)
        self._make_dir_with_files(base, "PNEUMONIA", 12)
        weights = compute_class_weights(base)
        assert weights[0] > 0
        assert weights[1] > 0


# ===========================================================================
# model — build_model
# ===========================================================================

class TestBuildModel:
    """Tests for the ResNet50-based classifier factory."""

    def test_model_output_shape(self):
        """The model must output a single sigmoid probability per image."""
        from pneumonia_detection.model import build_model
        model = build_model(trainable_base=False, weights=None)
        assert model.output_shape == (None, 1)

    def test_model_input_shape(self):
        from pneumonia_detection import config
        from pneumonia_detection.model import build_model
        model = build_model(trainable_base=False, weights=None)
        assert model.input_shape == (None, *config.INPUT_SHAPE)

    def test_frozen_base(self):
        """In feature-extraction mode the entire base must be non-trainable."""
        from pneumonia_detection.model import build_model
        model = build_model(trainable_base=False, weights=None)
        # The ResNet50 base sub-model should have no trainable weights
        base = next(
            l for l in model.layers
            if hasattr(l, "layers")  # it's a nested model
        )
        assert not any(w.trainable for w in base.trainable_weights)

    def test_partial_unfreeze_in_finetune_mode(self):
        """In fine-tuning mode some (but not necessarily all) base layers are trainable."""
        from pneumonia_detection import config
        from pneumonia_detection.model import build_model
        model = build_model(trainable_base=True, weights=None)
        base = next(l for l in model.layers if hasattr(l, "layers"))
        trainable_layers = [l for l in base.layers if l.trainable]
        frozen_layers = [l for l in base.layers if not l.trainable]
        assert len(trainable_layers) > 0, "At least some layers must be trainable"
        assert len(frozen_layers) > 0, "At least some layers must stay frozen"

    def test_compile_metrics(self):
        from pneumonia_detection.model import build_model
        model = build_model(trainable_base=False, weights=None)
        assert model.compiled, "Model must be compiled"
        assert model.loss == "binary_crossentropy"
        # Verify all required metrics are present in the compile configuration
        compile_cfg = model.get_compile_config()
        metric_names = []
        for m in compile_cfg.get("metrics", []):
            if isinstance(m, str):
                metric_names.append(m)
            elif isinstance(m, dict):
                metric_names.append(m.get("config", {}).get("name", ""))
        assert "accuracy" in metric_names
        assert "auc" in metric_names

    def test_predict_output_range(self):
        """Random noise input should produce a probability in [0, 1]."""
        import tensorflow as tf
        from pneumonia_detection import config
        from pneumonia_detection.model import build_model

        model = build_model(trainable_base=False, weights=None)
        dummy = np.random.rand(1, *config.INPUT_SHAPE).astype(np.float32)
        prob = model.predict(dummy, verbose=0)[0][0]
        assert 0.0 <= prob <= 1.0


# ===========================================================================
# predict — predict_image (with a tiny stub model)
# ===========================================================================

class TestPredictImage:
    """Tests for the predict_image function using a lightweight stub model."""

    @pytest.fixture()
    def stub_model(self):
        """A tiny Keras model that always returns 0.9 (PNEUMONIA)."""
        import tensorflow as tf
        from pneumonia_detection import config

        inputs = tf.keras.Input(shape=config.INPUT_SHAPE)
        # Use a Lambda layer to return a constant near 1
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid",
                                        kernel_initializer="ones",
                                        bias_initializer="ones")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    @pytest.fixture()
    def dummy_image(self, tmp_path):
        from PIL import Image
        img = Image.fromarray(
            np.full((64, 64, 3), 128, dtype=np.uint8), mode="RGB"
        )
        path = os.path.join(str(tmp_path), "test_xray.jpg")
        img.save(path)
        return path

    def test_result_keys(self, stub_model, dummy_image):
        from pneumonia_detection.predict import predict_image
        result = predict_image(dummy_image, model=stub_model)
        assert "label" in result
        assert "confidence" in result
        assert "probability" in result

    def test_label_is_valid_class(self, stub_model, dummy_image):
        from pneumonia_detection import config
        from pneumonia_detection.predict import predict_image
        result = predict_image(dummy_image, model=stub_model)
        assert result["label"] in config.CLASSES

    def test_confidence_in_range(self, stub_model, dummy_image):
        from pneumonia_detection.predict import predict_image
        result = predict_image(dummy_image, model=stub_model)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probability_in_range(self, stub_model, dummy_image):
        from pneumonia_detection.predict import predict_image
        result = predict_image(dummy_image, model=stub_model)
        assert 0.0 <= result["probability"] <= 1.0

    def test_confidence_matches_label(self, stub_model, dummy_image):
        """Confidence must reflect the probability of the *predicted* label."""
        from pneumonia_detection.predict import predict_image
        result = predict_image(dummy_image, model=stub_model)
        prob = result["probability"]
        if result["label"] == "PNEUMONIA":
            assert abs(result["confidence"] - prob) < 1e-4
        else:
            assert abs(result["confidence"] - (1.0 - prob)) < 1e-4


# ===========================================================================
# predict — predict_batch
# ===========================================================================

class TestPredictBatch:
    @pytest.fixture()
    def stub_model(self):
        import tensorflow as tf
        from pneumonia_detection import config

        inputs = tf.keras.Input(shape=config.INPUT_SHAPE)
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, outputs)

    @pytest.fixture()
    def dummy_images(self, tmp_path):
        from PIL import Image
        paths = []
        for i in range(3):
            img = Image.fromarray(
                np.full((64, 64, 3), i * 60, dtype=np.uint8), mode="RGB"
            )
            path = os.path.join(str(tmp_path), f"xray_{i}.jpg")
            img.save(path)
            paths.append(path)
        return paths

    def test_batch_length(self, stub_model, dummy_images):
        from pneumonia_detection.predict import predict_batch
        results = predict_batch(dummy_images, model=stub_model)
        assert len(results) == len(dummy_images)

    def test_each_result_valid(self, stub_model, dummy_images):
        from pneumonia_detection import config
        from pneumonia_detection.predict import predict_batch
        results = predict_batch(dummy_images, model=stub_model)
        for r in results:
            assert r["label"] in config.CLASSES
            assert 0.0 <= r["confidence"] <= 1.0
            assert 0.0 <= r["probability"] <= 1.0
