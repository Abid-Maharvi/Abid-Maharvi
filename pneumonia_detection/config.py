"""Central configuration for the pneumonia detection pipeline."""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_resnet50.keras")

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASSES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASSES)  # binary: 1 output neuron with sigmoid

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)   # ResNet50 expected input
IMAGE_CHANNELS = 3
INPUT_SHAPE = (*IMAGE_SIZE, IMAGE_CHANNELS)

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# Fine-tuning: unfreeze the top N layers of the base model
FINE_TUNE_AT = 140          # ResNet50 has 175 layers total
FINE_TUNE_LR = 1e-5
FINE_TUNE_EPOCHS = 10

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
ROTATION_RANGE = 10
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
FILL_MODE = "nearest"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
VERBOSE = 1
