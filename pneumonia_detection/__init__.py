"""
AI-Powered Disease Detection from Chest X-ray Images (Pneumonia)
=================================================================
A deep learning pipeline for binary classification of chest X-ray
images into NORMAL and PNEUMONIA categories.

Dataset: Chest X-Ray Images (Pneumonia) — Kaggle
  https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Model: Transfer learning with ResNet50 (ImageNet pre-trained weights)
"""

from pneumonia_detection.model import build_model
from pneumonia_detection.predict import predict_image

__all__ = ["build_model", "predict_image"]
