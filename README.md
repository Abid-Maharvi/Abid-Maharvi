

#  Muhammad Abid Hussain

**Lab Engineer | Researcher | Deep Learning & EEG Specialist**
📍 Quetta, Pakistan | 📧 [abidmaharvi@hotmail.com](mailto:abidmaharvi@hotmail.com) | 📞 +92 300 4426112
🔗 [Google Scholar](https://scholar.google.com.pk/citations?user=K-B3pSIAAAAJ&hl=en) · [LinkedIn](https://www.linkedin.com/in/engr-muhammad-abid-hussain-60b6a2124/)

---

## 🧠 About Me

Motivated and detail-oriented researcher with a strong interest in **EEG signal processing**, **Brain-Computer Interface (BCI)**, and **Deep Learning for Neural Decoding**.
My focus lies in understanding how the human brain perceives language through neural activity — integrating **AI, neuroscience, and signal processing** for **biometric and cognitive analysis applications**.

---

## 🎓 Education

**MS Computer Science** — *National University of Sciences and Technology (NUST), Quetta*
*(2023 – 2025)*

* **Thesis:** *EEG-Based Identification of Native and Non-Native Language Perception Using Deep Learning*
* Focus Areas: EEG Signal Processing, Deep Learning, Cognitive Neuroscience

**B.Sc. Computer Engineering** — *Bahauddin Zakariya University (BZU), Multan*
*(2012 – 2017)*

* Thesis: *Innovative Railway Track Surveying with Sensors and Wireless Communication*

---

## 🧪 Research Interests

* EEG Signal Processing & Neural Decoding
* Brain-Computer Interfaces (BCI)
* Deep Learning for Biomedical Signal Classification
* Cognitive and Affective Computing
* AI-Driven Healthcare & Neuroscience
* Pattern Recognition in EEG Data

---

## 🧬 Publications

1. **EEG-Based Identification of Native and Non-Native Language Perception Using Deep Learning**
   *2025 IEEE International Conference on Communication Technologies (ComTech)*
   [IEEE Xplore](https://ieeexplore.ieee.org/document/11034481)

2. **Study on Fixed and Dynamic Spectrum Access Models for Cellular Networks**
   *Journal of Telecommunications and the Digital Economy (JTDE), 2022*
   [DOI: 10.18080/jtde.v10n2.395](https://doi.org/10.18080/jtde.v10n2.395)

---

## 💼 Professional Experience

**Lab Engineer** — *National University of Sciences and Technology (NUST), Quetta*
*Jul 2019 – Present*

**A/CMS Coordinator (ERP/AM App)** — *NUST Quetta*
*Feb 2024 – Sep 2024*

**IT Support Officer** — *MGA Industries Pvt. Ltd., Lahore*
*May 2019 – Jul 2019*

**Vehicle Inspector (IT)** — *OPUS Inspection Pvt. Ltd., Lahore*
*Feb 2018 – Apr 2019*
*(Government of Punjab Vehicle Inspection Project)*

**Customer Support Executive (IT)** — *PTCL, Lahore*
*Oct 2017 – Feb 2018*

---

## 🧰 Technical Skills

* **Programming:** Python, MATLAB, C for microcontrollers, VHDL
* **Frameworks:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy
* **Tools:** Jupyter, WFDB, Odoo ERP, Git/GitHub
* **Signal Processing:** EEG Filtering, Time-Frequency Analysis, Feature Extraction
* **Embedded Systems:** Arduino, ESP32, IoT Applications
* **Soft Skills:** Research, Analytical Thinking, Teaching, Team Collaboration

---

## 🧑‍🏫 Projects Supervised

* Smart Object Detection Glasses for Visually Impaired (ML-based)
* IoT-Based Real-Time Water Quality Monitoring System
* Gesture-Controlled Robotic Arm using Flex Sensors
* Smart Home Automation (ESP32 + IoT)
* Obstacle Detection Robot (Arduino + Ultrasonic Sensors)

---

## 🤝 Research Collaboration & Committees

* Committee Member — National Research Proposals

  * *RDIA Saudi Arabia Grant (PKR 419M)*
  * *National Center for Artificial Intelligence (NCAI)*
  * *IGNITE-funded Precision Agriculture Project*

---

## 🌐 Languages

* **English:** C2 (Proficient)
* **Urdu:** Native

---

## 📫 Contact

**Muhammad Abid Hussain**
National University of Sciences and Technology (NUST), Quetta
📧 [abidmaharvi@hotmail.com](mailto:abidmaharvi@hotmail.com)
🔗 [Google Scholar](https://scholar.google.com.pk/citations?user=K-B3pSIAAAAJ&hl=en) | [LinkedIn](https://www.linkedin.com/in/engr-muhammad-abid-hussain-60b6a2124/)


---

## 🫁 AI-Powered Disease Detection from Chest X-ray Images (Pneumonia)

### Overview

A deep learning pipeline for **binary classification** of chest X-ray images
into **NORMAL** and **PNEUMONIA** categories, built with Python / TensorFlow
and transfer learning on **ResNet50** (ImageNet pre-trained weights).

### Project Structure

```
pneumonia_detection/
├── __init__.py        # Package entry-point
├── config.py          # Hyper-parameters, paths, and constants
├── preprocess.py      # Data loading, augmentation, and single-image preprocessing
├── model.py           # ResNet50-based classifier (transfer learning)
├── train.py           # Two-stage training pipeline
├── evaluate.py        # Metrics, confusion matrix, ROC curve
└── predict.py         # Single-image and batch inference

app.py                 # Streamlit web demo
requirements.txt       # Python dependencies
tests/
└── test_pneumonia_detection.py   # Unit tests
```

### Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle  
<https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>

Download and extract the dataset so that the directory layout is:

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python -m pneumonia_detection.train
```

Training runs in **two stages**:

| Stage | Base | LR | Purpose |
|-------|------|----|---------|
| 1 — Feature extraction | Frozen | 1e-4 | Train the custom head only |
| 2 — Fine-tuning | Partially unfrozen (top layers) | 1e-5 | Refine the whole network |

The best model (by validation AUC) is saved to `models/pneumonia_resnet50.keras`.

### Evaluation

```bash
python -m pneumonia_detection.evaluate
```

Prints accuracy, AUC, precision, recall, and F1 on the test set, and saves
`confusion_matrix.png` and `roc_curve.png` under `models/`.

### Inference

```bash
# Single image from the command line
python -m pneumonia_detection.predict --image path/to/xray.jpg

# From Python
from pneumonia_detection.predict import predict_image
result = predict_image("path/to/xray.jpg")
# {'label': 'PNEUMONIA', 'confidence': 0.92, 'probability': 0.92}
```

### Web Application (Streamlit)

```bash
streamlit run app.py
```

Upload any chest X-ray image in the browser and receive an instant prediction
with a confidence score.

> ⚠️ **Disclaimer:** For research and educational purposes only. Not a
> substitute for professional medical advice.

### Running Tests

```bash
python -m pytest tests/ -v
```

### Model Architecture

```
Input (224 × 224 × 3)
       │
  ResNet50 (ImageNet, top removed)
       │
  GlobalAveragePooling2D
       │
  Dense(256, relu)
       │
  Dropout(0.5)
       │
  Dense(1, sigmoid)  →  P(PNEUMONIA)
```



© 2025 Muhammad Abid Hussain — All rights reserved.
For academic and research purposes only.

---
