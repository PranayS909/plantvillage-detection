# Leaf Disease Detection Using Deep Convolutional Neural Networks

An end-to-end computer vision pipeline to classify plant leaf diseases using TensorFlow/Keras and the PlantVillage dataset. The project supports isolated single-crop models (e.g., Apple, Tomato, Potato) as well as multi-crop classification architectures with automated data ingestion, real-time augmentation, and callback-driven training.

---

## Table of Contents

1. [Overview](https://www.google.com/search?q=%23overview)
2. [Project Structure](https://www.google.com/search?q=%23project-structure)
3. [Environment Setup](https://www.google.com/search?q=%23environment-setup)
4. [Dataset Ingestion](https://www.google.com/search?q=%23dataset-ingestion)
5. [Model Architecture](https://www.google.com/search?q=%23model-architecture)
6. [Training & Evaluation Pipeline](https://www.google.com/search?q=%23training--evaluation-pipeline)
7. [Inference](https://www.google.com/search?q=%23inference)
8. [Common Issues & Solutions](https://www.google.com/search?q=%23common-issues--solutions)

---

## Overview

The primary goal is the automated detection of foliar fungal, bacterial, and viral diseases from leaf imagery. Key features:

* **Zero-leakage split strategy:** Deterministic generation of training, validation, and held-out test splits.
* **In-graph augmentations:** Real-time affine transformations (rotations, flips, zoom) baked directly into the model to avoid CPU/I/O bottlenecks.
* **Regularization & Stability:** Batch Normalization, Dropout, and `GlobalAveragePooling2D` to prevent overfitting on static laboratory backgrounds.
* **Callback-guided optimization:** Early stopping and adaptive learning rate decay (`ReduceLROnPlateau`).

---

## Project Structure

```text
plantvillage-detection/
├── data/
│   └── raw/
│       ├── apple_color/
│       │   ├── Apple___Apple_scab/
│       │   ├── Apple___Black_rot/
│       │   ├── Apple___Cedar_apple_rust/
│       │   └── Apple___healthy/
│       ├── cherry_color/
│       ├── corn_color/
│       ├── grape_color/
│       ├── peach_color/
│       ├── pepper_color/
│       ├── potato_color/
│       ├── strawberry_color/
│       └── tomato_color/
├── logs/                   # TensorBoard event logs
├── models/                 # Serialized model checkpoints (.keras)
├── detect.ipynb            # Interactive development notebook
├── requirements.txt
└── README.md

```

---

## Environment Setup

### 1. Prerequisites

* Python 3.10+
* CUDA-compatible GPU (recommended) or CPU
* Git / WSL2 (for Linux/Windows hybrid workflows)

### 2. Virtual Environment Installation

```bash
# Clone the project repository
git clone <your-repo-url>
cd plantvillage-detection

# Create and activate a virtual environment
python3 -m venv ml-env
source ml-env/bin/activate  # On Windows: ml-env\Scripts\activate

# Install required dependencies
pip install --upgrade pip
pip install tensorflow opencv-python pillow matplotlib numpy

```

---

## Dataset Ingestion

The images are sourced from the [PlantVillage Dataset](https://www.google.com/search?q=https://github.com/spMohanty/PlantVillage-Dataset). Use Git Sparse Checkout to download only the target crop subsets without pulling the entire ~54,000-image repository:

```bash
# 1. Create target directories for each crop
mkdir -p data/raw/{apple_color,cherry_color,corn_color,grape_color,peach_color,pepper_color,potato_color,strawberry_color,tomato_color}

# 2. Clone repository metadata only (fast, no images downloaded yet)
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/spMohanty/PlantVillage-Dataset.git temp_repo
cd temp_repo

# 3. Pull only raw/color directory
git sparse-checkout init --cone
git sparse-checkout set raw/color
git checkout master

# 4. Copy each crop into its respective target directory
cp -r raw/color/Apple* ../data/raw/apple_color/
cp -r raw/color/Cherry* ../data/raw/cherry_color/
cp -r raw/color/Corn* ../data/raw/corn_color/
cp -r raw/color/Grape* ../data/raw/grape_color/
cp -r raw/color/Peach* ../data/raw/peach_color/
cp -r raw/color/Pepper* ../data/raw/pepper_color/
cp -r raw/color/Potato* ../data/raw/potato_color/
cp -r raw/color/Strawberry* ../data/raw/strawberry_color/
cp -r raw/color/Tomato* ../data/raw/tomato_color/

# 5. Clean up temporary repository
cd ..
rm -rf temp_repo

```

---

## Model Architecture

The default baseline is an optimized convolutional neural network designed to balance parameter efficiency with representational capacity:

```text
Input (256, 256, 3)
   │
   ▼
[Rescaling (1/255)] ────► [RandomFlip / RandomRotation / RandomZoom]
   │
   ▼
Conv2D (32, 3x3) ──────► BatchNorm ──► MaxPool2D (2x2)
   │
   ▼
Conv2D (64, 3x3) ──────► BatchNorm ──► MaxPool2D (2x2)
   │
   ▼
Conv2D (128, 3x3) ─────► BatchNorm ──► MaxPool2D (2x2)
   │
   ▼
GlobalAveragePooling2D
   │
   ▼
Dense (128, ReLU) ─────► Dropout (0.4)
   │
   ▼
Dense (num_classes, Softmax)

```

---

## Training & Evaluation Pipeline

The notebook `detect.ipynb` is structured into isolated execution blocks:

1. **Setup & Hyperparameters:** Defines `IMG_SIZE = (256, 256)`, `BATCH_SIZE = 32`, and targets the chosen directory (e.g., `DATA_DIR = 'data/raw/apple_color'`).
2. **Dataset Partitioning:**
* 80% Training
* 10% Validation
* 10% Unseen Test


3. **I/O Pipeline Optimization:** Leverages `.cache()` and `prefetch(buffer_size=tf.data.AUTOTUNE)` to prevent I/O bottlenecks.
4. **Callbacks Configured:**
* `TensorBoard(log_dir='logs')`
* `EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)`
* `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)`


5. **Loss & Evaluation:** Evaluated using `sparse_categorical_crossentropy` and accuracy on the held-out test split.

---

## Inference

To run inference on a single leaf image using the serialized `.keras` model:

```python
import os
import cv2
import numpy as np
import tensorflow as tf

# Load serialized model
model = tf.keras.models.load_model('models/apple_leaf_classifier.keras')

# Preprocess image
IMG_SIZE = (256, 256)
image_path = 'path/to/test_leaf.jpg'

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img, IMG_SIZE)
input_tensor = np.expand_dims(resized, axis=0)

# Predict
predictions = model.predict(input_tensor)
class_idx = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print(f"Predicted class: {class_idx} ({confidence:.2f}%)")

```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
| --- | --- | --- |
| **`svn: path not found`** | GitHub deprecated SVN access. | Use `git sparse-checkout` instead. |
| **`bash: syntax error near unexpected token '('`** | Crop names like `Cherry_(including_sour)` contain unquoted special characters in bash. | Wrap folder names in quotes or pull `raw/color` in bulk, then copy. |
| **Training stops early (e.g., Epoch 6)** | `EarlyStopping` triggered because `val_loss` did not beat epoch 1 within `patience=5`. | Increase `patience=8` or `10`, reduce learning rate to `3e-4`, or monitor `val_accuracy`. |
| **Data leakage across splits** | Using `.take()` and `.skip()` on dynamic/shuffled datasets. | Use `image_dataset_from_directory` with fixed `validation_split`, `seed`, and `subset`. |