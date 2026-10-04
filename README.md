# 🍃 LeafScan AI — Two-Stage Plant Pathology Diagnostic System

An end-to-end deep learning diagnostic tool built on the **PlantVillage** dataset. The system uses a **two-stage hierarchical inference architecture** to first identify the plant species and then route the leaf to a dedicated specialist model for targeted foliar disease classification.

The application includes a lightweight **FastAPI** backend paired with a responsive **Tailwind CSS** web client supporting live camera capture, drag-and-drop file uploads, confidence scoring, and dynamic pathology tracking.

---

## 📌 Architecture Overview

```text
User Input Image (Leaf Photo)
           │
           ▼
┌────────────────────────────────────────┐
│ Stage 1: Species Identifier            │
│ Architecture: MobileNetV2 (Pretrained) │
│ Output: 9 Crop Classes (224x224 input) │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Crop Router & Gatekeeper               │
│ - Selects Dedicated Crop Specialist    │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 2: Crop Pathology Specialist     │
│ Architecture: Fine-Tuned CNN / Custom  │
│ Output: Disease vs. Healthy (256x256)  │
│ Coverage: All 9 Target Crops Active    │
└──────────────────┬─────────────────────┘
                   │
                   ▼
      Final Diagnostic & Advisory

```

### Why Hierarchical?

* **Constrained Decision Boundaries:** Individual crop specialist models only differentiate between diseases specific to their host plant (e.g., distinguishing *Early Blight* vs. *Late Blight* without confusing them with *Apple Scab*).
* **Independent Maintenance:** Any single crop model can be retrained or upgraded without retraining or invalidating the rest of the catalog.
* **Resource Optimization:** Inference runs through the lightweight Stage 1 gatekeeper and conditionally invokes only the selected Stage 2 specialist.

---

## 🌿 Species & Disease Coverage

All 9 plant families have active Stage 1 species routing and dedicated Stage 2 pathology diagnostics:

| Plant Species | Stage 1 (Species ID) | Stage 2 (Disease Model) | Detectable Conditions |
| --- | --- | --- | --- |
| **Apple** | ✅ | ✅ Active | *Apple Scab*, *Black Rot*, *Cedar Apple Rust*, *Healthy* |
| **Cherry** | ✅ | ✅ Active | *Powdery Mildew*, *Healthy* |
| **Corn (Maize)** | ✅ | ✅ Active | *Cercospora Leaf Spot (Gray Leaf Spot)*, *Common Rust*, *Northern Leaf Blight*, *Healthy* |
| **Grape** | ✅ | ✅ Active | *Black Rot*, *Esca (Black Measles)*, *Leaf Blight (Isariopsis Leaf Spot)*, *Healthy* |
| **Peach** | ✅ | ✅ Active | *Bacterial Spot*, *Healthy* |
| **Pepper (Bell)** | ✅ | ✅ Active | *Bacterial Spot*, *Healthy* |
| **Potato** | ✅ | ✅ Active | *Early Blight*, *Late Blight*, *Healthy* |
| **Strawberry** | ✅ | ✅ Active | *Leaf Scorch*, *Healthy* |
| **Tomato** | ✅ | ✅ Active | *Bacterial Spot*, *Early Blight*, *Late Blight*, *Leaf Mold*, *Septoria Leaf Spot*, *Spider Mites*, *Target Spot*, *Yellow Leaf Curl Virus*, *Mosaic Virus*, *Healthy* |

---

## 🛠 Tech Stack & Environment

* **Deep Learning & Modeling:** TensorFlow 2.x, Keras, OpenCV, NumPy, Scikit-learn
* **Backend API:** FastAPI, Uvicorn, Python-Multipart
* **Frontend UI:** HTML5, Tailwind CSS, Vanilla JavaScript, FontAwesome
* **Environment:** Ubuntu on WSL2, CUDA-accelerated GPU runtime

---

## 📁 Repository Structure

```text
plantvillage-detection/
├── app.py                           # FastAPI application & two-stage router
├── static/
│   └── index.html                   # Diagnostic UI (Tailwind CSS + JS)
├── models/
│   ├── species_classifier.keras     # Stage 1: MobileNetV2 species classifier
│   ├── species_labels.txt           # Stage 1: 9 species label mapping
│   ├── apple_classifier.keras  # Stage 2: Apple specialist
│   ├── cherry_classifier.keras # Stage 2: Cherry specialist
│   ├── corn_classifier.keras   # Stage 2: Corn specialist
│   ├── grape_classifier.keras  # Stage 2: Grape specialist
│   ├── peach_classifier.keras  # Stage 2: Peach specialist
│   ├── pepper_classifier.keras # Stage 2: Pepper specialist
│   ├── potato_classifier.keras # Stage 2: Potato specialist
│   ├── strawberry_classifier.keras # Stage 2: Strawberry specialist
│   └── tomato_classifier.keras # Stage 2: Tomato specialist
├── data/
│   └── raw/                         # PlantVillage raw directories (ignored by git)
│       ├── apple_color/
│       ├── cherry_color/
│       ├── corn_color/
│       ├── grape_color/
│       ├── peach_color/
│       ├── pepper_color/
│       ├── potato_color/
│       ├── strawberry_color/
│       └── tomato_color/
├── notebooks/
│   ├── species_classifier.ipynb
│   └── crop_disease_training.ipynb
├── reports/
│   ├── figures/                     # Training loss and accuracy plots
│   └── evaluation_summary.md    # Evaluated metrics across all test sets
├── .gitignore
├── requirements.txt
└── README.md

```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/plantvillage-detection.git](https://github.com/your-username/plantvillage-detection.git)
cd plantvillage-detection

```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. GPU Memory Management (WSL2 / Linux)

To prevent CUDA host allocation and VRAM out-of-memory errors on shared hardware, notebooks and runtime scripts enforce memory growth:

```python
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

```

### 4. Run the Web Application

Ensure all model artifacts exist in the `models/` directory, then start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

```

Open your browser and navigate to:

```text
http://localhost:8000

```

---

## 📊 Training Pipeline & Sampling Strategy

To train efficiently on local hardware without memory exhaustion:

1. **In-Memory Balanced Slicing:** A deterministic 25% subset (~10,000 images across the 40,000 raw samples) is extracted in memory across train (80%), validation (10%), and test (10%) splits.
2. **Transfer Learning Backbone:** Stage 1 freezes ImageNet weights on `MobileNetV2` and trains a custom classification head:
* `GlobalAveragePooling2D()`
* `BatchNormalization()` + `Dropout(0.3)`
* `Dense(128, activation='relu')`
* `Dense(9, activation='softmax')`


3. **Pipeline Optimization:** Data streaming uses prefetching (`AUTOTUNE`) and bounded shuffle buffers (`128`) instead of aggressive RAM caching to stay well within WSL2 host memory limits.

---

## 📈 Evaluation & Metrics Logging

Model checkpoints are evaluated against held-out test splits. Metrics (precision, recall, F1-score, and support) for every crop condition are logged to `reports/all_plants_evaluation.csv`, and training history curves are saved to `reports/figures/`.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

```

<ElicitationsGroup message="Suggested follow-up tasks to wrap up the project:">
  <Elicitation label="Generate a multi-stage Docker container build" query="Provide a multi-stage Dockerfile and docker-compose.yml to deploy this FastAPI application with all 9 models."/>
  <Elicitation label="Create a health-check test script for all 9 models" query="Write an automated test script that verifies every model in models/ loads and runs a test inference without crashing."/>
</ElicitationsGroup>

```