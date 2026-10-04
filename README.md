# 🍃 LeafScan AI — Two-Stage Plant Pathology Diagnostic System

An end-to-end deep learning diagnostic tool built on the **PlantVillage** dataset. The system employs a **two-stage hierarchical inference architecture** to first identify the plant species and then route the leaf to a dedicated specialist model for targeted foliar disease classification.

Includes a lightweight **FastAPI** backend with a responsive **Tailwind CSS** web client supporting live camera capture, drag-and-drop uploads, and dynamic crop pathology tracking.

---

## 📌 Architecture Overview


```

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
│ - Apple, Tomato, Corn, Potato, etc.    │
└──────────────────┬─────────────────────┘
│
▼
┌────────────────────────────────────────┐
│ Stage 2: Crop Pathology Specialist     │
│ Architecture: Custom CNN / Fine-Tuned  │
│ Output: Disease vs. Healthy (256x256)  │
└──────────────────┬─────────────────────┘
│
▼
Final Diagnostic & Advisory

```

### Why Hierarchical?
- **Simplified Decision Boundaries:** Individual specialist models only need to differentiate between diseases affecting that specific host species (e.g., distinguishing *Early Blight* vs. *Late Blight* without confusing them with *Apple Scab*).
- **Modular Scalability:** New crops can be added or retrained independently without retraining the entire catalog.
- **Resource Efficient:** Inference loads the Stage 1 gatekeeper and conditionally engages the specific Stage 2 model.

---

## 🌿 Species & Disease Coverage

| Plant Species | Stage 1 (Species ID) | Stage 2 (Disease Model) | Detectable Conditions |
| :--- | :---: | :---: | :--- |
| **Apple** | ✅ | ✅ Active | *Apple Scab*, *Black Rot*, *Cedar Apple Rust*, *Healthy* |
| **Cherry** | ✅ | 🔄 In Progress | — |
| **Corn (Maize)** | ✅ | 🔄 In Progress | — |
| **Grape** | ✅ | 🔄 In Progress | — |
| **Peach** | ✅ | 🔄 In Progress | — |
| **Pepper (Bell)**| ✅ | 🔄 In Progress | — |
| **Potato** | ✅ | 🔄 In Progress | — |
| **Strawberry** | ✅ | 🔄 In Progress | — |
| **Tomato** | ✅ | 🔄 In Progress | — |

---

## 🛠 Tech Stack & Environment

- **Deep Learning:** TensorFlow 2.x, Keras, OpenCV, NumPy, Scikit-learn
- **API & Backend:** FastAPI, Uvicorn, Python-Multipart
- **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript, FontAwesome
- **Environment:** Ubuntu on WSL2, CUDA-accelerated GPU runtime

---

## 📁 Repository Structure

```text
plantvillage-detection/
├── app.py                      # FastAPI application with two-stage pipeline
├── static/
│   └── index.html              # Diagnostic Web UI (Tailwind CSS + JS)
├── models/
│   ├── species_classifier.keras# Stage 1: MobileNetV2 species classifier
│   ├── species_labels.txt      # Stage 1: 9 class label mapping
│   ├── apple_leaf_classifier.keras # Stage 2: Apple disease specialist
│   └── .gitkeep
├── data/
│   └── raw/                    # PlantVillage image folders (ignored by git)
│       ├── apple_color/
│       ├── tomato_color/
│       └── ...
├── notebooks/
│   ├── 01_species_classifier.ipynb
│   └── 02_apple_disease_specialist.ipynb
├── reports/
│   ├── figures/                # Saved loss and accuracy curves
│   └── evaluation_summary.md   # Per-class precision, recall, and F1 logs
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

### 3. Configure GPU Memory Growth (WSL2 / Linux)

To prevent CUDA out-of-memory errors on shared VRAM systems, the training notebooks and application execute:

```python
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

```

### 4. Run the Web Application

Ensure your trained models exist in the `models/` directory, then start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

```

Access the UI at:

```text
http://localhost:8000

```

---

## 📊 Training Pipeline & Sampling Strategy

To train efficiently on local hardware without memory exhaustion:

1. **Deterministic Slicing:** A 25% balanced subset (~10,000 images across 40,000 raw samples) is extracted in memory across train (80%), validation (10%), and test (10%) splits.
2. **Transfer Learning:** The Stage 1 model freezes ImageNet weights on `MobileNetV2`, fine-tuning a custom classification head:
* `GlobalAveragePooling2D`
* `BatchNormalization` + `Dropout(0.3)`
* `Dense(128, activation='relu')`
* `Dense(9, activation='softmax')`


3. **Pipeline Optimization:** Data streaming uses prefetching (`AUTOTUNE`) and bounded shuffle buffers (`128`) instead of in-memory caching to fit WSL2 host RAM constraints.

---

## 📈 Evaluation

Run evaluation cells to output high-resolution performance plots to `reports/figures/` and append per-class classification metrics (`precision`, `recall`, `f1-score`, and `support`) to `reports/evaluation_summary.md`.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

```

<ElicitationsGroup message="Next steps for packaging and deploying:">
  <Elicitation label="Generate a Dockerfile for containerized deployment" query="Create a production-ready Dockerfile and docker-compose setup for this FastAPI plant disease app."/>
  <Elicitation label="Add a batch evaluation script for test sets" query="Write a standalone Python script to evaluate all models in models/ on test sets and update evaluation_summary.md."/>
</ElicitationsGroup>

```