import os
import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Plant Disease Diagnostic API")

# --- 1. Load Species Model & Labels ---
SPECIES_MODEL_PATH = "models/species_classifier.keras"
SPECIES_LABELS_PATH = "models/species_labels.txt"

if not os.path.exists(SPECIES_MODEL_PATH) or not os.path.exists(SPECIES_LABELS_PATH):
    raise RuntimeError("Species model or labels file not found in 'models/' directory.")

species_model = tf.keras.models.load_model(SPECIES_MODEL_PATH)
with open(SPECIES_LABELS_PATH, "r") as f:
    SPECIES_CLASSES = [line.strip() for line in f.readlines()]

# --- 2. Registry of Crop Disease Models ---
# Maps detected species name to (model_path, class_names_list)
DISEASE_REGISTRY = {
    "apple": {
        "model_path": "models/apple_classifier.keras",
        "classes": [
            "Apple Scab",
            "Black Rot",
            "Cedar Apple Rust",
            "Healthy Leaf"
        ]
    },
    "tomato": {
        "model_path": "models/tomato_classifier.keras",
        "classes": [
            "Bacterial Spot",
            "Early Blight",
            "Late Blight",
            "Leaf Mold",
            "Septoria Leaf Spot",
            "Spider Mites",
            "Target Spot",
            "Tomato Yellow Leaf Curl Virus",
            "Tomato Mosaic Virus",
            "Healthy Leaf"
        ]
    },
    "potato": {
        "model_path": "models/potato_classifier.keras",
        "classes": [
            "Early Blight",
            "Late Blight",
            "Healthy Leaf"
        ]
    },
    "cherry": {
        "model_path": "models/cherry_classifier.keras",
        "classes": [
            "Powdery Mildew",
            "Healthy Leaf"
        ]
    },
    "grape": {
        "model_path": "models/grape_classifier.keras",
        "classes": [
            "Black Rot",
            "Esca (Black Measles)",
            "Leaf Blight (Isariopsis Leaf Spot)",
            "Healthy Leaf"
        ]
    },
    "corn": {
        "model_path": "models/corn_classifier.keras",
        "classes": [
            "Cercospora Leaf Spot Gray Leaf Spot",
            "Common Rust",
            "Northern Leaf Blight",
            "Healthy Leaf"
        ]
    },
    "pepper": {
        "model_path": "models/pepper_classifier.keras",
        "classes": [
            "Bacterial Spot",
            "Healthy Leaf"
        ]
    },
    "peach": {
        "model_path": "models/peach_classifier.keras",
        "classes": [
            "Bacterial Spot",
            "Healthy Leaf"
        ]
    },
    "strawberry": {
        "model_path": "models/strawberry_classifier.keras",
        "classes": [
            "Leaf Scorch",
            "Healthy Leaf"
        ]
    }
}

# Cache loaded disease models in memory to avoid disk I/O on every request
loaded_disease_models = {}
for crop, meta in DISEASE_REGISTRY.items():
    if os.path.exists(meta["model_path"]):
        loaded_disease_models[crop] = tf.keras.models.load_model(meta["model_path"])

# --- 3. Preprocessing Helpers ---
def preprocess_image(image_bytes: bytes, target_size: tuple):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    resized = cv2.resize(img_np, target_size)
    return np.expand_dims(resized, axis=0)

# --- 4. Prediction Endpoint ---
@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    contents = await file.read()
    
    # ── Stage 1: Detect Plant Species (224x224) ──
    species_input = preprocess_image(contents, (224, 224))
    species_preds = species_model.predict(species_input, verbose=0)[0]
    top_species_idx = int(np.argmax(species_preds))
    detected_species = SPECIES_CLASSES[top_species_idx]
    species_conf = float(species_preds[top_species_idx]) * 100

    response_payload = {
        "species": {
            "name": detected_species.capitalize(),
            "confidence": round(species_conf, 2),
            "all_scores": {
                SPECIES_CLASSES[i].capitalize(): round(float(species_preds[i]) * 100, 2)
                for i in range(len(SPECIES_CLASSES))
            }
        },
        "disease": None
    }

    # ── Stage 2: Detect Condition via Specialist Model (256x256) ──
    norm_species = detected_species.lower()
    if norm_species in loaded_disease_models:
        d_model = loaded_disease_models[norm_species]
        d_classes = DISEASE_REGISTRY[norm_species]["classes"]

        disease_input = preprocess_image(contents, (256, 256))
        disease_preds = d_model.predict(disease_input, verbose=0)[0]
        top_disease_idx = int(np.argmax(disease_preds))
        
        disease_name = d_classes[top_disease_idx]
        disease_conf = float(disease_preds[top_disease_idx]) * 100
        is_healthy = "healthy" in disease_name.lower()

        response_payload["disease"] = {
            "status": "Healthy" if is_healthy else "Diseased",
            "diagnosis": disease_name,
            "confidence": round(disease_conf, 2),
            "is_healthy": is_healthy
        }
    else:
        response_payload["disease"] = {
            "status": "Unknown",
            "diagnosis": f"Specialist disease model for '{detected_species}' is not yet deployed.",
            "confidence": 0.0,
            "is_healthy": False
        }

    return JSONResponse(content=response_payload)


# --- 5. Supported Species Endpoint ---
@app.get("/api/supported-species")
async def get_supported_species():
    """Return all recognizable plant species and their diagnostic status."""
    catalog = []
    for sp in sorted(SPECIES_CLASSES):
        has_disease_model = sp.lower() in loaded_disease_models
        catalog.append({
            "name": sp.capitalize(),
            "has_specialist": has_disease_model,
            "conditions": (
                DISEASE_REGISTRY[sp.lower()]["classes"]
                if has_disease_model else []
            )
        })
    return JSONResponse(content={"species": catalog})

# Serve the static UI files
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")