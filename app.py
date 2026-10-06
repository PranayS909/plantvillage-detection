from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from PIL import Image
import io

from model_loader import load_all_models, DISEASE_REGISTRY

# In-memory application model cache
models_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all models once
    species_model, species_classes, disease_models = load_all_models()
    models_cache["species_model"] = species_model
    models_cache["species_classes"] = species_classes
    models_cache["disease_models"] = disease_models
    yield
    # Shutdown: Clear cache
    models_cache.clear()

app = FastAPI(title="LeafScan AI API", lifespan=lifespan)

# Helper for preprocessing
def preprocess_image(image_bytes: bytes, target_size: tuple):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    resized = cv2.resize(img_np, target_size)
    return np.expand_dims(resized, axis=0)

@app.get("/api/supported-species")
async def get_supported_species():
    species_classes = models_cache.get("species_classes", [])
    disease_models = models_cache.get("disease_models", {})
    
    catalog = []
    for sp in sorted(species_classes):
        has_specialist = sp.lower() in disease_models
        catalog.append({
            "name": sp.capitalize(),
            "has_specialist": has_specialist,
            "conditions": (
                DISEASE_REGISTRY[sp.lower()]["classes"]
                if has_specialist else []
            )
        })
    return JSONResponse(content={"species": catalog})

@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    contents = await file.read()
    species_model = models_cache["species_model"]
    species_classes = models_cache["species_classes"]
    disease_models = models_cache["disease_models"]

    # Stage 1: Detect Species (224x224)
    species_input = preprocess_image(contents, (224, 224))
    species_preds = species_model.predict(species_input, verbose=0)[0]
    top_species_idx = int(np.argmax(species_preds))
    detected_species = species_classes[top_species_idx]
    species_conf = float(species_preds[top_species_idx]) * 100

    response = {
        "species": {
            "name": detected_species.capitalize(),
            "confidence": round(species_conf, 2)
        },
        "disease": None
    }

    # Stage 2: Specialist Diagnosis (256x256)
    crop_key = detected_species.lower()
    if crop_key in disease_models:
        d_model = disease_models[crop_key]
        d_classes = DISEASE_REGISTRY[crop_key]["classes"]

        disease_input = preprocess_image(contents, (256, 256))
        disease_preds = d_model.predict(disease_input, verbose=0)[0]
        top_disease_idx = int(np.argmax(disease_preds))
        diagnosis = d_classes[top_disease_idx]
        d_conf = float(disease_preds[top_disease_idx]) * 100
        is_healthy = "healthy" in diagnosis.lower()

        response["disease"] = {
            "diagnosis": diagnosis,
            "confidence": round(d_conf, 2),
            "is_healthy": is_healthy
        }

    return JSONResponse(content=response)

app.mount("/", StaticFiles(directory="static", html=True), name="static")