import os
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Your Hugging Face repository
REPO_ID = "PranayS909/plant-diseases-models"
MODELS_DIR = "models"

# Stage 1: Species identification
SPECIES_MODEL_FILE = "species_classifier.keras"
SPECIES_LABELS_FILE = "species_labels.txt"

# Stage 2: Specialist crop disease models (filenames matched exactly to HF repo)
DISEASE_REGISTRY = {
    "apple": {
        "filename": "apple_classifier.keras",
        "classes": [
            "Apple Scab",
            "Black Rot",
            "Cedar Apple Rust",
            "Healthy Leaf"
        ]
    },
    "cherry": {
        "filename": "cherry_classifier.keras",
        "classes": [
            "Powdery Mildew",
            "Healthy"
        ]
    },
    "corn": {
        "filename": "corn_classifier.keras",
        "classes": [
            "Cercospora Leaf Spot (Gray Leaf Spot)",
            "Common Rust",
            "Northern Leaf Blight",
            "Healthy"
        ]
    },
    "grape": {
        "filename": "grape_classifier.keras",
        "classes": [
            "Black Rot",
            "Esca (Black Measles)",
            "Leaf Blight (Isariopsis Leaf Spot)",
            "Healthy"
        ]
    },
    "peach": {
        "filename": "peach_classifier.keras",
        "classes": [
            "Bacterial Spot",
            "Healthy"
        ]
    },
    "pepper": {
        "filename": "pepper_classifier.keras",
        "classes": [
            "Bacterial Spot",
            "Healthy"
        ]
    },
    "potato": {
        "filename": "potato_classifier.keras",
        "classes": [
            "Early Blight",
            "Late Blight",
            "Healthy"
        ]
    },
    "strawberry": {
        "filename": "strawberry_classifier.keras",
        "classes": [
            "Leaf Scorch",
            "Healthy"
        ]
    },
    "tomato": {
        "filename": "tomato_classifier.keras",
        "classes": [
            "Bacterial Spot",
            "Early Blight",
            "Late Blight",
            "Leaf Mold",
            "Septoria Leaf Spot",
            "Spider Mites",
            "Target Spot",
            "Yellow Leaf Curl Virus",
            "Mosaic Virus",
            "Healthy"
        ]
    }
}


def ensure_asset_downloaded(filename: str) -> str:
    """Download from Hugging Face if not present locally."""
    local_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from Hugging Face ({REPO_ID})...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR
        )
    return local_path


def load_all_models():
    """
    Ensures all 10 models + labels are cached locally, then loads them.
    Returns:
        species_model, species_classes, loaded_disease_models
    """
    # 1. Load Species Classifier
    species_model_path = ensure_asset_downloaded(SPECIES_MODEL_FILE)
    species_labels_path = ensure_asset_downloaded(SPECIES_LABELS_FILE)

    print("Loading Stage 1 Species Classifier...")
    species_model = tf.keras.models.load_model(species_model_path)

    with open(species_labels_path, "r") as f:
        species_classes = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Load Stage 2 Specialist Models
    loaded_disease_models = {}
    print("Loading Stage 2 Specialist Models...")
    for crop_name, meta in DISEASE_REGISTRY.items():
        try:
            model_path = ensure_asset_downloaded(meta["filename"])
            loaded_disease_models[crop_name] = tf.keras.models.load_model(model_path)
            print(f"  Loaded {crop_name.capitalize()} model.")
        except Exception as e:
            print(f"  Failed loading {crop_name} model: {e}")

    return species_model, species_classes, loaded_disease_models