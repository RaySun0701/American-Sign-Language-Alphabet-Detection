from tensorflow import keras
import numpy as np
import cv2
from pathlib import Path

# ---- Resolve model path (portable) ----
# Base directory = the folder that contains this file (app/)
BASE_DIR = Path(__file__).resolve().parent

# Primary location: app/ASL.h5
MODEL_PATH = BASE_DIR / "ASL.h5"

# Fallback location: project-root/ML_model/ASL.h5 (optional, if you keep a copy there)
if not MODEL_PATH.exists():
    alt_path = BASE_DIR.parent / "ML_model" / "ASL.h5"
    if alt_path.exists():
        MODEL_PATH = alt_path
    else:
        raise FileNotFoundError(
            "ASL.h5 not found. Expected at:\n"
            f" - {BASE_DIR / 'ASL.h5'}\n"
            f" - {BASE_DIR.parent / 'ML_model' / 'ASL.h5'}\n"
            "Please ensure the model file exists in one of these paths."
        )

# Load the trained model
model = keras.models.load_model(str(MODEL_PATH))

# Class labels in index order returned by the model
class_names = [
    'A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

def image_pre(path: str) -> np.ndarray:
    """Load an image from `path`, resize to 128x128, convert BGR->RGB,
    and return a 4D batch tensor of shape (1, 128, 128, 3)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image at: {path}")
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = np.array(img).reshape(-1, 128, 128, 3)
    return data

def predict(data: np.ndarray) -> str:
    """Run model prediction on a single-image batch and return the class name."""
    probs = model.predict(data)[0]          # shape: (num_classes,)
    idx = int(np.argmax(probs))
    return class_names[idx]

