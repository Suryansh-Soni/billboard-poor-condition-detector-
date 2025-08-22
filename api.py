# api.py
import os
import json
from io import BytesIO
from typing import Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# ✅ Use the re-saved Keras v3-compatible model
MODEL_PATH = "final_hazard_detector_3class_resaved.keras"
model = load_model(MODEL_PATH)

# ✅ Class label setup
try:
    with open("class_indices.json", "r") as f:
        class_indices: Dict[str, int] = json.load(f)
    class_names = [None] * len(class_indices)
    for label, idx in class_indices.items():
        class_names[idx] = label
except FileNotFoundError:
    class_names = ["rust", "broken_frame", "safe"]  # ⚠️ Must match training order

IMG_SIZE = (224, 224)

# ✅ FastAPI app
app = FastAPI(title="Hazard Detector API", version="1.0.0")

# ✅ Allow all CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Image preprocessing
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ✅ Health check
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": True, "classes": class_names}

# ✅ Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()
        img = Image.open(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    x = preprocess_image(img)
    preds = model.predict(x)
    probs = preds[0].astype(float)
    idx = int(np.argmax(probs))

    return {
        "predicted_class": class_names[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))}
    }

# ✅ Entry point (Render or local)
if __name__ == "__main__":
    import uvicorn

    # 🔧 Use the port provided by Render (if available), fallback to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
