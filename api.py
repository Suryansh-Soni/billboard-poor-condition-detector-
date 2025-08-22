# api.py
import json
from io import BytesIO
from typing import Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# ✅ Use the re-saved model name here
MODEL_PATH = "final_hazard_detector_3class_resaved.keras"
model = load_model(MODEL_PATH)

# --- Class label handling ---
try:
    with open("class_indices.json", "r") as f:
        class_indices: Dict[str, int] = json.load(f)  # e.g. {"broken_frame": 0, "rust": 1, "safe": 2}
    class_names = [None] * len(class_indices)
    for label, idx in class_indices.items():
        class_names[idx] = label
except FileNotFoundError:
    class_names = ["rust", "broken_frame", "safe"]  # <- Ensure this matches training order

IMG_SIZE = (224, 224)

app = FastAPI(title="Hazard Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Consider restricting in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": True, "classes": class_names}

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

# 🔧 Only runs locally (not needed for Render)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
