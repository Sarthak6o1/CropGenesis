from fastapi import FastAPI, File, UploadFile
from model_loader import load_model_resources, predict
import io

app = FastAPI()

# Load model and resources
model, transform, label_encoder, class_names, device = load_model_resources()

@app.post("/predict/")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        class_name = predict(image_bytes, model, transform, label_encoder, device)
        return {"prediction": class_name}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Plant Disease Classifier API is running!"}
