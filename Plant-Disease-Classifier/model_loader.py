import torch
import json
import pickle
from plant_disease_classifier import PlantDiseaseModel, predict_image

def load_model_resources():
    """Loads the model, transformations, and label encoder."""
    with open("models/model_config.json", "r") as f:
        config = json.load(f)

    with open(config["class_names_path"], "r") as f:
        class_names = json.load(f)

    with open(config["label_encoder_path"], "rb") as f:
        label_encoder = pickle.load(f)

    with open(config["transform_path"], "rb") as f:
        transform = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes=len(class_names))
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.to(device)
    model.eval()

    return model, transform, label_encoder, class_names, device

def predict(image_bytes, model, transform, label_encoder, device):
    """Runs inference on the given image."""
    with open("temp_upload.jpg", "wb") as f:
        f.write(image_bytes)

    class_name = predict_image(model, "temp_upload.jpg", transform, device, label_encoder)
    return class_name