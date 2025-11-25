import torch
import json
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plant_disease_classifier import PlantDiseaseModel, predict_image
import warnings

warnings.filterwarnings("ignore")


def load_model_resources():
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
    model.load_state_dict(torch.load(
        config["model_path"], map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, transform, label_encoder, class_names, device


def predict(image_path, model, transform, label_encoder, device):
    # Runs prediction on a real image path
    class_name = predict_image(
        model, image_path, transform, device, label_encoder
    )
    return class_name


def main():
    # Load model + resources
    try:
        model, transform, label_encoder, class_names, device = load_model_resources()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Path of sample image to test
    image_path = r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\Plant-Disease-Classifier\images\examples\Potato_Late_blight.jpeg"

    if not os.path.exists(image_path):
        print("Error: Image file not found!")
        return

    # Run prediction
    class_name = predict(image_path, model, transform, label_encoder, device)

    print("Predicted Class:", class_name)


if __name__ == "__main__":
    main()
