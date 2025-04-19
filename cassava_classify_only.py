#!/usr/bin/env python3
"""
Cassava Leaf Disease Classification Only Inference
Classify pre-cropped cassava leaf images using a trained classification model.
"""
import os
import glob
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import argparse

# Update this list to match your training classes
DISEASE_CLASSES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]

def load_classification_model(model_path):
    model_name = os.path.basename(model_path)
    if 'resnet50' in model_name.lower():
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(DISEASE_CLASSES))
    else:
        raise ValueError("Only resnet50 supported in this example. Update as needed.")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def classify_leaves_in_folder(model_path, image_folder, output_folder="classification_results", num_images=None):
    print(f"Loading classification model from {model_path}...")
    model = load_classification_model(model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_paths += glob.glob(os.path.join(image_folder, "*.jpeg"))
    image_paths += glob.glob(os.path.join(image_folder, "*.png"))
    if num_images and len(image_paths) > num_images:
        image_paths = image_paths[:num_images]
    print(f"Processing {len(image_paths)} images...")
    os.makedirs(output_folder, exist_ok=True)
    for i, img_path in enumerate(image_paths):
        print(f"Classifying image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = DISEASE_CLASSES[pred_idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        result_path = os.path.join(output_folder, f"{base_name}_class.txt")
        with open(result_path, 'w') as f:
            f.write(f"Predicted class: {pred_class}\n")
        print(f"Result saved to {result_path}")
    print(f"All images processed. Results saved to {output_folder}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cassava Leaf Disease Classification Only Inference')
    parser.add_argument('--classification-model', type=str, required=True, help='Path to classification model')
    parser.add_argument('--test-folder', type=str, required=True, help='Folder containing cropped leaf images')
    parser.add_argument('--output-folder', type=str, default='classification_results', help='Folder to save classification results')
    parser.add_argument('--num-images', type=int, default=None, help='Number of images to process')
    args = parser.parse_args()
    classify_leaves_in_folder(args.classification_model, args.test_folder, args.output_folder, args.num_images)
