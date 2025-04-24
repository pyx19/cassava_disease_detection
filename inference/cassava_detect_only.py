#!/usr/bin/env python3
"""
Cassava Leaf Detection Only Inference
Detect cassava leaves in images using a YOLO model and visualize/save results.
"""
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from matplotlib.patches import Rectangle
import argparse
import datetime
import re

def extract_yolo_version(model_path):
    """Extract YOLO version from model path"""
    filename = os.path.basename(model_path)
    match = re.search(r'(yolov\d+[a-z]?)', filename.lower())
    if match:
        return match.group(1)
    return "yolo"

def detect_leaves_in_folder(detection_model_path, image_folder, output_folder=None, num_images=None, conf_threshold=0.4):
    # Extract model version and create timestamp
    yolo_version = extract_yolo_version(detection_model_path)
    timestamp = datetime.datetime.now().strftime("%H%M%S_%d%m%y")
    
    # Generate output folder name with model and timestamp if not provided
    if output_folder is None:
        output_folder = f"detection_{yolo_version}_{timestamp}"
    
    print(f"Loading detection model from {detection_model_path}...")
    model = YOLO(detection_model_path)

    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_paths += glob.glob(os.path.join(image_folder, "*.jpeg"))
    image_paths += glob.glob(os.path.join(image_folder, "*.png"))

    if num_images and len(image_paths) > num_images:
        image_paths = image_paths[:num_images]

    print(f"Processing {len(image_paths)} images...")
    os.makedirs(output_folder, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        img = np.array(Image.open(img_path).convert("RGB"))
        results = model(img, conf=conf_threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"Conf: {conf:.2f}", color='black', fontsize=10, fontweight='bold', 
                   bbox=dict(facecolor='lime', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))
        ax.axis('off')
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_folder, f"{base_name}_detected.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Detection visualization saved to {save_path}")
    print(f"All images processed. Results saved to {output_folder}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cassava Leaf Detection Only Inference')
    parser.add_argument('--detection_model', type=str, 
                        default='models/yolov9s_training_results/yolov9s_best.pt',
                        help='Path to YOLO detection model')
    parser.add_argument('--test-folder', type=str, 
                        default='data/Cassava_Leaf_Detector.v1i.yolov8/test/images',
                        help='Folder containing test images')
    parser.add_argument('--output-folder', type=str, default=None, 
                        help='Folder to save detection results (default: auto-generated with timestamp)')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--conf-threshold', type=float, default=0.4, help='Detection confidence threshold')
    args = parser.parse_args()
    detect_leaves_in_folder(args.detection_model, args.test_folder, args.output_folder, args.num_images, args.conf_threshold)
