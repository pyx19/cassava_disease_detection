#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cassava Best Leaf Classification Pipeline
Combines:
1. Leaf detection using YOLO to find the best leaf in an image (highest confidence)
2. Disease classification only on the highest confidence leaf
3. Visualization of results

This is primarily a classification pipeline with YOLO assisting to find the optimal leaf to classify.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from ultralytics import YOLO
from matplotlib.patches import Rectangle
import argparse
import datetime
import re

# Disease classes
DISEASE_CLASSES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]

def extract_model_version(model_path, model_type="yolo"):
    """Extract model version from model path"""
    if not model_path:
        return model_type
        
    filename = os.path.basename(model_path)
    
    if model_type.lower() == "yolo":
        # Look for patterns like yolov5s, yolov8n, yolov9t, etc.
        match = re.search(r'(yolov\d+[a-z]?)', filename.lower())
        if match:
            return match.group(1)
    else:
        # For classifier models: resnet50, efficientnet_b3, densenet121, etc.
        classifiers = ["resnet", "efficientnet", "densenet", "mobilenet", "vit"]
        for clf in classifiers:
            if clf in filename.lower():
                # Extract the full model name (e.g., resnet50, efficientnet_b3)
                match = re.search(f'({clf}[\\w_]*)', filename.lower())
                if match:
                    return match.group(1)
    
    # Default if no match found
    return model_type

class CassavaBestLeafClassifier:
    def __init__(self, detection_model_path, classification_model_path=None):
        """Initialize the pipeline with detection and classification models"""
        print(f"Loading detection model from {detection_model_path}...")
        self.detection_model = YOLO(detection_model_path)
        self.detection_model_path = detection_model_path
        
        # Initialize classification model
        if classification_model_path and os.path.exists(classification_model_path):
            print(f"Loading classification model from {classification_model_path}...")
            self.classification_model = self._load_classification_model(classification_model_path)
            self.has_classifier = True
            self.classification_model_path = classification_model_path
        else:
            print("No classification model provided, using mock predictions...")
            self.classification_model = None
            self.has_classifier = False
            self.classification_model_path = None
        
        # Define image transforms for classification
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_classification_model(self, model_path):
        """Load the disease classification model"""
        model_name = os.path.basename(model_path)
        
        # Create model based on filename
        if 'resnet50' in model_name.lower():
            print(f"Loading ResNet50 model from {model_path}")
            model = resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(DISEASE_CLASSES))
        elif 'densenet121' in model_name.lower():
            print(f"Loading DenseNet121 model from {model_path}")
            from torchvision.models import densenet121
            model = densenet121(pretrained=False)
            model.classifier = torch.nn.Linear(model.classifier.in_features, len(DISEASE_CLASSES))
        elif 'efficientnet' in model_name.lower():
            print(f"Loading EfficientNet model from {model_path}")
            from torchvision.models import efficientnet_b3
            model = efficientnet_b3(pretrained=False)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(DISEASE_CLASSES))
        elif 'mobilenet' in model_name.lower():
            print(f"Loading MobileNet model from {model_path}")
            from torchvision.models import mobilenet_v3_large
            model = mobilenet_v3_large(pretrained=False)
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(DISEASE_CLASSES))
        else:
            # Default to ResNet50
            print(f"Model type not recognized, defaulting to ResNet50")
            model = resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(DISEASE_CLASSES))
        
        # Load trained weights if available
        if os.path.exists(model_path):
            # Load the state dict
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Try to load the state dict directly
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Failed to load state dict directly: {e}")
                print("Attempting to load model using a different approach...")
                
                # If the model was saved as a complete model rather than just state_dict
                if hasattr(state_dict, 'state_dict'):
                    model = state_dict
                else:
                    print(f"Could not load model weights from {model_path}")
        
        model.eval()
        return model
    
    def _classify_leaf(self, leaf_img):
        """Classify a leaf image to predict disease"""
        if self.has_classifier and self.classification_model is not None:
            # Preprocess the image
            img_tensor = self.transform(leaf_img).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.classification_model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                disease_idx = predicted.item()
            
            # Get confidence scores
            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(outputs)[0]
            confidence = probabilities[disease_idx].item()
            
            # Return disease class and confidence
            return DISEASE_CLASSES[disease_idx], confidence
        else:
            # Mock classification - randomly assign disease with fake confidence
            disease_idx = np.random.randint(0, len(DISEASE_CLASSES))
            confidence = np.random.uniform(0.7, 0.95)
            return DISEASE_CLASSES[disease_idx], confidence
    
    def process_image(self, image_path, conf_threshold=0.4):
        """Process a single image through the pipeline to get best leaf only"""
        # Load the image
        img = Image.open(image_path)
        
        # Convert to RGB if image has alpha channel (RGBA)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        img_array = np.array(img)
        
        # Detect leaves
        results = self.detection_model.predict(image_path, conf=conf_threshold)
        detections = results[0]
        
        # Extract each detected leaf and find the one with highest confidence
        best_result = None
        best_conf = -1
        best_leaf_img = None
        
        if len(detections.boxes) > 0:
            boxes = detections.boxes.xyxy.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()
            
            # Find highest confidence detection
            best_idx = np.argmax(confs)
            best_conf = confs[best_idx]
            best_box = boxes[best_idx]
            
            # Extract the best leaf region
            x1, y1, x2, y2 = best_box.astype(int)
            best_leaf_img = img.crop((x1, y1, x2, y2))
            
            # Classify the leaf
            disease, disease_conf = self._classify_leaf(best_leaf_img)
            
            # Store the result for the best leaf
            best_result = {
                'bbox': (x1, y1, x2, y2),
                'detection_conf': best_conf,
                'disease': disease,
                'disease_conf': disease_conf
            }
        
        return img, best_result, best_leaf_img
    
    def visualize_results(self, img, best_result, best_leaf_img, save_path=None):
        """Visualize detection and classification results for best leaf only"""
        if best_result is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(np.array(img))
            ax.set_title("No leaves detected")
            ax.axis('off')
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=200)
                print(f"Visualization saved to {save_path}")
            
            return fig
        
        # Create figure with two subplots: original image and cropped leaf
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), 
                                      gridspec_kw={'width_ratios': [3, 1]})
        
        # Plot original image with bounding box
        ax1.imshow(np.array(img))
        
        # Define colors for different diseases
        disease_colors = {
            "Healthy": 'green',
            "Cassava Mosaic Disease (CMD)": 'red',
            "Cassava Bacterial Blight (CBB)": 'orange',
            "Cassava Brown Streak Disease (CBSD)": 'purple',
            "Cassava Green Mite (CGM)": 'blue'
        }
        
        # Plot the best leaf with its classification
        x1, y1, x2, y2 = best_result['bbox']
        disease = best_result['disease']
        det_conf = best_result['detection_conf']
        dis_conf = best_result['disease_conf']
        
        # Select color based on disease
        color = disease_colors.get(disease, 'red')
        
        # Draw bounding box
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        # Add text label
        label = f"{disease}\nDetect: {det_conf:.2f}, Disease: {dis_conf:.2f}"
        ax1.text(x1, y1-5, label, color='white', fontsize=10,
               bbox=dict(facecolor=color, alpha=0.7))
        
        # Set title for first subplot
        ax1.set_title("Detected Best Leaf")
        ax1.axis('off')
        
        # Plot cropped leaf in second subplot
        ax2.imshow(np.array(best_leaf_img))
        ax2.set_title(f"Predicted: {disease}\nConfidence: {dis_conf:.2f}")
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def process_folder(self, image_folder, output_folder=None, num_images=None):
        """Process all images in a folder"""
        # Generate output folder name if not provided
        if output_folder is None:
            # Extract model versions
            yolo_version = extract_model_version(self.detection_model_path, "yolo")
            classifier_type = "mock"
            if self.has_classifier:
                classifier_type = extract_model_version(self.classification_model_path, "classifier")
            
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%H%M%S_%d%m%y")
            
            # Create output folder name
            output_folder = f"bestleaf_{yolo_version}_{classifier_type}_{timestamp}"
        
        # Find all images
        image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
        image_paths += glob.glob(os.path.join(image_folder, "*.jpeg"))
        image_paths += glob.glob(os.path.join(image_folder, "*.png"))
        
        if num_images and len(image_paths) > num_images:
            image_paths = image_paths[:num_images]
        
        print(f"Processing {len(image_paths)} images...")
        
        # Create output folder for cropped leaves
        crops_folder = os.path.join(output_folder, "crops")
        os.makedirs(crops_folder, exist_ok=True)
        
        # Keep track of classification results
        classification_results = []
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Process image through pipeline
            img, best_result, best_leaf_img = self.process_image(img_path)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Save classification results to list
            if best_result:
                classification_results.append({
                    'image': base_name,
                    'disease': best_result['disease'],
                    'detection_conf': best_result['detection_conf'],
                    'disease_conf': best_result['disease_conf']
                })
                
                # Save the cropped leaf image
                if best_leaf_img:
                    crop_path = os.path.join(crops_folder, f"{base_name}_leaf.png")
                    best_leaf_img.save(crop_path)
            
            # Save visualization
            output_path = os.path.join(output_folder, f"{base_name}_analyzed.png")
            self.visualize_results(img, best_result, best_leaf_img, output_path)
        
        # Save classification summary to CSV
        if classification_results:
            import csv
            csv_path = os.path.join(output_folder, "classification_summary.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['image', 'disease', 'detection_conf', 'disease_conf']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in classification_results:
                    writer.writerow(result)
            print(f"Classification summary saved to {csv_path}")
        
        print(f"All images processed. Results saved to {output_folder}/")
        return classification_results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Cassava Best Leaf Classification Pipeline')
    parser.add_argument('--detection_model', 
                        type=str, 
                        default='models/yolov9s_training_results/yolov9s_best.pt',
                        help='Path to YOLO detection model')
    parser.add_argument('--classification-model', type=str, 
                        default='models/efficientnet_b3_cassava.pth',
                        help='Path to classification model')
    parser.add_argument('--test-folder', type=str, 
                        default='data/Cassava_Leaf_Detector.v1i.yolov8/test/images',
                        help='Folder containing test images')
    parser.add_argument('--output-folder', type=str, 
                        default=None,
                        help='Folder to save analysis results (default: auto-generated with timestamp)')
    parser.add_argument('--num-images', type=int, 
                        default=5,
                        help='Number of images to process')
    args = parser.parse_args()

    # Define model paths
    detection_model_path = args.detection_model
    classification_model_path = args.classification_model if os.path.exists(args.classification_model) else None
    
    # Create the pipeline
    pipeline = CassavaBestLeafClassifier(detection_model_path, classification_model_path)
    
    # Process test images
    test_folder = args.test_folder
    output_folder = args.output_folder
    
    # Process a limited number of test images
    results = pipeline.process_folder(test_folder, output_folder, num_images=args.num_images)
    
    # Print summary
    if results:
        disease_counts = {}
        for result in results:
            disease = result['disease']
            if disease in disease_counts:
                disease_counts[disease] += 1
            else:
                disease_counts[disease] = 1
        
        print("\nClassification Summary:")
        print(f"Total Images: {len(results)}")
        for disease, count in disease_counts.items():
            print(f"{disease}: {count} ({count/len(results)*100:.1f}%)")
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main() 