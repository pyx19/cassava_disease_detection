#!/usr/bin/env python3
"""
Cassava YOLO + Classification Model Evaluation Script
- Uses YOLO to detect and extract best leaf from each image
- Applies classification model to detected leaf
- Computes accuracy, precision, recall, F1-score per class
- Generates confusion matrix and visualizations of results
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm
import argparse
import datetime
import re
import cv2
from typing import List, Dict, Tuple, Optional
import json
import csv
from ultralytics import YOLO

# Disease class mapping
DISEASE_CLASSES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]

# Short names for better visualization
DISEASE_SHORT_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

class CassavaDataset(Dataset):
    """Dataset for Cassava Leaf Disease Classification with YOLO preprocessing"""
    def __init__(self, data_dir, csv_file=None, transform=None):
        """
        Args:
            data_dir: Directory with all the images
            csv_file: Path to the csv file with annotations (optional)
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Try to find CSV file if not provided
        if csv_file is None:
            # Check if we're working with the test split from a processed dataset
            if data_dir.endswith('/test') or data_dir.endswith('\\test'):
                base_dir = os.path.dirname(data_dir)
                csv_path = os.path.join(base_dir, "test.csv")
                if os.path.exists(csv_path):
                    csv_file = csv_path
                    print(f"Found CSV file: {csv_file}")
        
        # Check if this is a directory-based dataset or csv-based
        if csv_file and os.path.exists(csv_file):
            # CSV-based dataset (flat structure)
            print(f"Loading labels from CSV file: {csv_file}")
            self.df = pd.read_csv(csv_file)
            self.samples = [(row['image_id'], row['label']) for _, row in self.df.iterrows()]
        else:
            # Fallback to directory-based dataset (class subdirectories)
            print("No CSV file found, looking for class subdirectories...")
            
            # Check if using class subdirectories
            has_class_dirs = False
            for class_idx in range(len(DISEASE_CLASSES)):
                class_dir = os.path.join(data_dir, str(class_idx))
                if os.path.isdir(class_dir):
                    has_class_dirs = True
                    break
            
            if has_class_dirs:
                # Handle directory structure (class folders)
                for class_idx in range(len(DISEASE_CLASSES)):
                    class_dir = os.path.join(data_dir, str(class_idx))
                    if not os.path.isdir(class_dir):
                        continue
                    
                    # Get all image files in this class directory
                    img_paths = glob.glob(os.path.join(class_dir, '*.jpg'))
                    img_paths += glob.glob(os.path.join(class_dir, '*.jpeg'))
                    img_paths += glob.glob(os.path.join(class_dir, '*.png'))
                    
                    # Add each image with its class
                    for img_path in img_paths:
                        rel_path = os.path.relpath(img_path, data_dir)
                        self.samples.append((rel_path, class_idx))
            else:
                # Flat structure with no CSV file - we can't get labels
                print("Warning: Flat directory structure with no CSV file found.")
                print("Images will be loaded but labels will be missing.")
                img_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
                img_paths += glob.glob(os.path.join(data_dir, '*.jpeg'))
                img_paths += glob.glob(os.path.join(data_dir, '*.png'))
                
                # Set dummy labels to -1
                for img_path in img_paths:
                    rel_path = os.path.basename(img_path)
                    self.samples.append((rel_path, -1))  # -1 indicates unknown label
        
        print(f"Loaded dataset with {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        
        # For both directory and flat structures
        img_path = os.path.join(self.data_dir, img_name)
        if not os.path.exists(img_path):
            # For flat structure, try using just the filename
            img_path = os.path.join(self.data_dir, os.path.basename(img_name))
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_name
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample
            if self.transform:
                dummy = torch.zeros((3, 224, 224))
            else:
                dummy = Image.new('RGB', (224, 224))
            return dummy, -1, img_name

def extract_best_leaf(image, yolo_model, conf_threshold=0.4):
    """Extract the highest confidence leaf from an image using YOLO
    
    Args:
        image: PIL Image or path to image
        yolo_model: Loaded YOLO model
        conf_threshold: Confidence threshold for detection
        
    Returns:
        PIL Image of the cropped leaf, or original image if no leaf detected
    """
    # If image is a file path, load it
    if isinstance(image, str):
        try:
            pil_image = Image.open(image).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image}: {e}")
            return None
    else:
        pil_image = image
    
    # Convert PIL to numpy for YOLO
    img_array = np.array(pil_image)
    
    # Run YOLO detection
    try:
        results = yolo_model.predict(img_array, conf=conf_threshold)
        detections = results[0]
        
        # Find highest confidence detection
        if len(detections.boxes) > 0:
            boxes = detections.boxes.xyxy.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()
            
            # Get highest confidence box
            best_idx = np.argmax(confs)
            best_box = boxes[best_idx]
            
            # Crop image to that box
            x1, y1, x2, y2 = best_box.astype(int)
            cropped_img = pil_image.crop((x1, y1, x2, y2))
            
            return cropped_img
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
    
    # Return original image if no detection or error occurred
    return pil_image

def extract_model_version(model_path):
    """Extract model version from model path with simplified naming"""
    if not model_path:
        return "classifier"
        
    filename = os.path.basename(model_path)
    
    # For classifier models: resnet50, efficientnet_b3, densenet121, etc.
    classifier_type = None
    classifiers = ["resnet", "efficientnet", "densenet", "mobilenet", "vit"]
    for clf in classifiers:
        if clf in filename.lower():
            # Get the model type with optional number (e.g., resnet50, efficientnet_b3)
            match = re.search(f'({clf}[\\d_]*b?\\d*)', filename.lower())
            if match:
                classifier_type = match.group(1)
                break
    
    if not classifier_type:
        return "classifier"  # default if no match found
        
    return classifier_type  # Return simplified name

def load_model(model_path, num_classes=5, device="cpu"):
    """Load a pretrained classification model"""
    model_name = os.path.basename(model_path).lower()
    model = None
    
    # Create model based on filename
    if 'resnet50' in model_name:
        print(f"Loading ResNet50 model from {model_path}")
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'densenet121' in model_name:
        print(f"Loading DenseNet121 model from {model_path}")
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'efficientnet' in model_name:
        print(f"Loading EfficientNet model from {model_path}")
        model = models.efficientnet_b3(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'mobilenet' in model_name:
        print(f"Loading MobileNet model from {model_path}")
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif 'vit' in model_name:
        print(f"Loading ViT model from {model_path}")
        model = models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Load model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=torch.device(device))
        
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
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    model.eval()
    return model.to(device)

def evaluate_yolo_classifier(yolo_model_path, classification_model_path, 
                             data_dir, output_dir, device="cpu", 
                             batch_size=32, top_k=3, num_vis_samples=10,
                             yolo_conf=0.4, max_images=None):
    """
    Evaluate a YOLO + classifier pipeline on a test dataset.
    
    Args:
        yolo_model_path: Path to the saved YOLO model file
        classification_model_path: Path to the saved classification model file
        data_dir: Directory containing test images
        output_dir: Directory to save evaluation results
        device: Device to run inference on ('cpu' or 'cuda')
        batch_size: Batch size for inference
        top_k: Number of top predictions to consider for top-k accuracy
        num_vis_samples: Number of samples to visualize with comparison
        yolo_conf: Confidence threshold for YOLO detection
        max_images: Maximum number of images to process (None=process all)
    """
    # Track overall evaluation time
    eval_start_time = datetime.datetime.now()
    
    timestamp = datetime.datetime.now().strftime("%H%M%S_%d%m%y")
    
    # Create output directory
    model_version = extract_model_version(classification_model_path)
    results_dir = os.path.join(output_dir, f"yolo_classify_metrics_{model_version}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Log files
    metrics_file = os.path.join(results_dir, "metrics.txt")
    csv_file = os.path.join(results_dir, "per_image_results.csv")
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading YOLO model from {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    
    print(f"Loading classification model from {classification_model_path}")
    class_model = load_model(classification_model_path, num_classes=len(DISEASE_CLASSES), device=device)
    
    # Classification transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset (without transform, as we'll apply YOLO first)
    test_dataset = CassavaDataset(data_dir)
    
    # Limit the number of images if specified
    if max_images is not None and max_images > 0:
        num_images = min(len(test_dataset), max_images)
        print(f"Processing {num_images} images (limited by --max-images)")
    else:
        num_images = len(test_dataset)
        print(f"Processing all {num_images} images")
    
    # Initialize CSV writer
    csv_out = open(csv_file, 'w', newline='')
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(['image_name', 'true_label', 'predicted_label', 'confidence', 
                         'correct', 'top3_correct', 'top5_correct', 'top3_predictions',
                         'leaf_detected'])
    
    # Evaluation
    class_model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    all_image_names = []
    all_top_k_preds = []
    misclassified = []
    successful_detections = 0
    
    # Timing metrics
    total_yolo_time = 0.0
    total_classification_time = 0.0
    total_inference_time = 0.0  # Combined YOLO + classification
    
    print(f"Evaluating YOLO+classifier on {num_images} images...")
    
    # Process images one by one
    for i in tqdm(range(num_images)):
        # Get image and label
        _, label, img_name = test_dataset[i]
        
        # Skip invalid images/labels
        if label < 0:
            continue
        
        # Prepare image path
        img_path = os.path.join(data_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, os.path.basename(img_name))
        
        # Extract best leaf using YOLO
        leaf_detected = False
        try:
            # Start timing YOLO inference
            yolo_start = datetime.datetime.now()
            
            cropped_leaf = extract_best_leaf(img_path, yolo_model, conf_threshold=yolo_conf)
            
            # End timing YOLO inference
            yolo_end = datetime.datetime.now()
            yolo_time = (yolo_end - yolo_start).total_seconds()
            total_yolo_time += yolo_time
            
            # Check if leaf was detected (will return original image if not)
            if cropped_leaf is not None:
                leaf_detected = True
                successful_detections += 1
                
                # Skip visualization of detected leaves to save time
                
                # Apply classification transform
                input_tensor = transform(cropped_leaf).unsqueeze(0).to(device)
                
                # Run classification
                with torch.no_grad():
                    # Start timing classification inference
                    classify_start = datetime.datetime.now()
                    
                    outputs = class_model(input_tensor)
                    
                    # End timing classification inference
                    classify_end = datetime.datetime.now()
                    classify_time = (classify_end - classify_start).total_seconds()
                    total_classification_time += classify_time
                    
                    # Calculate combined inference time
                    total_inference_time += (yolo_time + classify_time)
                    
                    # Get predictions
                    softmax = torch.nn.Softmax(dim=1)
                    probabilities = softmax(outputs)
                    
                    # Top-1 predictions
                    top1_values, top1_indices = torch.max(probabilities, dim=1)
                    
                    # Top-k predictions
                    topk_values, topk_indices = torch.topk(probabilities, k=min(top_k, len(DISEASE_CLASSES)), dim=1)
                    
                    # Get predictions as numpy arrays
                    pred_idx = top1_indices.item()
                    confidence = top1_values.item()
                    topk_preds = topk_indices[0].cpu().numpy()
                    topk_vals = topk_values[0].cpu().numpy()
                    
                    # Store prediction results
                    all_preds.append(pred_idx)
                    all_labels.append(label)
                    all_confidences.append(confidence)
                    all_image_names.append(img_name)
                    
                    # Store top-k predictions
                    all_top_k_preds.append({
                        'labels': topk_preds.tolist(),
                        'confidence': topk_vals.tolist()
                    })
                    
                    # Check if misclassified
                    if pred_idx != label:
                        misclassified.append({
                            'image_name': img_name,
                            'true_label': label,
                            'pred_label': pred_idx,
                            'confidence': confidence,
                            'top_k_preds': topk_preds.tolist(),
                            'top_k_conf': topk_vals.tolist(),
                            'cropped_leaf': cropped_leaf
                        })
                    
                    # Write to CSV
                    top_k_correct = 1 if label in topk_preds else 0
                    top_5_correct = 1 if label in topk_preds[:5] else 0
                    top_3_preds = [DISEASE_CLASSES[idx] for idx in topk_preds[:3]]
                    
                    csv_writer.writerow([
                        img_name, DISEASE_CLASSES[label], DISEASE_CLASSES[pred_idx], 
                        f"{confidence:.4f}", int(pred_idx == label), top_k_correct, top_5_correct,
                        ",".join(top_3_preds), leaf_detected
                    ])
            else:
                print(f"Warning: No leaf detected in {img_name}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    csv_out.close()
    
    print(f"Leaf detection success rate: {successful_detections}/{num_images} ({successful_detections/num_images*100:.2f}%)")
    
    # Calculate total evaluation time
    eval_end_time = datetime.datetime.now()
    total_eval_time = (eval_end_time - eval_start_time).total_seconds()
    
    # Calculate metrics (if we have predictions)
    if len(all_preds) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Find unique labels in the dataset
        unique_labels = sorted(set(all_labels + all_preds))
        
        # Create subset of disease classes that are present in this evaluation
        present_classes = [DISEASE_CLASSES[i] for i in unique_labels]
        present_short_names = [DISEASE_SHORT_NAMES[i] for i in unique_labels]
        
        # Generate classification report with the classes that are present
        report = classification_report(all_labels, all_preds, 
                                      labels=unique_labels, 
                                      target_names=present_classes, 
                                      output_dict=True)
        
        # Print report keys for debugging
        print("Report keys:", report.keys())
        print(f"Found {len(unique_labels)} classes in evaluation set: {present_classes}")
        
        # Extract metrics from the report
        class_precisions = [0.0] * len(DISEASE_CLASSES)
        class_recalls = [0.0] * len(DISEASE_CLASSES)
        class_f1_scores = [0.0] * len(DISEASE_CLASSES)
        
        for i, class_name in enumerate(present_classes):
            if class_name in report:
                label_idx = DISEASE_CLASSES.index(class_name)
                class_precisions[label_idx] = report[class_name]['precision']
                class_recalls[label_idx] = report[class_name]['recall']
                class_f1_scores[label_idx] = report[class_name]['f1-score']
            else:
                print(f"Warning: Class '{class_name}' not found in report keys")
        
        # Generate confusion matrix for present classes only
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=unique_labels)
        
        # Calculate top-k accuracy
        top_k_correct = 0
        top_k_correct_per_class = [0] * len(DISEASE_CLASSES)
        samples_per_class = [0] * len(DISEASE_CLASSES)
        
        for i, label in enumerate(all_labels):
            samples_per_class[label] += 1
            if label in all_top_k_preds[i]['labels']:
                top_k_correct += 1
                top_k_correct_per_class[label] += 1
        
        top_k_accuracy = top_k_correct / len(all_labels) if len(all_labels) > 0 else 0
        top_k_accuracy_per_class = [
            top_k_correct_per_class[i] / samples_per_class[i] if samples_per_class[i] > 0 else 0
            for i in range(len(DISEASE_CLASSES))
        ]
        
        # Generate plots only after all evaluation is done
        # Skip most visualizations to save time and focus on metrics
        
        # Plot confusion matrix - basic visualization that's useful
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=present_short_names, 
                    yticklabels=present_short_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
        plt.close()
        
        # Plot class-wise accuracies - basic visualization that's useful
        plt.figure(figsize=(12, 6))
        # Only plot present classes
        present_precisions = [class_precisions[DISEASE_CLASSES.index(cls)] for cls in present_classes]
        plt.bar(present_short_names, present_precisions)
        plt.xlabel('Class')
        plt.ylabel('Precision')
        plt.title('Class-wise Precision')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "class_precision.png"))
        plt.close()
        
        # Skip visualization of misclassified samples to save time
        
        # Write metrics to text file
        with open(metrics_file, 'w') as f:
            f.write(f"Cassava Leaf Disease YOLO+Classification Model Evaluation\n")
            f.write(f"===================================================\n\n")
            f.write(f"YOLO Model: {os.path.basename(yolo_model_path)}\n")
            f.write(f"Classification Model: {os.path.basename(classification_model_path)}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Test directory: {data_dir}\n\n")
            
            f.write(f"Leaf Detection Stats:\n")
            f.write(f"Total images: {num_images}\n")
            f.write(f"Successful detections: {successful_detections}\n")
            f.write(f"Detection rate: {successful_detections/num_images*100:.2f}%\n\n")
            
            f.write(f"Overall Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Top-{top_k} Accuracy: {top_k_accuracy:.4f}\n\n")
            
            f.write(f"Class-wise Metrics:\n")
            for i, class_name in enumerate(DISEASE_CLASSES):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_precisions[i]:.4f}\n")
                f.write(f"  Recall: {class_recalls[i]:.4f}\n")
                f.write(f"  F1-score: {class_f1_scores[i]:.4f}\n")
                f.write(f"  Top-{top_k} Accuracy: {top_k_accuracy_per_class[i]:.4f}\n\n")
            
            f.write(f"Total samples: {len(all_labels)}\n")
            f.write(f"Correct predictions: {sum(1 for i, j in zip(all_preds, all_labels) if i == j)}\n")
            f.write(f"Incorrect predictions: {sum(1 for i, j in zip(all_preds, all_labels) if i != j)}\n\n")
            
            f.write(f"Timing Information:\n")
            f.write(f"Total evaluation time: {total_eval_time:.2f} seconds\n")
            f.write(f"Total inference time: {total_inference_time:.2f} seconds\n")
            f.write(f"  - YOLO detection time: {total_yolo_time:.2f} seconds\n")
            f.write(f"  - Classification time: {total_classification_time:.2f} seconds\n")
            f.write(f"Average inference time per image: {total_inference_time/successful_detections:.4f} seconds\n")
            f.write(f"  - Average YOLO time: {total_yolo_time/num_images:.4f} seconds\n")
            f.write(f"  - Average classification time: {total_classification_time/successful_detections:.4f} seconds\n")
        
        print(f"Evaluation completed in {total_eval_time:.2f} seconds")
        print(f"Total inference time: {total_inference_time:.2f} seconds")
        print(f"Results saved to {results_dir}")
        return {
            'accuracy': accuracy,
            'top_k_accuracy': top_k_accuracy,
            'detection_rate': successful_detections/num_images,
            'results_dir': results_dir,
            'eval_time': total_eval_time,
            'inference_time': total_inference_time,
            'yolo_time': total_yolo_time,
            'classification_time': total_classification_time
        }
    else:
        print("No predictions were made. Check the logs for errors.")
        return {
            'accuracy': 0,
            'top_k_accuracy': 0,
            'detection_rate': successful_detections/num_images if num_images > 0 else 0,
            'results_dir': results_dir,
            'eval_time': total_eval_time,
            'inference_time': total_inference_time,
            'yolo_time': total_yolo_time,
            'classification_time': total_classification_time
        }

def main():
    parser = argparse.ArgumentParser(description='Cassava YOLO+Classification Model Evaluation')
    parser.add_argument('--yolo-model', type=str, 
                        default='models/yolov9s_training_results/yolov9s_best.pt',
                        help='Path to the saved YOLO model file')
    parser.add_argument('--classification-model', type=str, 
                        default='models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth',
                        help='Path to the saved classification model file')
    parser.add_argument('--data-dir', type=str, 
                        default='data/preprocessed_leaf_classify/processed_dataset/test',
                        help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, 
                        default='yolo_classification_metrics',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, 
                        default=32,
                        help='Batch size for inference')
    parser.add_argument('--top-k', type=int, 
                        default=3,
                        help='Number of top predictions to consider for top-k accuracy')
    parser.add_argument('--num-vis-samples', type=int, 
                        default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--yolo-conf', type=float,
                        default=0.4,
                        help='Confidence threshold for YOLO detection')
    parser.add_argument('--max-images', type=int,
                        default=None,
                        help='Maximum number of images to process (None=process all)')
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_yolo_classifier(
        yolo_model_path=args.yolo_model,
        classification_model_path=args.classification_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        num_vis_samples=args.num_vis_samples,
        yolo_conf=args.yolo_conf,
        max_images=args.max_images
    )
    
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Top-{args.top_k} Accuracy: {results['top_k_accuracy']:.4f}")
    print(f"Leaf Detection Rate: {results['detection_rate']:.4f}")
    print(f"Full results saved to {results['results_dir']}")

if __name__ == "__main__":
    main() 