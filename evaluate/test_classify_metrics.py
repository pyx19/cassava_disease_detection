#!/usr/bin/env python3
"""
Cassava Classification Model Evaluation Script
- Computes accuracy, precision, recall, F1-score per class
- Generates confusion matrix
- Analyzes misclassified examples
- Identifies top-k accuracies
- Displays class activation maps for visualizing model attention
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

# Disease class mapping
DISEASE_CLASSES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]

# Add short names for better visualization
DISEASE_SHORT_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

class CassavaDataset(Dataset):
    """Dataset for Cassava Leaf Disease Classification"""
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

def generate_cam(model, input_tensor, target_layer_name="layer4"):
    """Generate Class Activation Map for the image"""
    # Only for supported models like ResNet
    if not hasattr(model, target_layer_name):
        return None
    
    # Get the feature maps from the target layer
    target_layer = getattr(model, target_layer_name)
    
    # Create hooks
    feature_maps = []
    gradients = []
    
    def save_features(module, input, output):
        feature_maps.append(output.detach().cpu().numpy())
    
    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach().cpu().numpy())
    
    # Register hooks
    handle_features = target_layer.register_forward_hook(save_features)
    handle_gradients = target_layer.register_backward_hook(save_gradients)
    
    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    
    # Get predicted class index
    pred_class = torch.argmax(output).item()
    
    # Backward pass
    output[0, pred_class].backward()
    
    # Remove hooks
    handle_features.remove()
    handle_gradients.remove()
    
    # Generate CAM
    if len(feature_maps) == 0 or len(gradients) == 0:
        return None
    
    # Get feature maps and gradients
    feature = feature_maps[0][0]  # First item in batch
    weight = np.mean(gradients[0][0], axis=(1, 2))  # Global average pooling
    
    # Weighted combination of feature maps
    cam = np.zeros(feature.shape[1:], dtype=np.float32)
    for i, w in enumerate(weight):
        cam += w * feature[i]
    
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Normalize
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    
    # Resize to input image size
    cam = cv2.resize(cam, (224, 224))
    
    return cam, pred_class

def evaluate_classifier(model_path, data_dir, output_dir, device="cpu", 
                       batch_size=32, top_k=3, num_vis_samples=10):
    """
    Evaluate a classifier model on a test dataset.
    
    Args:
        model_path: Path to the saved model file
        data_dir: Directory containing test images
        output_dir: Directory to save evaluation results
        device: Device to run inference on ('cpu' or 'cuda')
        batch_size: Batch size for inference
        top_k: Number of top predictions to consider for top-k accuracy
        num_vis_samples: Number of samples to visualize with CAM
    """
    # Track overall evaluation time
    eval_start_time = datetime.datetime.now()
    
    timestamp = datetime.datetime.now().strftime("%H%M%S_%d%m%y")
    
    # Create output directory
    model_version = extract_model_version(model_path)
    results_dir = os.path.join(output_dir, f"classify_metrics_{model_version}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Log files
    metrics_file = os.path.join(results_dir, "metrics.txt")
    csv_file = os.path.join(results_dir, "per_image_results.csv")
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize CSV writer
    csv_out = open(csv_file, 'w', newline='')
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(['image_name', 'true_label', 'predicted_label', 'confidence', 
                          'correct', 'top3_correct', 'top5_correct', 'top3_predictions'])
    
    # Set device
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, num_classes=len(DISEASE_CLASSES), device=device)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    test_dataset = CassavaDataset(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Loaded test dataset with {len(test_dataset)} images")
    
    # Track inference time separately
    total_inference_time = 0.0
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    all_image_names = []
    all_top_k_preds = []
    misclassified = []
    
    print(f"Evaluating model on {len(test_dataset)} images...")
    
    with torch.no_grad():
        for i, (images, labels, img_names) in enumerate(tqdm(test_loader)):
            # Skip invalid images
            valid_indices = [i for i, label in enumerate(labels) if label >= 0]
            if not valid_indices:
                continue
            
            images = images[valid_indices].to(device)
            labels = labels[valid_indices].to(device)
            img_names = [img_names[i] for i in valid_indices]
            
            # Time the inference
            infer_start = datetime.datetime.now()
            
            # Forward pass
            outputs = model(images)
            
            # Record inference time
            infer_end = datetime.datetime.now()
            batch_inference_time = (infer_end - infer_start).total_seconds()
            total_inference_time += batch_inference_time
            
            # Get predictions
            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(outputs)
            
            # Top-1 predictions
            top1_values, top1_indices = torch.max(probabilities, dim=1)
            
            # Top-k predictions
            topk_values, topk_indices = torch.topk(probabilities, k=min(top_k, len(DISEASE_CLASSES)), dim=1)
            
            # Store predictions and labels
            all_preds.extend(top1_indices.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(top1_values.cpu().numpy())
            all_image_names.extend(img_names)
            
            # Store top-k predictions
            for j in range(len(valid_indices)):
                topk_preds_j = topk_indices[j].cpu().numpy()
                topk_vals_j = topk_values[j].cpu().numpy()
                all_top_k_preds.append({
                    'labels': topk_preds_j.tolist(),
                    'confidence': topk_vals_j.tolist()
                })
                
                # Check if this is a misclassification
                if top1_indices[j].item() != labels[j].item():
                    misclassified.append({
                        'image_name': img_names[j],
                        'true_label': labels[j].item(),
                        'pred_label': top1_indices[j].item(),
                        'confidence': top1_values[j].item(),
                        'top_k_preds': topk_preds_j.tolist(),
                        'top_k_conf': topk_vals_j.tolist()
                    })
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=DISEASE_CLASSES, output_dict=True)
    
    # Print report keys for debugging
    print("Report keys:", report.keys())
    
    # Make sure to use the actual keys from the report
    class_precisions = []
    class_recalls = []
    class_f1_scores = []
    
    for i, class_name in enumerate(DISEASE_CLASSES):
        if class_name in report:
            class_precisions.append(report[class_name]['precision'])
            class_recalls.append(report[class_name]['recall'])
            class_f1_scores.append(report[class_name]['f1-score'])
        else:
            print(f"Warning: Class '{class_name}' not found in report keys")
            class_precisions.append(0.0)
            class_recalls.append(0.0)
            class_f1_scores.append(0.0)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
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
    
    # Calculate total evaluation time
    eval_end_time = datetime.datetime.now()
    total_eval_time = (eval_end_time - eval_start_time).total_seconds()
    
    # Write CSV results
    for i, (img_name, label, pred, conf) in enumerate(zip(all_image_names, all_labels, all_preds, all_confidences)):
        top_k_correct = 1 if label in all_top_k_preds[i]['labels'] else 0
        top_5_correct = 1 if label in all_top_k_preds[i]['labels'][:5] else 0
        top_3_preds = [DISEASE_CLASSES[idx] for idx in all_top_k_preds[i]['labels'][:3]]
        
        csv_writer.writerow([
            img_name, DISEASE_CLASSES[label], DISEASE_CLASSES[pred], 
            f"{conf:.4f}", int(pred == label), top_k_correct, top_5_correct,
            ",".join(top_3_preds)
        ])
    
    # Close CSV file
    csv_out.close()
    
    # Plot confusion matrix (update to use short names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=DISEASE_SHORT_NAMES, 
                yticklabels=DISEASE_SHORT_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    # Plot class-wise accuracies (update to use short names)
    plt.figure(figsize=(12, 6))
    
    # Use the precisions we collected above with short names
    plt.bar(DISEASE_SHORT_NAMES, class_precisions)
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Class-wise Precision')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "class_precision.png"))
    plt.close()
    
    # Visualize some misclassified examples with CAM
    if len(misclassified) > 0 and num_vis_samples > 0:
        # Sort misclassified by confidence (most confident mistakes first)
        misclassified.sort(key=lambda x: x['confidence'], reverse=True)
        samples_to_visualize = misclassified[:min(num_vis_samples, len(misclassified))]
        
        for i, sample in enumerate(samples_to_visualize):
            img_path = os.path.join(data_dir, sample['image_name'])
            
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Generate class activation map
                cam_result = None
                try:
                    cam_result = generate_cam(model, img_tensor)
                except Exception as e:
                    print(f"Error generating CAM for {img_path}: {e}")
                
                # Create visualization
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(np.array(img))
                plt.title(f"Original - True: {DISEASE_CLASSES[sample['true_label']]}")
                plt.axis('off')
                
                # Heatmap overlay if available
                plt.subplot(1, 3, 2)
                plt.imshow(np.array(img))
                if cam_result is not None:
                    cam, _ = cam_result
                    plt.imshow(cam, cmap='jet', alpha=0.5)
                plt.title(f"Predicted: {DISEASE_CLASSES[sample['pred_label']]}")
                plt.axis('off')
                
                # Top-3 predictions
                plt.subplot(1, 3, 3)
                top_k_labels = sample['top_k_preds'][:3]
                top_k_conf = sample['top_k_conf'][:3]
                
                y_pos = np.arange(len(top_k_labels))
                plt.barh(y_pos, top_k_conf)
                plt.yticks(y_pos, [DISEASE_SHORT_NAMES[l] for l in top_k_labels])
                plt.title("Top-3 Predictions")
                plt.xlabel('Confidence')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(vis_dir, f"misclassified_{i+1}.png"))
                plt.close()
            except Exception as e:
                print(f"Error visualizing {img_path}: {e}")
    
    # Write metrics to text file
    with open(metrics_file, 'w') as f:
        f.write(f"Cassava Leaf Disease Classification Model Evaluation\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Test directory: {data_dir}\n\n")
        
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
        f.write(f"Average inference time per batch: {total_inference_time / (i+1):.4f} seconds\n")
        f.write(f"Average inference time per image: {total_inference_time / len(all_labels):.4f} seconds\n")
    
    print(f"Evaluation completed in {total_eval_time:.2f} seconds")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Results saved to {results_dir}")
    return {
        'accuracy': accuracy,
        'top_k_accuracy': top_k_accuracy,
        'results_dir': results_dir,
        'eval_time': total_eval_time,
        'inference_time': total_inference_time
    }

def main():
    parser = argparse.ArgumentParser(description='Cassava Leaf Disease Classification Model Evaluation')
    parser.add_argument('--model', type=str, 
                        default='models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth',
                        help='Path to the saved model file')
    parser.add_argument('--data-dir', type=str, 
                        default='data/preprocessed_leaf_classify/processed_dataset/test',
                        help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, 
                        default='classification_metrics',
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
                        help='Number of samples to visualize with CAM')
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_classifier(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        num_vis_samples=args.num_vis_samples
    )
    
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Top-{args.top_k} Accuracy: {results['top_k_accuracy']:.4f}")
    print(f"Full results saved to {results['results_dir']}")

if __name__ == "__main__":
    main() 