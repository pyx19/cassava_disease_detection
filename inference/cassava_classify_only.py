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
import datetime
import re
import pandas as pd

# Update this list to match your training classes
DISEASE_CLASSES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]

def extract_classifier_version(model_path):
    """Extract classifier model version from model path with simplified naming"""
    filename = os.path.basename(model_path)
    
    # First get the classifier type (resnet, efficientnet, etc.)
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

def load_classification_model(model_path):
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
        raise ValueError("Model type not supported in this example. Update as needed.")
        
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
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

def classify_leaves_in_folder(model_path, image_folder, output_folder=None, num_images=None):
    # Track overall evaluation time
    eval_start_time = datetime.datetime.now()
    
    # Extract model version and create timestamp
    model_version = extract_classifier_version(model_path)
    timestamp = datetime.datetime.now().strftime("%H%M%S_%d%m%y")
    
    # Generate output folder name with model and timestamp if not provided
    if output_folder is None:
        output_folder = f"classification_{model_version}_{timestamp}"
        
    print(f"Loading classification model from {model_path}...")
    model = load_classification_model(model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Determine if we're working with processed_dataset or cropped_dataset
    if "processed_dataset" in image_folder:
        dataset_type = "processed_dataset"
        base_dir = os.path.dirname(image_folder.rstrip('/'))
    elif "cropped_dataset" in image_folder:
        dataset_type = "cropped_dataset"
        base_dir = os.path.dirname(image_folder.rstrip('/'))
    else:
        dataset_type = "unknown"
        base_dir = os.path.dirname(image_folder.rstrip('/'))
    
    # Try to find the CSV file to get true labels
    split_name = os.path.basename(image_folder.rstrip('/'))
    csv_path = os.path.join(base_dir, f"{split_name}.csv")
    
    # Load labels from CSV if available
    labels_dict = {}
    if os.path.exists(csv_path):
        print(f"Found CSV file with labels: {csv_path}")
        df = pd.read_csv(csv_path)
        labels_dict = {row['image_id']: row['label'] for _, row in df.iterrows()}
    
    # Find images directly in the folder (flat structure)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if num_images and len(image_paths) > num_images:
        image_paths = image_paths[:num_images]
    
    print(f"Found {len(image_paths)} images...")
    print(f"Processing {len(image_paths)} images...")
    os.makedirs(output_folder, exist_ok=True)
    
    # Create output files
    results_csv_path = os.path.join(output_folder, "classification_results.csv")
    summary_path = os.path.join(output_folder, "summary.txt")
    
    # Prepare results data
    results_data = []
    
    # Track metrics
    correct_count = 0
    total_count = 0
    class_counts = {i: 0 for i in range(len(DISEASE_CLASSES))}
    class_correct = {i: 0 for i in range(len(DISEASE_CLASSES))}
    confusion_matrix = np.zeros((len(DISEASE_CLASSES), len(DISEASE_CLASSES)), dtype=int)
    
    # Track inference time
    total_inference_time = 0.0
    
    for i, img_path in enumerate(image_paths):
        img_filename = os.path.basename(img_path)
        print(f"Classifying image {i+1}/{len(image_paths)}: {img_filename}")
        
        # Get true label from CSV if available
        true_label = None
        if img_filename in labels_dict:
            true_label = labels_dict[img_filename]
        
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        
        # Measure inference time
        infer_start = datetime.datetime.now()
        
        with torch.no_grad():
            logits = model(input_tensor)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get top prediction and confidence
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()
            pred_class = DISEASE_CLASSES[pred_idx]
            
            # Get top 3 predictions
            if probs.shape[1] >= 3:
                top3_values, top3_indices = torch.topk(probs, k=3, dim=1)
                top3_values = top3_values[0].cpu().numpy()
                top3_indices = top3_indices[0].cpu().numpy()
                top3_classes = [DISEASE_CLASSES[idx] for idx in top3_indices]
                top3_probs = [f"{val:.4f}" for val in top3_values]
            else:
                top3_classes = [pred_class]
                top3_probs = [f"{confidence:.4f}"]
        
        # End timing inference
        infer_end = datetime.datetime.now()
        inference_time = (infer_end - infer_start).total_seconds()
        total_inference_time += inference_time
        
        # Track accuracy and metrics
        is_correct = False
        if true_label is not None:
            total_count += 1
            class_counts[true_label] += 1
            
            if pred_idx == true_label:
                correct_count += 1
                class_correct[true_label] += 1
                is_correct = True
            
            # Update confusion matrix
            confusion_matrix[true_label][pred_idx] += 1
        
        # Store results for CSV
        result_row = {
            'image_name': img_filename,
            'predicted_class': pred_class,
            'confidence': confidence
        }
        
        # Add true label info if available
        if true_label is not None:
            result_row['true_class'] = DISEASE_CLASSES[true_label]
            result_row['correct'] = is_correct
        
        # Add top-3 predictions
        for i, (cls, prob) in enumerate(zip(top3_classes, top3_probs)):
            result_row[f'top{i+1}_class'] = cls
            result_row[f'top{i+1}_prob'] = prob
        
        results_data.append(result_row)
    
    # Calculate total evaluation time
    eval_end_time = datetime.datetime.now()
    total_eval_time = (eval_end_time - eval_start_time).total_seconds()
    
    # Write results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
    
    # Write summary metrics to text file
    with open(summary_path, 'w') as f:
        f.write(f"Cassava Leaf Disease Classification Summary\n")
        f.write(f"=========================================\n\n")
        
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Test data: {image_folder}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        if total_count > 0:
            # Overall metrics
            accuracy = correct_count / total_count
            f.write(f"Overall Metrics\n")
            f.write(f"--------------\n")
            f.write(f"Total images: {total_count}\n")
            f.write(f"Correctly classified: {correct_count}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            
            # Per-class metrics
            f.write(f"Class-wise Metrics\n")
            f.write(f"----------------\n")
            for class_idx in range(len(DISEASE_CLASSES)):
                if class_counts[class_idx] > 0:
                    class_acc = class_correct[class_idx] / class_counts[class_idx]
                    f.write(f"{DISEASE_CLASSES[class_idx]}:\n")
                    f.write(f"  Total: {class_counts[class_idx]}\n")
                    f.write(f"  Correct: {class_correct[class_idx]}\n")
                    f.write(f"  Accuracy: {class_acc:.4f}\n\n")
            
            # Confusion matrix
            f.write(f"Confusion Matrix (row=true, col=predicted):\n")
            header = "True\\Pred |" + "|".join(f" {i:^6}" for i in range(len(DISEASE_CLASSES)))
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            for i in range(len(DISEASE_CLASSES)):
                row = f"{i:^9} |"
                for j in range(len(DISEASE_CLASSES)):
                    row += f" {confusion_matrix[i][j]:^6}"
                f.write(row + "\n")
        
        # Add timing information
        f.write(f"\nTiming Information\n")
        f.write(f"-----------------\n")
        f.write(f"Total evaluation time: {total_eval_time:.2f} seconds\n")
        f.write(f"Total inference time: {total_inference_time:.2f} seconds\n")
        if len(image_paths) > 0:
            f.write(f"Average inference time per image: {total_inference_time/len(image_paths):.4f} seconds\n")
    
    print(f"Summary saved to {summary_path}")
    
    # Print accuracy if we have true labels
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"\nAccuracy on {total_count} images: {accuracy:.4f} ({correct_count}/{total_count})")
    
    print(f"Evaluation completed in {total_eval_time:.2f} seconds")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"All images processed. Results saved to {output_folder}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cassava Leaf Disease Classification Only Inference')
    parser.add_argument('--classification-model', type=str, 
                        default='models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth',
                        help='Path to classification model')
    parser.add_argument('--test-folder', type=str, 
                        default='data/preprocessed_leaf_classify/processed_dataset/test',
                        help='Folder containing test images')
    parser.add_argument('--output-folder', type=str, default=None, 
                        help='Folder to save classification results (default: auto-generated with timestamp)')
    parser.add_argument('--num-images', type=int, default=20, help='Number of images to process')
    args = parser.parse_args()
    classify_leaves_in_folder(args.classification_model, args.test_folder, args.output_folder, args.num_images)
