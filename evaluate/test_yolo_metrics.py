#!/usr/bin/env python3
"""
YOLO Detection Model Evaluation Script
- Computes mAP@0.5, mean IoU, average confidence, NA cases, false negatives
- Generates GradCAM saliency maps (heat maps) in a new folder
"""
import os
import glob
from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2
from ultralytics import YOLO
import argparse
from collections import defaultdict
import csv
import datetime
import re

def iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def parse_yolo_label(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # YOLO: class cx cy w h (normalized)
            cls, cx, cy, w, h = map(float, parts[:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes

def yolo_to_xyxy(box, img_w, img_h):
    # YOLO: class, cx, cy, w, h (normalized)
    _, cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return [x1, y1, x2, y2]

def extract_model_version(model_path):
    """Extract YOLO model version from model path"""
    if not model_path:
        return "yolo"
        
    filename = os.path.basename(model_path)
    
    # Look for patterns like yolov5s, yolov8n, yolov9t, etc.
    match = re.search(r'(yolov\d+[a-z]?)', filename.lower())
    if match:
        return match.group(1)
    
    # Default if no match found
    return "yolo"

def evaluate_yolo(detection_model_path, image_folder, label_folder, output_metrics, saliency_dir, conf_threshold=0.4, iou_thresh=0.5, use_gradcam=False, csv_output=None):
    """
    Evaluates a YOLO model on a test dataset, computes metrics, and generates saliency maps.
    
    Args:
        detection_model_path: Path to YOLO model weights
        image_folder: Folder containing test images
        label_folder: Folder containing ground truth labels in YOLO format
        output_metrics: File to write evaluation metrics
        saliency_dir: Directory to save saliency map visualizations
        conf_threshold: Detection confidence threshold
        iou_thresh: IoU threshold for TP/FP determination
        use_gradcam: Whether to use confidence-based heatmaps for visualizations
        csv_output: Optional path to CSV file for detailed per-image metrics
    """
    os.makedirs(saliency_dir, exist_ok=True)
    model = YOLO(detection_model_path)
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
    image_paths += glob.glob(os.path.join(image_folder, '*.jpeg'))
    image_paths += glob.glob(os.path.join(image_folder, '*.png'))

    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    all_confidences = []
    all_ious = []
    na_cases = 0
    total_gt = 0
    total_pred = 0
    
    # Initialize CSV writer if output file is provided
    csv_file = None
    csv_writer = None
    if csv_output:
        csv_file = open(csv_output, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image_name', 'true_positives', 'false_positives', 'false_negatives', 
                           'gt_boxes', 'pred_boxes', 'avg_confidence', 'avg_iou', 'confidences', 'ious'])

    with open(output_metrics, 'w') as fout:
        fout.write(f"YOLO Detection Model Evaluation Results\n")
        fout.write(f"Model: {os.path.basename(detection_model_path)}\n")
        fout.write(f"Confidence threshold: {conf_threshold}\n")
        fout.write(f"IoU threshold: {iou_thresh}\n")
        fout.write(f"Test images folder: {image_folder}\n")
        fout.write(f"Labels folder: {label_folder}\n\n")
        fout.write(f"Per-Image Results:\n")
        fout.write(f"{'Image':<60} {'TP':<4} {'FP':<4} {'FN':<4} {'GT':<4} {'Pred':<4} {'Avg Conf':<10} {'Avg IoU':<10}\n")
        fout.write(f"{'-'*120}\n")
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(label_folder, label_name)
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size
            gt_boxes_yolo = parse_yolo_label(label_path)
            gt_boxes = [yolo_to_xyxy(b, img_w, img_h) for b in gt_boxes_yolo]
            total_gt += len(gt_boxes)

            # Predict
            results = model(np.array(img), conf=conf_threshold)
            pred_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            pred_confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
            total_pred += len(pred_boxes)
            all_confidences.extend(pred_confs)

            # Match predictions to ground truth
            matched_gt = set()
            matched_pred = set()
            ious_img = []
            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0
                best_gt = -1
                for j, gt_box in enumerate(gt_boxes):
                    iou_val = iou(pred_box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt = j
                if best_iou >= iou_thresh and best_gt not in matched_gt:
                    all_true_positives += 1
                    matched_gt.add(best_gt)
                    matched_pred.add(i)
                    ious_img.append(best_iou)
                else:
                    all_false_positives += 1
            # False negatives: ground truths not matched
            false_neg = len(gt_boxes) - len(matched_gt)
            all_false_negatives += false_neg
            all_ious.extend(ious_img)
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                na_cases += 1
                
            # Calculate per-image metrics
            avg_confidence = np.mean(pred_confs) if len(pred_confs) > 0 else 0
            avg_iou = np.mean(ious_img) if len(ious_img) > 0 else 0
            
            # Write to CSV
            if csv_writer:
                confidences_str = ','.join([f"{conf:.4f}" for conf in pred_confs])
                ious_str = ','.join([f"{val:.4f}" for val in ious_img])
                csv_writer.writerow([img_name, len(matched_gt), len(pred_boxes)-len(matched_gt), 
                                    false_neg, len(gt_boxes), len(pred_boxes), 
                                    f"{avg_confidence:.4f}", f"{avg_iou:.4f}", 
                                    confidences_str, ious_str])
            
            # Write to text file with confidence and IoU 
            fout.write(f"{img_name:<60} {len(matched_gt):<4} {len(pred_boxes)-len(matched_gt):<4} {false_neg:<4} {len(gt_boxes):<4} {len(pred_boxes):<4} {avg_confidence:.4f}     {avg_iou:.4f}\n")

            # Generate heatmap-based saliency maps
            original_img = np.array(img)
            saliency_img = original_img.copy()
            
            if use_gradcam and len(pred_boxes) > 0 and len(pred_confs) > 0:
                # Create confidence-based heatmaps
                for i, (box, conf) in enumerate(zip(pred_boxes, pred_confs)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Create a heatmap based on confidence
                    # Higher confidence = more red
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    # Create radial gradient heatmap
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    heatmap = np.zeros((box_height, box_width, 3), dtype=np.float32)
                    
                    # Generate a gradient from center (hot) to edges (cool)
                    for x in range(box_width):
                        for y in range(box_height):
                            # Calculate distance from center (normalized)
                            dist_x = (x - box_width/2) / (box_width/2)
                            dist_y = (y - box_height/2) / (box_height/2)
                            dist = np.sqrt(dist_x**2 + dist_y**2)
                            dist = min(1.0, dist)  # Cap at 1.0
                            
                            # Invert distance and scale by confidence
                            intensity = (1.0 - dist) * conf
                            
                            # JET colormap: blue (low) -> green -> yellow -> red (high)
                            if intensity < 0.25:
                                # Blue to cyan
                                r, g, b = 0, intensity * 4, 1.0
                            elif intensity < 0.5:
                                # Cyan to green
                                r, g, b = 0, 1.0, 1.0 - (intensity - 0.25) * 4
                            elif intensity < 0.75:
                                # Green to yellow
                                r, g, b = (intensity - 0.5) * 4, 1.0, 0
                            else:
                                # Yellow to red
                                r, g, b = 1.0, 1.0 - (intensity - 0.75) * 4, 0
                            
                            heatmap[y, x] = [r * 255, g * 255, b * 255]
                    
                    # Create box mask to only show heatmap in detection area
                    mask = np.zeros_like(original_img)
                    # Ensure we don't go out of bounds
                    y_end = min(y2, mask.shape[0])
                    x_end = min(x2, mask.shape[1])
                    patch_height = y_end - y1
                    patch_width = x_end - x1
                    
                    if patch_height > 0 and patch_width > 0:
                        resized_heatmap = cv2.resize(heatmap, (patch_width, patch_height))
                        mask[y1:y_end, x1:x_end] = resized_heatmap
                        
                        # Apply the heatmap with transparency
                        alpha = 0.7  # Transparency level
                        saliency_img = cv2.addWeighted(saliency_img, 1, mask, alpha, 0)
                        
                        # Add a border around the box 
                        color = (0, 255, 0)  # Green border
                        thickness = 2
                        cv2.rectangle(saliency_img, (x1, y1), (x2, y2), color, thickness)
                        
                        # Add confidence text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(saliency_img, f"{conf:.2f}", (x1, y1-5), font, 0.5, color, 2)
                
                # Save the saliency map
                saliency_pil = Image.fromarray(saliency_img)
                sal_path = os.path.join(saliency_dir, img_name)
                saliency_pil.save(sal_path)
            else:
                # Simple bounding box visualization
                draw = ImageDraw.Draw(img)
                for box in pred_boxes:
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
                sal_path = os.path.join(saliency_dir, img_name)
                img.save(sal_path)

            fout.write(f"{img_name}: TP={len(matched_gt)}, FP={len(pred_boxes)-len(matched_gt)}, FN={false_neg}, GT={len(gt_boxes)}, Pred={len(pred_boxes)}\n")

        # Metrics
        precision = all_true_positives / (all_true_positives + all_false_positives + 1e-6)
        recall = all_true_positives / (all_true_positives + all_false_negatives + 1e-6)
        mean_iou = np.mean(all_ious) if all_ious else 0
        mean_conf = np.mean(all_confidences) if all_confidences else 0
        
        fout.write(f"\n{'-'*120}\n")
        fout.write(f"Summary Metrics:\n")
        fout.write(f"Total images: {len(image_paths)}\n")
        fout.write(f"Total GT boxes: {total_gt}\n")
        fout.write(f"Total Pred boxes: {total_pred}\n")
        fout.write(f"True Positives: {all_true_positives}\n")
        fout.write(f"False Positives: {all_false_positives}\n")
        fout.write(f"False Negatives: {all_false_negatives}\n")
        fout.write(f"NA Cases (no pred or gt): {na_cases}\n")
        fout.write(f"Precision: {precision:.4f}\n")
        fout.write(f"Recall: {recall:.4f}\n")
        fout.write(f"Mean IoU: {mean_iou:.4f}\n")
        fout.write(f"Mean Confidence: {mean_conf:.4f}\n")
        # mAP@0.5 (approximate, as full AP needs more code)
        fout.write(f"mAP@0.5 (proxy): {precision*recall:.4f}\n")
    
    # Close CSV file if open
    if csv_file:
        csv_file.close()
        print(f"Per-image metrics written to {csv_output}")
    
    print(f"Evaluation complete. Metrics written to {output_metrics}\nSaliency maps saved in {saliency_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO Detection Model Evaluation with Metrics and Saliency Maps')
    parser.add_argument('--detection_model', type=str, 
                        default='models/yolov9s_training_results/yolov9s_best.pt',
                        help='Path to YOLO detection model')
    parser.add_argument('--test-folder', type=str, 
                        default='data/Cassava_Leaf_Detector.v1i.yolov8/test/images', 
                        help='Folder containing test images')
    parser.add_argument('--label-folder', type=str, 
                        default='data/Cassava_Leaf_Detector.v1i.yolov8/test/labels', 
                        help='Folder containing YOLO label txt files')
    parser.add_argument('--output-metrics', type=str, default='yolo_metrics_summary.txt', help='Output txt file for metrics summary')
    parser.add_argument('--csv-output', type=str, default='yolo_metrics_per_image.csv', help='Output CSV file for per-image metrics')
    parser.add_argument('--saliency-dir', type=str, default='detection_saliency_maps', help='Folder to save saliency maps')
    parser.add_argument('--conf-threshold', type=float, default=0.4, help='Detection confidence threshold')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='IoU threshold for TP/FP')
    parser.add_argument('--use-gradcam', action='store_true', default=True, help='Use confidence-based heatmaps for saliency maps')
    args = parser.parse_args()
    
    # Extract model version from path
    yolo_version = extract_model_version(args.detection_model)
    
    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%H%M%S_%d%m%y")
    
    # Update output filenames/dirs with model version and timestamp
    model_timestamp = f"{yolo_version}_{timestamp}"
    
    # Create main saliency directory
    saliency_type = "gradcam" if args.use_gradcam else "simple"
    saliency_dir = f"saliencymaps_{saliency_type}_{model_timestamp}"
    os.makedirs(saliency_dir, exist_ok=True)
    
    # Create generated_images subfolder for the actual image outputs
    generated_images_dir = os.path.join(saliency_dir, "generated_images")
    os.makedirs(generated_images_dir, exist_ok=True)
    
    # Update paths with new naming scheme - keep files inside main saliency directory
    output_metrics = os.path.join(saliency_dir, f"metrics_{model_timestamp}.txt")
    csv_output = os.path.join(saliency_dir, f"metrics_{model_timestamp}.csv")
    
    print(f"Generating {saliency_type} saliency maps...")
    print(f"Output files will use base prefix: {model_timestamp}")
    print(f"Metrics file: {output_metrics}")
    print(f"CSV metrics: {csv_output}")
    print(f"Saliency maps: {generated_images_dir}")
    
    evaluate_yolo(args.detection_model, args.test_folder, args.label_folder, 
                output_metrics, generated_images_dir, args.conf_threshold, 
                args.iou_thresh, args.use_gradcam, csv_output)
