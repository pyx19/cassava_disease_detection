# Cassava Leaf Disease Classification Pipeline

This repository provides a pipeline to detect cassava leaves in images and classify their diseases using deep learning models (YOLO for detection and EfficientNet/other models for classification).

## Requirements
- Python 3.8+
- Conda (recommended)

## Installation & Environment Setup

1. Clone this repository and navigate to the project directory.
2. Ensure you have the required model weights:
   - **YOLOv9s detection model**: e.g., `models/yolov9s_training_results/yolov9s_best.pt`
   - **EfficientNet classification model**: e.g., `models/efficientnet_b3_cassava.pth`
3. Create and activate the conda environment:

```bash
conda create -n your_new_env python=3.9 -y
conda activate your_new_env
```

4. Install dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

This will install all required packages, including CUDA-enabled versions of torch and torchvision if you have a compatible GPU. If you need to add more packages, edit `requirements.txt` as needed.


## Usage

### 1. Full Detection and Classification Pipeline

```bash
python cassava_detect_and_classify.py \
  --detection_model models/yolov9s_training_results/yolov9s_best.pt \
  --classification-model models/efficientnet_b3_cassava.pth \
  --test-folder path/to/image_folder \
  --output-folder path/to/output_folder \
  --num-images 8
```

- `--detection_model`: Path to the YOLO detection model weights (`.pt` file).
- `--classification-model`: Path to the classification model weights (`.pth` file).
- `--test-folder`: Folder containing input images (jpg, jpeg, png).
- `--output-folder`: Folder to save analysis results (visualizations, etc.).
- `--num-images`: Number of images to process (default: 8; set to a large number to process all images).

### Example
Run on all images in `cassava_disease_example`:

```bash
python cassava_detect_and_classify.py \
  --detection_model models/yolov9s_training_results/yolov9s_best.pt \
  --classification-model models/efficientnet_b3_cassava.pth \
  --test-folder cassava_disease_example \
  --output-folder disease_analysis_yolov9s_efficientnet_example \
  --num-images 1000
```

---

### 2. Detection Only
Detect cassava leaves in images and save detection visualizations:

```bash
python cassava_detect_only.py \
  --detection_model models/yolov9s_training_results/yolov9s_best.pt \
  --test-folder path/to/image_folder \
  --output-folder detection_results \
  --num-images 8
```

- `--detection_model`: Path to the YOLO detection model weights (`.pt` file).
- `--test-folder`: Folder containing input images.
- `--output-folder`: Folder to save detection result images.
- `--num-images`: Number of images to process (optional).

---

### 3. Classification Only
Classify cropped cassava leaf images using a trained classification model:

```bash
python cassava_classify_only.py \
  --classification-model models/efficientnet_b3_cassava.pth \
  --test-folder path/to/cropped_leaf_folder \
  --output-folder classification_results \
  --num-images 8
```

- `--classification-model`: Path to the classification model weights (`.pth` file).
- `--test-folder`: Folder containing cropped leaf images.
- `--output-folder`: Folder to save classification result files.
- `--num-images`: Number of images to process (optional).


## Notes
- Make sure the model paths and folders exist.

---

### 4. Detection Evaluation & Metrics
Evaluate a YOLO detection model, compute comprehensive metrics, and generate confidence-based heatmap visualizations:

```bash
python test_yolo_metrics.py \
  --detection_model models/yolov9s_training_results/yolov9s_best.pt \
  --test-folder data/Cassava_Leaf_Detector.v1i.yolov8/test/images \
  --label-folder data/Cassava_Leaf_Detector.v1i.yolov8/test/labels \
  --output-metrics yolo_metrics_summary.txt \
  --saliency-dir detection_saliency_maps \
  --use-gradcam
```

Key capabilities:
- Computes detection quality metrics: mAP@0.5, precision, recall, IoU
- Measures average detection confidence
- Identifies false positives, false negatives, and NA cases
- Generates beautiful heatmap visualizations showing detection confidence

Arguments:
- `--detection_model`: Path to the YOLO detection model weights (`.pt` file)
- `--test-folder`: Folder containing test images
- `--label-folder`: Folder containing YOLO-format label txt files (ground truth)
- `--output-metrics`: File to save detailed metrics summary
- `--saliency-dir`: Folder to save heatmap visualizations
- `--use-gradcam`: Generate confidence-based heatmaps (red = high confidence, blue = low)
- `--conf-threshold`: Detection confidence threshold (default: 0.4)
- `--iou-thresh`: IoU threshold for TP/FP determination (default: 0.5)
- For best results, use the provided conda environment.
- The pipeline will print progress and save visualizations for each image.

## Help
To see all available options, run:

```bash
python cassava_detect_and_classify.py --help
python cassava_detect_only.py --help
python cassava_classify_only.py --help
python test_yolo_metrics.py --help

```

## Contact
For questions or issues, please open an issue in this repository.
