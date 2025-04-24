# Cassava Leaf Disease Classification Pipeline

This repository provides a pipeline to detect cassava leaves in images and classify their diseases using deep learning models (YOLO for detection and EfficientNet/other models for classification).

## Important Notice About Data and Model Files

This repository contains only the source code for the Cassava Disease Detection project. Due to file size limitations, the following directories and files are excluded and need to be obtained separately:

- `data/`: Contains the dataset used for training and evaluation
- `models/`: Contains trained model weights

**To get started, download the required data and models from the link below and place them in the corresponding `data/` and `models/` directories within the project structure:**

**[Download Data and Models (Zip Archive)](https://drive.google.com/file/d/1eXN2t4-hAA6BL1B218eEkLNxnWeIRMNb/view?usp=sharing)**

## Requirements
- Python 3.8+
- Conda (recommended)

## Installation & Environment Setup

1. Clone this repository and navigate to the project directory.
2. **Ensure you have downloaded and placed the required data and model weights** using the link provided in the "Important Notice" section above. Key models include:
   - **YOLOv9s detection model**: e.g., `models/yolov9s_training_results/yolov9s_best.pt`
   - **EfficientNet classification model**: e.g., `models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth`
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
python inference/cassava_detect_and_classify.py \
  --detection_model models/yolov9s_training_results/yolov9s_best.pt \
  --classification-model models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth \
  --test-folder path/to/image_folder \
  --output-folder path/to/output_folder \
  --num-images 8
```

- `--detection_model`: Path to the YOLO detection model weights (`.pt` file). Default: `models/yolov9s_training_results/yolov9s_best.pt`
- `--classification-model`: Path to the classification model weights (`.pth` file). Default: `models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth`
- `--test-folder`: Folder containing input images (jpg, jpeg, png). Default: `data/Cassava_Leaf_Detector.v1i.yolov8/test/images`
- `--output-folder`: Folder to save analysis results (visualizations, etc.). Default: auto-generated with model name and timestamp
- `--num-images`: Number of images to process. Default: `5`

If you run the script without arguments (`python inference/cassava_detect_and_classify.py`), it will process 5 images from the default test folder using the default models and save results to an auto-generated output folder.

### Example
Run on all images in `cassava_disease_example`:

```bash
python inference/cassava_detect_and_classify.py \
  --detection_model models/yolov9s_training_results/yolov9s_best.pt \
  --classification-model models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth \
  --test-folder cassava_disease_example \
  --output-folder disease_analysis_yolov9s_efficientnet_example \
  --num-images 1000
```

---

### 2. Detection Only
Detect cassava leaves in images and save detection visualizations:

```bash
python inference/cassava_detect_only.py \
  --detection_model models/yolov9s_training_results/yolov9s_best.pt \
  --test-folder path/to/image_folder \
  --output-folder detection_results \
  --num-images 8
```

- `--detection_model`: Path to the YOLO detection model weights (`.pt` file). Default: `models/yolov9s_training_results/yolov9s_best.pt`
- `--test-folder`: Folder containing input images. Default: `data/Cassava_Leaf_Detector.v1i.yolov8/test/images`
- `--output-folder`: Folder to save detection result images. Default: auto-generated with model name and timestamp
- `--num-images`: Number of images to process. Default: `5`
- `--conf-threshold`: Detection confidence threshold. Default: `0.4`

If you run the script without arguments (`python inference/cassava_detect_only.py`), it will process 5 images from the default test folder using the default detection model and save visualization results to an auto-generated output folder.

---

### 3. Classification Only
Classify cropped cassava leaf images using a trained classification model:

```bash
python inference/cassava_classify_only.py \
  --classification-model models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth \
  --test-folder path/to/leaf_folder \
  --output-folder classification_results \
  --num-images 8
```

- `--classification-model`: Path to the classification model weights (`.pth` file). Default: `models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth`
- `--test-folder`: Folder containing leaf images. Default: `data/preprocessed_leaf_classify/processed_dataset/test`
- `--output-folder`: Folder to save classification result files. Default: auto-generated with model name and timestamp
- `--num-images`: Number of images to process. Default: `20`

If you run the script without arguments (`python inference/cassava_classify_only.py`), it will classify 20 images from the default test folder using the default classification model and save results to an auto-generated output folder.

---

### 4. Best Leaf Classification
Find the highest confidence leaf in each image and classify only that leaf:

```bash
python inference/cassava_best_leaf_classify.py \
  --detection_model models/yolov9s_training_results/yolov9s_best.pt \
  --classification-model models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth \
  --test-folder path/to/image_folder \
  --output-folder best_leaf_results \
  --num-images 8
```

- `--detection_model`: Path to the YOLO detection model weights (`.pt` file). Default: `models/yolov9s_training_results/yolov9s_best.pt`
- `--classification-model`: Path to the classification model weights (`.pth` file). Default: `models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth`
- `--test-folder`: Folder containing input images. Default: `data/Cassava_Leaf_Detector.v1i.yolov8/test/images`
- `--output-folder`: Folder to save results. Default: auto-generated with model name and timestamp
- `--num-images`: Number of images to process. Default: `5`

If you run the script without arguments (`python inference/cassava_best_leaf_classify.py`), it will process 5 images from the default test folder using the default models and save results to an auto-generated output folder.

## Evaluation Tools

The repository includes several evaluation scripts to measure model performance and generate detailed metrics and visualizations.

### 1. YOLO Detection Evaluation

Evaluate a YOLO detection model, compute comprehensive metrics, and generate confidence-based heatmap visualizations:

```bash
python evaluate/test_yolo_metrics.py \
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
- `--detection_model`: Path to the YOLO detection model weights (`.pt` file). Default: `models/yolov9s_training_results/yolov9s_best.pt`
- `--test-folder`: Folder containing test images. Default: `data/Cassava_Leaf_Detector.v1i.yolov8/test/images`
- `--label-folder`: Folder containing YOLO-format label txt files (ground truth). Default: `data/Cassava_Leaf_Detector.v1i.yolov8/test/labels`
- `--output-metrics`: File to save detailed metrics summary. Default: auto-generated with model name and timestamp
- `--saliency-dir`: Folder to save heatmap visualizations. Default: auto-generated with model name and timestamp
- `--use-gradcam`: Generate confidence-based heatmaps (red = high confidence, blue = low). Default: `True`
- `--conf-threshold`: Detection confidence threshold. Default: `0.4`
- `--iou-thresh`: IoU threshold for TP/FP determination. Default: `0.5`

If you run the script without arguments (`python evaluate/test_yolo_metrics.py`), it will evaluate the default detection model on the default test set with the default parameters and save all results to auto-generated folders.

### 2. Classification Model Evaluation

Evaluate a classification model on a dataset of cropped leaf images:

```bash
python evaluate/test_classify_metrics.py \
  --model models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth \
  --data-dir data/cassava_disease_processed/test \
  --output-dir classification_evaluation_results
```

Key capabilities:
- Computes accuracy, precision, recall, F1-score per disease class
- Generates confusion matrix 
- Calculates top-k accuracy metrics (top-1, top-3)
- Identifies most confused classes and challenging examples
- Creates class activation maps showing model attention

Arguments:
- `--model`: Path to the classification model weights. Default: `models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth`
- `--data-dir`: Directory containing test images. Default: `data/preprocessed_leaf_classify/processed_dataset/test`
- `--output-dir`: Directory to save evaluation results. Default: `classification_metrics`
- `--device`: Device to run inference on. Default: `cuda` if available, otherwise `cpu`
- `--batch-size`: Batch size for evaluation. Default: `32`
- `--top-k`: Calculate top-k accuracy. Default: `3`
- `--num-vis-samples`: Number of samples to visualize. Default: `10`

If you run the script without arguments (`python evaluate/test_classify_metrics.py`), it will evaluate the default classification model on the default test dataset and save results to the default output directory.

### 3. Full Pipeline Evaluation (Detection + Classification)

Evaluate the complete detection and classification pipeline:

```bash
python evaluate/test_yolo_classify_metrics.py \
  --yolo-model models/yolov9s_training_results/yolov9s_best.pt \
  --classification-model models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth \
  --data-dir data/preprocessed_leaf_classify/processed_dataset/test \
  --output-dir pipeline_evaluation_results
```

Key capabilities:
- Evaluates both detection accuracy and classification performance
- Analyzes end-to-end pipeline performance
- Identifies where errors occur in the pipeline (detection vs. classification)
- Generates comprehensive visualization of results

Arguments:
- `--yolo-model`: Path to YOLO detection model weights. Default: `models/yolov9s_training_results/yolov9s_best.pt`
- `--classification-model`: Path to classification model weights. Default: `models/results_efficientnet_b3_tuned_ra_patience10/efficientnet_b3_cassava_best_tuned_ra.pth`
- `--data-dir`: Directory containing test images. Default: `data/preprocessed_leaf_classify/processed_dataset/test`
- `--output-dir`: Directory to save evaluation results. Default: `yolo_classification_metrics`
- `--device`: Device to run evaluation on. Default: `cuda` if available, otherwise `cpu`
- `--batch-size`: Batch size for evaluation. Default: `32`
- `--top-k`: Calculate top-k accuracy. Default: `3`
- `--num-vis-samples`: Number of samples to visualize. Default: `10`
- `--yolo-conf`: YOLO detection confidence threshold. Default: `0.4`
- `--max-images`: Maximum number of images to process. Default: process all images

If you run the script without arguments (`python evaluate/test_yolo_classify_metrics.py`), it will evaluate the default pipeline (both detection and classification models) on the default test dataset and save all results to the default output directory.

## Notes
- Make sure the model paths and folders exist.
- For best results, use the provided conda environment.
- All inference and evaluation tools will print progress and save visualizations for each image.
- All scripts generate timestamp-based output folders by default, so you won't overwrite previous results.

## Help
To see all available options, run any script with `--help`:

```bash
python inference/cassava_detect_and_classify.py --help
python evaluate/test_yolo_metrics.py --help
```

## Contact
For questions or issues, please open an issue in this repository.
