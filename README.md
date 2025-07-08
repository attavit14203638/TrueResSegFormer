# TrueResSegFormer: Optimizing SegFormer for Tree Crown Delineation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-ACPR%202025-red.svg)](link-to-paper-when-available)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/attavit14203638)

**🏆 State-of-the-Art Results: Boundary IoU 0.6201 on OAM-TCD Dataset**

This repository contains a state-of-the-art implementation for Tree Crown Delineation (TCD) using our enhanced TrueResSegformer architecture. The codebase provides a complete pipeline for training, evaluating, and deploying models for identifying and delineating individual tree crowns in aerial or satellite imagery with superior boundary accuracy.

## Key Innovations

- **TrueResSegformer Architecture**: Our enhanced SegFormer variant that produces segmentation maps at the full resolution of input images, improving boundary delineation without significantly increasing model complexity
- **Multi-Resolution Processing**: Supports processing of both high and low resolution imagery with detailed boundary preservation
- **Weighted Loss Strategy**: Adaptive class weighting to address severe imbalance in tree crown segmentation tasks

## Features

- **Unified Pipeline**: Centralized training, evaluation, and prediction operations
- **Robust Dataset Handling**: Consistent dataset processing with error detection and recovery
- **Optimization Techniques**: Mixed precision training, gradient accumulation, and class weighting
- **Comprehensive Visualization**: Tools for dataset inspection, prediction visualization, confidence maps, and performance analysis
- **Tiled Inference**: Support for processing large satellite/aerial images through tiling and stitching

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/username/tcd-segformer.git
cd tcd-segformer

# Install dependencies
pip install -r requirements.txt
```

### Command-Line Interface (CLI)

The primary way to interact with the project is through the `main.py` script using subcommands:

**1. Training a Model:**

```bash
# Basic training with default config values (if applicable)
python main.py train --output_dir ./training_output

# Training with specific parameters (overriding defaults or config file)
python main.py train \
    --dataset_name restor/tcd \
    --model_name nvidia/segformer-b0-finetuned-ade-512-512 \
    --output_dir ./my_trained_model \
    --num_epochs 15 \
    --learning_rate 6e-5 \
    --train_batch_size 4 \
    --mixed_precision

# Training using a base config file and overriding some parameters
python main.py train --config_path ./configs/base_config.json --learning_rate 7e-5 --output_dir ./tuned_model
```
*Use `python main.py train --help` for all options.*

**2. Making Predictions:**

```bash
# Predict on a single image using a trained model and its config
python main.py predict \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --image_paths ./path/to/your/image.png \
    --output_dir ./prediction_results

# Predict on multiple images with visualization and confidence maps
python main.py predict \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --image_paths ./images/img1.tif ./images/img2.tif \
    --output_dir ./prediction_results_batch \
    --visualize --show_confidence
```
*Use `python main.py predict --help` for all options.*

**3. Evaluating a Model:**

```bash
# Evaluate a trained model using its config
python main.py evaluate \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --output_dir ./evaluation_results

# Evaluate with specific evaluation parameters
python main.py evaluate \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --output_dir ./evaluation_results_custom \
    --eval_batch_size 32 \
    --no-visualize_worst
```
*Use `python main.py evaluate --help` for all options.*

### Training a Model (Python API)

```python
from config import Config
from pipeline import run_training_pipeline

# Create configuration
config = Config({
    "dataset_name": "restor/tcd",
    "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "output_dir": "./outputs",
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "train_batch_size": 8
})

# Run training pipeline
results = run_training_pipeline(config=config)
```

### Interactive Training

Use the `tcd_segformer_training_refactored.ipynb` notebook for interactive training and experimentation.

## Code Structure

The codebase is organized into modular components with clear separation of concerns:

| Module | Description |
|--------|-------------|
| `pipeline.py` | Centralized training and evaluation pipeline |
| `config.py` | Configuration management with validation |
| `dataset.py` | Dataset loading and processing with error handling |
| `model.py` | SegFormer model with high-resolution output |
| `metrics.py` | Evaluation metrics for segmentation tasks |
| `checkpoint.py` | Checkpoint management with metadata |
| `weights.py` | Class weight computation for handling imbalance |
| `visualization.py` | Visualization tools for images and results |
| `image_utils.py` | Image processing utilities |
| `exceptions.py` | Custom exception hierarchy |
| `main.py` | CLI entry point |

## Example Usage

### Dataset Inspection

```python
from inspect_dataset import inspect_dataset_samples

# Inspect dataset with visualization
inspect_dataset_samples(
    dataset_name="restor/tcd",
    num_samples=5,
    save_dir="./dataset_inspection",
    seed=42
)
```

### Prediction (Python API)

```python
# Using the prediction pipeline function
from config import Config
from pipeline import run_prediction_pipeline

# Load config used during training
config = Config.load("./my_trained_model/effective_train_config.json")

# Run prediction
results = run_prediction_pipeline(
    config=config,
    image_paths=["./path/to/your/image.png", "./another/image.tif"],
    model_path="./my_trained_model/final_model", # Optional, defaults based on config
    output_dir="./api_predictions", # Optional, defaults based on config
    visualize=True
)

# Access results (e.g., segmentation maps)
# segmentation_map = results["segmentation_maps"][0]
```

### Uploading to Hugging Face Hub

```bash
python upload_to_hub.py --model_dir ./outputs/final_model --repo_id username/tcd-segformer-model
```

## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{wilaiwongsakul2025tcd,
  title={TCD-SegFormer: Enhancing Tree Crown Delineation with True-Resolution Segmentation},
  author={Wilaiwongsakul, Attavit and Liang, Bin and Chen, Fang},
  booktitle={Asian Conference on Pattern Recognition (ACPR)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
