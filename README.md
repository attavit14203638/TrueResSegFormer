# Empirical Insights into Optimizing SegFormer for High-Fidelity Tree Crown Delineation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-ACPR%202025-red.svg)](https://github.com/attavit14203638/TrueResSegFormer)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/attavit14203638)

**🏆 State-of-the-Art Results: Optimized Standard SegFormer achieves Boundary IoU 0.6201 on OAM-TCD Dataset**

This repository contains the official codebase for our ACPR 2025 paper, "Empirical Insights into Optimizing SegFormer for High-Fidelity Tree Crown Delineation." It provides a comprehensive investigation into optimizing SegFormer-based architectures, presenting empirical evidence that **training strategy is more critical than architectural modifications** for achieving superior boundary delineation. 

The codebase includes our implementations for both standard SegFormer optimizations and the `TrueResSegFormer` architectural variant, allowing for full reproduction of our comparative analysis.

## 🔬 Key Research Findings

Our comprehensive study reveals:

- **🥇 Training Strategy Dominance**: An optimized standard SegFormer with class weighting and full-resolution (H×W) loss supervision achieves a **state-of-the-art Boundary IoU of 0.6201** on the OAM-TCD dataset.
- **📊 Architectural Analysis**: The `TrueResSegFormer` architectural variant (B-IoU: 0.5892), while an insightful exploration, is outperformed by an appropriately optimized standard SegFormer.
- **🎯 Critical Factors**: The combination of **external full-resolution supervision** (upsampling logits before loss calculation) and **class weighting** proves to be the most effective strategy.
- **⚡ Practical Impact**: Our findings suggest that methodical optimization of existing architectures offers a more direct path to performance gains than pursuing isolated architectural changes for this task.

## 🛠️ Experimental Configurations Implemented

This repository allows you to train and evaluate all configurations from our paper:

#### Standard SegFormer Variants
- **Baseline**: Standard SegFormer with H/4 loss.
- **Class Weighted**: Standard SegFormer + class weighting.
- **Full-Resolution Loss**: Standard SegFormer + external H×W loss supervision.
- **🏆 Optimized (Best)**: Standard SegFormer + class weighting + H×W loss (**B-IoU: 0.6201**).

#### TrueResSegFormer Architectural Variant
- **TrueResSegFormer**: Internal H×W logit upsampling architecture.
- **TrueResSegFormer + CW**: The above with class weighting (B-IoU: 0.5892).

## ✨ Features

- **🔬 Comparative Pipeline**: Systematically train and evaluate multiple SegFormer optimization strategies.
- **📈 Advanced Metrics**: Includes Boundary IoU (`B-IoU`) evaluation for precise boundary assessment.
- **⚖️ Class Imbalance Handling**: Integrated weighted loss strategies.
- **🖼️ Rich Visualization**: Tools for error analysis, boundary comparison, and prediction visualization.
- **🧮 Tiled Inference**: Support for processing large satellite/aerial images.
- **📊 Reproducible Research**: A complete experimental pipeline with detailed documentation.

## 🖥️ System Requirements

- **Python**: 3.8+ (tested with 3.8, 3.9, 3.10)
- **Operating System**: Linux, macOS, or Windows
- **Hardware**: 
  - **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for training)
  - **CPU**: 8+ cores (for CPU-only inference)
  - **RAM**: 16GB+ recommended
- **CUDA**: 11.8+ (if using GPU)

### ⏱️ Expected Training Times

| Configuration | GPU (V100) | GPU (RTX 4090) | CPU Only |
|---|---|---|---|
| 50 epochs (full paper results) | ~6-8 hours | ~4-6 hours | ~2-3 days |
| 10 epochs (quick experiment) | ~1.5 hours | ~1 hour | ~8-12 hours |
| 5 epochs (fast test) | ~45 minutes | ~30 minutes | ~4-6 hours |

## 🚀 Quick Start

### Installation

**Step 1: Clone the Repository**

```bash
# Clone the repository
git clone https://github.com/attavit14203638/TrueResSegFormer.git
cd TrueResSegFormer
```

**Step 2: Set Up Environment**

**Option 1: Using conda (Recommended for stability)**

```bash
# Create a new conda environment
conda create -n trueressegformer python=3.9
conda activate trueressegformer

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

**Option 2: Using pip with virtual environment**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (includes PyTorch)
pip install -r requirements.txt
```

**Option 3: CPU-only installation**

```bash
# If you only have CPU or want to test without GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 🧪 Quick Test

Verify your installation works correctly:

```bash
# Test 1: Check dataset access and basic imports
python -c "
from datasets import load_dataset
import torch
from transformers import SegformerForSemanticSegmentation
print('✅ All dependencies installed correctly!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test 2: Quick dataset inspection (should work out-of-the-box)
python main.py inspect --dataset_name restor/tcd --num_samples 3 --output_dir ./test_output

# Test 3: Check if training can start (stops after first batch)
python main.py train \
    --output_dir ./quick_test \
    --num_epochs 1 \
    --train_batch_size 1 \
    --learning_rate 1e-5
```

If all tests pass, you're ready to go! 🎉

### Training Models

**1. Reproduce Best Results (Optimized Standard SegFormer):**

```bash
# Train the state-of-the-art configuration (Std. SegFormer + CW + HxW Loss)
python main.py train \
    --dataset_name restor/tcd \
    --model_name nvidia/mit-b5 \
    --output_dir ./outputs/segformer_b5_cw_full_res \
    --num_epochs 50 \
    --learning_rate 1e-5 \
    --class_weights_enabled \
    --apply_loss_at_original_resolution
```

**2. Train TrueResSegFormer for Comparison:**

```bash
# Train the TrueResSegFormer architectural variant
python main.py train \
    --dataset_name restor/tcd \
    --model_name nvidia/mit-b5 \
    --output_dir ./outputs/TrueResSegformer_b5_cw \
    --use_true_res_segformer \
    --class_weights_enabled \
    --num_epochs 50 \
    --learning_rate 1e-5
```

### Evaluation and Prediction

```bash
# Evaluate any trained model
python main.py evaluate \
    --config_path ./outputs/segformer_b5_cw_full_res/effective_train_config.json \
    --model_path ./outputs/segformer_b5_cw_full_res/final_model \
    --output_dir ./evaluation_results

# Make predictions on new images
python main.py predict \
    --config_path ./outputs/segformer_b5_cw_full_res/effective_train_config.json \
    --model_path ./outputs/segformer_b5_cw_full_res/final_model \
    --image_paths ./path/to/image.png \
    --output_dir ./predictions \
    --visualize --show_confidence
```

## 🔧 Troubleshooting

### Common Issues and Solutions

**1. PyTorch/CUDA Installation Issues**
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Dataset Access Issues**
```bash
# If you get dataset loading errors
export HF_DATASETS_CACHE="/path/to/cache"  # Set custom cache location
huggingface-cli login  # Login if needed for dataset access
```

**3. Memory Issues**
```bash
# For GPU out of memory errors, reduce batch size:
python main.py train --train_batch_size 1 --gradient_accumulation_steps 8

# For CPU training (slower but works with limited GPU memory):
python main.py train --device cpu --train_batch_size 2
```

**4. Dependency Conflicts**
```bash
# If you encounter version conflicts, try:
pip install --force-reinstall -r requirements.txt

# Or create a fresh environment:
conda create -n fresh_env python=3.9
conda activate fresh_env
# Then install as above
```

**5. Slow Training**
```bash
# Enable mixed precision for faster training:
python main.py train --mixed_precision

# Use smaller model for quick testing:
python main.py train --model_name nvidia/mit-b0  # instead of mit-b5
```

### Getting Help

- **Check the logs**: Training logs are saved in `{output_dir}/training.log`
- **Use CLI help**: `python main.py train --help` for all available options
- **Reduce complexity**: Start with smaller models/datasets to test your setup
- **Hardware monitoring**: Use `nvidia-smi` to monitor GPU usage during training

### Expected File Structure After Training

After a successful training run, you should see:
```
outputs/segformer_b5_cw_full_res/
├── effective_train_config.json
├── final_model/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── best_checkpoint/
├── training.log
└── tensorboard_logs/
```

## 📁 Code Structure

The codebase is organized into modular components for clarity and reproducibility:

| Module | Description |
|---|---|
| **Core Modules** | |
| `pipeline.py` | Unified training/evaluation pipeline for all model variants. |
| `model.py` | Implementations for SegFormer and the `TrueResSegformer` variant. |
| `config.py` | Centralized configuration management and validation. |
| `dataset.py` | OAM-TCD dataset loading and processing. |
| `main.py` | Command-line interface for running all experiments. |
| **Utilities & Helpers** | |
| `metrics.py` | Comprehensive evaluation metrics including Boundary IoU. |
| `visualization.py` | Tools for generating plots and visual comparisons. |
| `checkpoint.py` | Manages saving and loading of model checkpoints. |
| `weights.py` | Class weight computation for imbalanced datasets. |
| `image_utils.py` | Helper functions for image manipulation (tiling, stitching). |
| `tensorboard_utils.py` | Utilities for logging metrics to TensorBoard. |
| `utils.py` | General helper functions (e.g., seeding, logging setup). |
| `exceptions.py` | Custom exception hierarchy for robust error handling. |
| `inspect_dataset.py`| Script to visualize and inspect dataset samples. |

### 📓 Jupyter Notebooks

- **`tcd_segformer_pipeline.ipynb`**: Provides a high-level, interactive way to run the training and evaluation pipeline, ideal for experimentation.
- **`visualize_validation_predictions_enhanced.ipynb`**: A detailed notebook for visualizing and analyzing the qualitative results of model predictions on the validation set.

## 🔬 Reproducing Paper Results

Our paper demonstrates the following performance hierarchy on the OAM-TCD test set:

| Configuration | IoU | F1-Score | **Boundary IoU** | Notes |
|---------------|-----|----------|-------------------|-------|
| **Std SegFormer + CW + H×W Loss** | **0.848** | **0.918** | **🏆 0.620** | **State-of-the-art** |
| Std SegFormer + H×W Loss | 0.828 | 0.906 | 0.610 | Shows H×W loss impact |
| Std SegFormer + CW + H/4 Loss | 0.844 | 0.916 | 0.606 | Strong baseline |
| Std SegFormer (baseline) | 0.817 | 0.899 | 0.590 | Standard configuration |
| TrueResSegFormer + CW | 0.838 | 0.912 | 0.589 | Architectural variant |
| TrueResSegFormer (no CW) | 0.797 | 0.887 | 0.577 | Without optimization |

You can reproduce any of these results by using the appropriate flags in the `main.py train` command.

## 📄 Citation

If you use this work in your research, please cite our ACPR 2025 paper:

```bibtex
@inproceedings{wilaiwongsakul2025empirical,
  title={Empirical Insights into Optimizing SegFormer for High-Fidelity Tree Crown Delineation},
  author={Wilaiwongsakul, Attavit and Liang, Bin and Jia, Wenfeng and Chen, Fang},
  booktitle={Asian Conference on Pattern Recognition (ACPR)},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
