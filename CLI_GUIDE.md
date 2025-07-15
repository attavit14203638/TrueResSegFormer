# Command-Line Interface (CLI) Guide

This guide provides a comprehensive overview of how to use the command-line interface (CLI) provided by `main.py` to run all experiments, evaluations, and predictions from our research paper.

## Prerequisites

Before using this CLI guide, make sure you have:

1. **Completed installation** as described in the [README.md](README.md#installation)
2. **Verified your setup** using the [Quick Test](README.md#quick-test) section
3. **Activated your environment**: `conda activate trueressegformer` or `source venv/bin/activate`

## Quick Reference

**Most Common Commands:**

```bash
# Quick test your installation (requires a base_config.json, see below)
python main.py inspect --dataset_name restor/tcd --num_samples 3 --output_dir ./test_output

# Train the best model from our paper (requires a base_config.json)
# 1. First, modify base_config.json to set "apply_loss_at_original_resolution": true
# 2. Then, run the train command:
python main.py train --config_path ./base_config.json --compute_class_weights --output_dir ./best_model

# Evaluate a trained model  
python main.py evaluate --config_path ./best_model/effective_train_config.json --model_path ./best_model/final_model

# Make predictions on new images
python main.py predict --config_path ./best_model/effective_train_config.json --image_paths ./your_image.png
```

## 1. Main Entry Point

All commands are initiated through the `main.py` script. It uses a subcommand structure for different operations. You can always get help for any command or subcommand by using the `-h` or `--help` flag.

```bash
# Get help for the main script and see available subcommands
python main.py --help
```

**Available Subcommands:**
- `train`: Train a new model configuration.
- `evaluate`: Evaluate a trained model on the test set.
- `predict`: Make predictions on new images using a trained model.
- `inspect`: Visualize samples from the dataset.

---

## 2. Training Models (`train`)

The `train` subcommand is used to run the training pipeline for any model configuration.

```bash
# Get help for the train command
python main.py train --help
```

### Key Training Arguments

| Argument | Flag | Description | Default |
|---|---|---|---|
| **Model & Data** | | | |
| `--dataset_name` | | Hugging Face dataset identifier. | `restor/tcd` |
| `--model_name` | | Base model from Hugging Face Hub. | `nvidia/mit-b5` |
| `--output_dir` | | Directory to save checkpoints and results. | Required |
| **Training Strategy** | | | |
| `--config_path` | | Path to a base JSON configuration file. | Required |
| `--use_true_res_segformer` | | Use the `TrueResSegformer` architectural variant (overrides config). | `False` |
| `apply_loss_at_original_resolution`| (in config) | Upsample standard SegFormer logits to HxW for loss calculation. | `False` |
| `--compute_class_weights` | | Use weighted cross-entropy loss to handle class imbalance (overrides config). | `False` |
| **Hyperparameters** | | | |
| `--num_epochs` | | Number of training epochs. | 10 |
| `--learning_rate` | `-lr` | Initial learning rate for the optimizer. | `6e-5` |
| `--train_batch_size`| `-b` | Batch size for the training dataloader. | 4 |
| `--gradient_accumulation_steps` | | Number of steps to accumulate gradients before updating weights. | 1 |
| `--mixed_precision`| | Use mixed precision (FP16) for training to save memory. | `False` |


### Training Examples

#### Example 1: Reproduce State-of-the-Art Results
This command trains the best-performing model from the paper: an **optimized standard SegFormer**.

```bash
# First, ensure 'base_config.json' has "apply_loss_at_original_resolution": true
python main.py train \
    --config_path ./base_config.json \
    --output_dir ./outputs/segformer_b5_cw_full_res \
    --num_epochs 50 \
    --learning_rate 1e-5 \
    --compute_class_weights
```
*This configuration combines class weighting and full-resolution loss supervision on a standard SegFormer.*

#### Example 2: Train the `TrueResSegformer` Architectural Variant
This command trains the `TrueResSegformer` architecture with class weights for comparison.

```bash
# First, ensure 'base_config.json' has "use_true_res_segformer": true
python main.py train \
    --config_path ./base_config.json \
    --output_dir ./outputs/TrueResSegformer_b5_cw \
    --dataset_name restor/tcd \
    --num_epochs 50 \
    --learning_rate 1e-5 \
    --use_true_res_segformer \
    --class_weights_enabled
```

#### Example 3: Train a Basic Baseline Model
This trains a standard SegFormer without any of the key optimizations.

```bash
python main.py train \
    --output_dir ./outputs/segformer_b5_baseline \
    --model_name nvidia/mit-b5 \
    --dataset_name restor/tcd \
    --num_epochs 50 \
    --learning_rate 1e-5
```

#### Example 4: Quick Test Training (Fast)
For testing your setup with a short training run:

```bash
python main.py train \
    --output_dir ./quick_test \
    --num_epochs 2 \
    --train_batch_size 2 \
    --learning_rate 1e-5 \
    --mixed_precision
```

---

## 3. Evaluating Models (`evaluate`)

The `evaluate` subcommand runs a trained model on the test split of the dataset and computes all relevant metrics, including **Boundary IoU**.

```bash
# Get help for the evaluate command
python main.py evaluate --help
```

### Key Evaluation Arguments

| Argument | Flag | Description |
|---|---|---|
| `--config_path` | | Path to the `effective_train_config.json` file from a training run. |
| `--model_path` | | Path to the `final_model` directory containing the trained model weights. |
| `--output_dir` | | Directory to save evaluation results and visualizations. |
| `--visualize_worst`| | Save visualizations of the worst-performing predictions. |
| `--no-visualize_worst`| | Disable visualization of worst predictions. |

### Evaluation Example

```bash
# Evaluate the best-performing model
python main.py evaluate \
    --config_path ./outputs/segformer_b5_cw_full_res/effective_train_config.json \
    --model_path ./outputs/segformer_b5_cw_full_res/final_model \
    --output_dir ./evaluation_results \
    --visualize_worst
```

---

## 4. Making Predictions (`predict`)

The `predict` subcommand uses a trained model to generate segmentation masks for new, unseen images.

```bash
# Get help for the predict command
python main.py predict --help
```

### Key Prediction Arguments

| Argument | Flag | Description |
|---|---|---|
| `--config_path` | | Path to the `effective_train_config.json` file from a training run. |
| `--model_path` | | Path to the `final_model` directory containing the trained model weights. |
| `--image_paths` | | One or more paths to the input images you want to predict on. |
| `--output_dir` | | Directory to save the output segmentation masks. |
| `--visualize` | | Save visualizations of the predictions overlaid on the original images. |
| `--show_confidence` | | Generate and save confidence maps along with predictions. |

### Prediction Example

```bash
# Predict on two new images with visualization and confidence maps
python main.py predict \
    --config_path ./outputs/segformer_b5_cw_full_res/effective_train_config.json \
    --model_path ./outputs/segformer_b5_cw_full_res/final_model \
    --image_paths ./new_data/image1.png ./new_data/image2.tif \
    --output_dir ./prediction_outputs \
    --visualize \
    --show_confidence
```

---

## 5. Inspecting the Dataset (`inspect_dataset`)

The `inspect_dataset` subcommand is a utility script to visualize samples from the dataset, which is useful for verification and understanding the data.

```bash
# Get help for the inspect_dataset command
python main.py inspect_dataset --help
```

### Key Inspection Arguments

| Argument | Flag | Description |
|---|---|---|
| `--dataset_name` | | Hugging Face dataset identifier. |
| `--num_samples` | | Number of samples to visualize. |
| `--save_dir` | | Directory to save the visualization images. |
| `--seed` | | Random seed for selecting samples to ensure reproducibility. |

### Inspection Example

```bash
# Visualize 10 random samples from the training split
python main.py inspect_dataset \
    --dataset_name restor/tcd \
    --num_samples 10 \
    --save_dir ./dataset_inspection_output \
    --seed 42
```

---

## 6. Tips and Best Practices

### Memory Management
```bash
# For limited GPU memory, use smaller batch sizes with gradient accumulation:
python main.py train --train_batch_size 1 --gradient_accumulation_steps 8

# Enable mixed precision to save memory:
python main.py train --mixed_precision
```

### Monitoring Training
```bash
# Monitor with TensorBoard (logs saved automatically):
tensorboard --logdir ./outputs/your_model_name/tensorboard_logs

# Check training progress:
tail -f ./outputs/your_model_name/training.log
```

### Quick Experiments
```bash
# Use smaller model for faster testing:
python main.py train --model_name nvidia/mit-b0

# Shorter training for quick tests:
python main.py train --num_epochs 5
```

For more detailed troubleshooting, see the [Troubleshooting section](README.md#troubleshooting) in the README. 