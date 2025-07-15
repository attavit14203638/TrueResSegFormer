# Setup Guide for Experimentation Notebooks

This guide helps you configure the notebooks to work with your local environment after cloning the repository.

## Path Configuration

The notebooks have been generalized to work with different user environments. Here's how to configure them:

### 1. Environment Variables (Recommended)

Set these environment variables before running the notebooks:

```bash
# For the main pipeline notebook
export TEST_IMAGE_PATH="/path/to/your/test_image.tif"

# For the visualization notebook (sets the base directory for finding model checkpoints)
export PROJECT_ROOT="/path/to/your/TrueResSegFormer_Release"
```

### 2. Direct Path Modification

If you prefer not to use environment variables, you can directly modify the paths in the notebooks:

#### In `tcd_segformer_pipeline.ipynb`:
```python
# Replace this line:
image_path = os.environ.get('TEST_IMAGE_PATH', './data/test_image.tif')

# With your actual path:
image_path = "/your/actual/path/to/test_image.tif"
```

#### In `visualize_validation_predictions.ipynb`:
```python
# The use of `__file__` is unreliable in notebooks. Replace this line:
project_root_abs = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.abspath(__file__)))

# With a more reliable method if not using environment variables:
import os
project_root_abs = os.path.abspath('.') # Assumes you run the notebook from the project root

# With your actual path:
project_root_abs = "/your/actual/path/to/TrueResSegFormer_Release"
```

### 3. Directory Structure Setup

The visualization notebook needs to locate your trained model outputs. These are typically saved in the `output_dir` you specify during training. For example:

```
TrueResSegFormer_Release/
├── outputs/
│   └── train_test/                <-- Your output_dir
│       ├── best_checkpoint/         <-- Your model checkpoint
│       ├── effective_train_config.json
│       └── ...
├── tcd_segformer_pipeline.ipynb
└── visualize_validation_predictions.ipynb
```

**Important**: Update the paths in the visualization notebook to match your actual directory structure:

```python
# Update these paths to match your model checkpoints:
config_path = os.path.join(project_root_abs, "your_actual_run_dir/config.json")
model_checkpoint_path = os.path.join(project_root_abs, "your_actual_checkpoint_dir")
```

## Data Directory Setup

For the main pipeline notebook, create a data directory:

```bash
mkdir -p TrueResSegFormer_Release/data
# Place your test images in this directory
```

## Running the Notebooks

1. **Main Pipeline** (`tcd_segformer_pipeline.ipynb`):
   - Provides end-to-end training, evaluation, and prediction
   - Requires minimal setup - just configure test image path
   - Works with default configuration

2. **Visualization** (`visualize_validation_predictions.ipynb`):
   - Requires existing trained model checkpoints
   - Needs specific directory structure (see above)
   - Update paths to match your trained models

## Troubleshooting

### Common Issues:

1. **"Config file not found"**: Update the `config_path` variable to point to your actual config file
2. **"Checkpoint not found"**: Update the `model_checkpoint_path` to point to your trained model
3. **"Image not found"**: Ensure `TEST_IMAGE_PATH` points to a valid image file
4. **Import errors**: Ensure you're running the notebook from the correct directory

### Tips:

- Use absolute paths when in doubt
- Check file existence with `os.path.exists(path)` before running
- Enable logging to see detailed path information
- Create the expected directory structure if missing

## Example Configuration

Here's a complete example setup:

```bash
# Set environment variables
export PROJECT_ROOT="/home/username/projects/TrueResSegFormer_Release"
export TEST_IMAGE_PATH="/home/username/data/test_image.tif"

# Or create a setup script
cat > setup_paths.sh << 'EOF'
#!/bin/bash
export PROJECT_ROOT="$(pwd)"
export TEST_IMAGE_PATH="./data/test_image.tif"
echo "Paths configured for TrueResSegFormer"
EOF

chmod +x setup_paths.sh
source setup_paths.sh
```

Then run your notebooks as usual.
