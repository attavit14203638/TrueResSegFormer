#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union

"""
Unified Command-Line Interface for TCD-SegFormer.

This script serves as the main entry point for interacting with the TCD-SegFormer
project. It provides subcommands for training, evaluation, prediction, and dataset inspection.

Subcommands:
  train     Train a new segmentation model.
  predict   Generate segmentation predictions for input images using a trained model.
  evaluate  Evaluate a trained model on a dataset.
  inspect   Inspect dataset samples and analyze dataset statistics.

Use 'python main.py <subcommand> --help' for specific options.
"""

import os
import argparse
import torch
import time
import logging
from typing import Dict, Optional, Any, Tuple
import sys # Added sys for exit

# Import the centralized configuration and pipeline modules
from config import Config, load_config_from_args, load_config_from_file_and_args
from dataset import (
    TCDDataset, 
    load_and_shuffle_dataset, 
    create_dataloaders, 
    create_eval_dataloader,
    create_augmentation_transform
)

from inspect_dataset import (
    examine_raw_annotations, 
    examine_dataset_statistics, 
    inspect_dataset_samples, 
    verify_training_tiling
)
from visualization import plot_augmented_samples
from pipeline import run_training_pipeline, run_prediction_pipeline, evaluate_model
from utils import setup_logging, log_or_print, get_logger
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from checkpoint import verify_checkpoint
from exceptions import ConfigurationError, FileNotFoundError, DatasetError

# --- Argument Parsing Setup ---

def create_parent_parser() -> argparse.ArgumentParser:
    """Creates a parent parser for common arguments."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--output_dir", type=str, default=None,
                               help="Base directory for outputs (logs, models, visualizations). Overrides config.")
    parent_parser.add_argument("--config_path", type=str, default=None,
                               help="Path to a base JSON config file to load (required for predict/evaluate).")
    parent_parser.add_argument("--seed", type=int, default=None,
                               help="Random seed for reproducibility (overrides config).")
    parent_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                               help="Enable verbose logging.")
    return parent_parser

def setup_train_parser(subparsers, parent_parser):
    """Adds arguments for the 'train' subcommand."""
    parser = subparsers.add_parser("train", help="Train a new model", parents=[parent_parser])
    # Dataset parameters (specific overrides)
    parser.add_argument("--dataset_name", type=str,
                        help="HuggingFace dataset name (overrides config)")
    parser.add_argument("--image_size", type=int, nargs=2,
                        help="Size to resize images to (overrides config)")
    parser.add_argument("--validation_split", type=float,
                        help="Fraction for validation split (overrides config)")

    # Model parameters
    parser.add_argument("--model_name", type=str,
                        help="Base model name (overrides config)")
    # Using a different name to avoid conflict with parent parser
    parser.add_argument("--train_output_dir", type=str, dest="train_output_dir",
                        help="Directory for training outputs (overrides config and parent output_dir)")

    # Training parameters
    parser.add_argument("--train_batch_size", type=int,
                        help="Training batch size (overrides config)")
    parser.add_argument("--eval_batch_size", type=int,
                        help="Evaluation batch size (overrides config)")
    parser.add_argument("--num_epochs", type=int,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--learning_rate", type=float,
                        help="Learning rate (overrides config)")
    parser.add_argument("--weight_decay", type=float,
                        help="Weight decay (overrides config)")
    # Seed is already defined in parent parser, no need to redefine it here
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction,
                        help="Use mixed precision (overrides config)")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        help="Gradient accumulation steps (overrides config)")
    parser.add_argument("--num_workers", type=int,
                        help="Dataloader workers (overrides config)")
    parser.add_argument("--compute_class_weights", action=argparse.BooleanOptionalAction,
                        help="Compute class weights (overrides config)")
    parser.add_argument("--use_true_res_segformer", action=argparse.BooleanOptionalAction, default=None,
                        help="Use TrueResSegformer model (overrides config).")

    # Scheduler parameters (specific overrides)
    # Note: scheduler_type, patience, factor are already in config.py's DEFAULT_CONFIG
    # and can be overridden by a JSON config file.
    # Adding min_lr_scheduler here for direct CLI control.
    parser.add_argument("--min_lr_scheduler", type=float, default=None,
                        help="Minimum learning rate for schedulers like ReduceLROnPlateau (overrides config).")
    # warmup_ratio, num_cycles, power are also in DEFAULT_CONFIG.

    # Logging parameters
    parser.add_argument("--logging_steps", type=int,
                        help="Log every X steps (overrides config)")
    parser.add_argument("--eval_steps", type=int,
                        help="Evaluate every X steps (overrides config)")
    parser.add_argument("--save_steps", type=int,
                        help="Save checkpoint every X steps (overrides config)")

    # Visualization parameters
    parser.add_argument("--visualize_worst", action=argparse.BooleanOptionalAction, default=True,
                        help="Visualize worst predictions during eval (overrides config)")
    parser.add_argument("--num_worst_samples", type=int, default=None,
                        help="Number of worst samples to show (overrides config)")
    # Add flags for evaluation options during training
    parser.add_argument("--analyze_errors", action=argparse.BooleanOptionalAction, default=None,
                        help="Perform error analysis during evaluation steps (overrides config).")
    parser.add_argument("--visualize_confidence_comparison", action=argparse.BooleanOptionalAction, default=None,
                        help="Visualize confidence comparison during evaluation steps (overrides config).")

    parser.set_defaults(func=handle_train)

def setup_predict_parser(subparsers, parent_parser):
    """Adds arguments for the 'predict' subcommand."""
    parser = subparsers.add_parser("predict", help="Make predictions with a trained model", parents=[parent_parser])
    # config_path is required for predict, so we'll modify the parent's argument
    for action in parent_parser._actions:
        if '--config_path' in action.option_strings:
            action.required = True
            action.help = "Path to the JSON config file from training output. (required for predict)"
            break
    parser.add_argument("--image_paths", type=str, required=True, nargs='+',
                        help="Path(s) to input image(s).")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model directory (defaults to output_dir/final_model).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for prediction.")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate and save prediction visualizations.")
    parser.add_argument("--show_confidence", action=argparse.BooleanOptionalAction, default=False,
                        help="Generate and save confidence map visualizations.")
    parser.add_argument("--show_class_activation_maps", action=argparse.BooleanOptionalAction, default=False,
                        help="Generate and save class activation map visualizations.")
    parser.set_defaults(func=handle_predict)

def setup_evaluate_parser(subparsers, parent_parser):
    """Adds arguments for the 'evaluate' subcommand."""
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained model", parents=[parent_parser])
    # config_path is required for evaluate, so we'll modify the parent's argument
    for action in parent_parser._actions:
        if '--config_path' in action.option_strings:
            action.required = True
            action.help = "Path to the JSON config file from training output. (required for evaluate)"
            break
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory to evaluate.")
    # Dataset/Eval specific overrides
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name (overrides config).")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Evaluation batch size (overrides config).")
    parser.add_argument("--visualize_worst", action=argparse.BooleanOptionalAction, default=None, # Default None to use config
                        help="Visualize worst predictions (overrides config).")
    parser.add_argument("--num_worst_samples", type=int, default=None,
                        help="Number of worst samples to show (overrides config).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Dataloader workers (overrides config).")
    parser.add_argument("--validation_split", type=float, default=None,
                        help="Validation split fraction if needed (overrides config).")
    parser.add_argument("--analyze_errors", action=argparse.BooleanOptionalAction, default=None,
                        help="Perform error analysis during evaluation (overrides config).")
    parser.add_argument("--visualize_confidence_comparison", action=argparse.BooleanOptionalAction, default=None,
                        help="Visualize confidence comparison during evaluation (overrides config).")

    parser.set_defaults(func=handle_evaluate)

def setup_inspect_parser(subparsers, parent_parser):
    """Adds arguments for the 'inspect' subcommand."""
    # Create a new parent parser without the required config_path
    inspect_parent_parser = argparse.ArgumentParser(add_help=False)
    for action in parent_parser._actions:
        if '--config_path' not in action.option_strings:
            inspect_parent_parser._add_action(action)
    
    parser = subparsers.add_parser("inspect", help="Inspect dataset samples", parents=[inspect_parent_parser])
    
    # Add config_path as optional for inspect
    parser.add_argument("--config_path", type=str, default=None,
                      help="Optional path to a config file for additional settings")
    
    # Add required dataset name
    parser.add_argument("--dataset_name", type=str, required=True,
                      help="HuggingFace dataset name")
    
    # Add other inspect-specific arguments
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to inspect")
    
    # Set default output directory
    parser.set_defaults(output_dir="./dataset_inspection")
    
    # Add optional flags
    parser.add_argument("--enhanced_vis", action=argparse.BooleanOptionalAction, default=True,
                      help="Use enhanced visualization techniques")
    parser.add_argument("--analyze_statistics", action=argparse.BooleanOptionalAction, default=True,
                      help="Generate and save dataset statistics")
    parser.add_argument("--max_attempts", type=int, default=15,
                      help="Maximum number of attempts to find valid samples")
    
    # Set the handler function
    parser.set_defaults(func=handle_inspect)

def setup_verify_tiling_parser(subparsers, parent_parser):
    """Adds arguments for the 'verify-tiling' subcommand."""
    parser = subparsers.add_parser("verify-tiling", help="Verify training tiling configuration", parents=[parent_parser])
    # config_path is required for verify-tiling, so we'll modify the parent's argument
    for action in parent_parser._actions:
        if '--config_path' in action.option_strings:
            action.required = True
            action.help = "Path to the JSON config file to verify. (required for verify-tiling)"
            break
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to check.")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False,
                        help="Visualize the fetched tiles during verification.")
    parser.set_defaults(func=handle_verify_tiling)


# --- Helper Functions ---

def _load_model_for_eval_predict(model_path: str, device: torch.device, logger: logging.Logger) -> Optional[torch.nn.Module]:
    """Loads a trained SegFormer model for prediction or evaluation."""
    if not verify_checkpoint(model_path):
        logger.error(f"Model not found or invalid at path: {model_path}")
        return None
    try:
        logger.info(f"Loading model from {model_path}...")
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        model = model.to(device)
        model.eval() # Set to evaluation mode
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return None

def _determine_output_dir(args: argparse.Namespace, config: Optional[Config]) -> str:
    """Determines the final output directory based on args and config."""
    # Command-specific defaults if output_dir is not provided via CLI or config
    default_dirs = {
        'train': './outputs',
        'predict': './outputs/predictions',
        'evaluate': './outputs/evaluation',
        'inspect': './dataset_inspection'
    }

    # 1. Use CLI --output_dir if provided
    if args.output_dir:
        return args.output_dir

    # 2. Use config's output_dir if config exists
    if config and config.get("output_dir"):
        # Adjust for predict/evaluate if no specific CLI output_dir was given
        if args.command == 'predict':
            return os.path.join(config["output_dir"], "predictions")
        elif args.command == 'evaluate':
            return os.path.join(config["output_dir"], "evaluation")
        else: # train or other commands using config
            return config["output_dir"]

    # 3. Use command-specific default
    return default_dirs.get(args.command, ".")


# --- Command Handlers ---

def handle_train(args: argparse.Namespace, logger: logging.Logger, config: Config, device: torch.device):
    """Handles the 'train' subcommand."""
    logger.info("Mode: Training")
    # Config is already loaded and merged

    # Use train_output_dir if provided, otherwise fall back to output_dir
    output_dir = args.train_output_dir if args.train_output_dir else config.get("output_dir", "./outputs")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the final effective config
    config.save(os.path.join(output_dir, "effective_train_config.json"))
    logger.info(f"Effective training config saved to {output_dir}/effective_train_config.json")

    # Update the output_dir in config to ensure consistency
    config["output_dir"] = output_dir
    
    results = run_training_pipeline(
        config=config,
        logger=logger,
        is_notebook=False # Assuming CLI is not a notebook
    )
    logger.info(f"Training completed. Model saved to {results.get('model_dir', 'unknown')}")
    logger.info(f"Final metrics: {results.get('metrics', 'No metrics available')}")

def handle_predict(args: argparse.Namespace, logger: logging.Logger, config: Config, device: torch.device):
    """Handles the 'predict' subcommand."""
    logger.info("Mode: Prediction")
    # Config is loaded

    # Determine model path
    model_path = args.model_path if args.model_path else os.path.join(config.get("output_dir", "."), "final_model") # Use config's output_dir as base if model_path not given

    # Load model using helper
    model = _load_model_for_eval_predict(model_path, device, logger)
    if model is None:
        return # Error logged in helper

    # Output directory is already determined and created in main
    output_dir = args.output_dir

    logger.info(f"Predicting using model: {model_path}")
    logger.info(f"Input images: {args.image_paths}")
    logger.info(f"Output directory: {output_dir}")

    # Pass the loaded model to the pipeline function
    results = run_prediction_pipeline(
        config=config,
        image_paths=args.image_paths,
        model_path=model_path, # Pass model_path for reference, though model obj is used
        output_dir=output_dir,
        batch_size=args.batch_size,
        visualize=args.visualize,
        show_confidence=args.show_confidence,
        show_class_activation_maps=args.show_class_activation_maps,
        logger=logger,
        is_notebook=False
        # Note: run_prediction_pipeline currently re-loads the model,
        # ideally it should accept the loaded model object. Refactoring pipeline.py is needed for that.
    )
    logger.info(f"Prediction completed. Outputs saved to {output_dir}")

def handle_evaluate(args: argparse.Namespace, logger: logging.Logger, config: Config, device: torch.device):
    """
    Handles the 'evaluate' subcommand.
    """
    try:
        # Create output directory
        output_dir = _determine_output_dir(args, config)
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging to file in the output directory
        log_filename = f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = os.path.join(output_dir, log_filename)
        setup_logging(output_dir=output_dir, filename=log_filename, 
                     log_level=logging.INFO, file_log_level=logging.DEBUG)
        
        logger.info("Starting model evaluation...")
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Output will be saved to: {output_dir}")
        logger.info(f"Using device: {device}")
        
        # Load the model
        logger.info("Loading model...")
        model = _load_model_for_eval_predict(args.model_path, device, logger)
        
        # Load the dataset
        logger.info("Loading dataset...")
        dataset_dict = load_and_shuffle_dataset(
            dataset_name=args.dataset_name or config.get("dataset_name"),
            seed=args.seed or config.get("seed", 42)
        )
        
        # Get the image processor
        image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b5")
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        eval_batch_size = args.eval_batch_size or config.get("eval_batch_size", 8)
        num_workers = args.num_workers or config.get("num_workers", 4)
        
        _, eval_dataloader, _, _ = create_dataloaders(
            dataset_dict=dataset_dict,
            image_processor=image_processor,
            config=config,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers
        )
        
        # Run evaluation
        logger.info("Starting evaluation...")
        metrics = evaluate_model(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device,
            output_dir=output_dir,
            visualize_worst=args.visualize_worst if args.visualize_worst is not None else config.get("visualize_worst", False),
            num_worst_samples=args.num_worst_samples or config.get("num_worst_samples", 5),
            analyze_errors=args.analyze_errors if args.analyze_errors is not None else config.get("analyze_errors", False),
            visualize_confidence_comparison=args.visualize_confidence_comparison if args.visualize_confidence_comparison is not None else config.get("visualize_confidence_comparison", False)
        )
        
        # Log metrics
        logger.info("\nEvaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"{key}: {value:.4f}")
        
        logger.info("\nEvaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise

def handle_inspect(args: argparse.Namespace, logger: logging.Logger, config: Optional[Config], device: torch.device):
    """
    Handles the 'inspect' subcommand.
    """
    logger.info("Mode: Dataset Inspection")
    
    # Output directory is args.output_dir (renamed from save_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Set log level based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)

    try:
        start_time = time.time()
        logger.info(f"Starting dataset inspection for {args.dataset_name}...")

        # If config_path was provided, load the config
        if args.config_path:
            logger.info(f"Loading config from {args.config_path}")
            try:
                config = Config.load(args.config_path)
                logger.info("Config loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load config from {args.config_path}: {e}")
                logger.warning("Continuing with default configuration.")
                config = None
        else:
            logger.info("No config file provided, using default settings.")
            config = None
        
        # Setup logging to file in the output directory
        log_filename = f"inspect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = os.path.join(output_dir, log_filename)
        setup_logging(output_dir=output_dir, filename=log_filename, 
                     log_level=logging.INFO, file_log_level=logging.DEBUG)
        
        logger.info(f"Starting dataset inspection for {args.dataset_name}")
        logger.info(f"Output will be saved to: {output_dir}")
        logger.info(f"Using device: {device}")
        
        # Create a default config if none provided
        if config is None:
            from config import Config
            config = Config({
                "augmentation": {
                    "apply": True,
                    "random_crop_size": [512, 512],
                    "horizontal_flip": True,
                    "vertical_flip": True,
                    "rotation_range": 15,
                    "brightness_range": [0.8, 1.2],
                    "contrast_range": [0.8, 1.2],
                    "saturation_range": [0.8, 1.2],
                    "hue_range": [-0.1, 0.1],
                    "normalize": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    }
                },
                "id2label": {0: "background", 1: "tree"},
                "label2id": {"background": 0, "tree": 1},
                "ignore_index": 255
            })
            logger.info("Using default configuration with augmentations enabled")
        
        # Load the dataset
        logger.info(f"\nLoading dataset: {args.dataset_name}")
        dataset_dict = load_and_shuffle_dataset(
            dataset_name=args.dataset_name,
            seed=args.seed
        )
        logger.info(f"Dataset loaded successfully. Available splits: {list(dataset_dict.keys())}")
        
        # 1. Examine raw annotations
        logger.info("\n1. Examining raw annotations...")
        raw_annots_dir = os.path.join(output_dir, "raw_annotations")
        os.makedirs(raw_annots_dir, exist_ok=True)
        
        # Pass the dataset_dict to avoid reloading
        examine_raw_annotations(
            dataset_name=args.dataset_name,
            num_samples=args.num_samples,
            save_dir=raw_annots_dir,
            seed=args.seed,
            enhanced_vis=args.enhanced_vis,
            dataset_dict=dataset_dict,
            logger=logger,
            is_notebook=False
        )
        logger.info(f"Raw annotations inspection complete. Results saved to {raw_annots_dir}")

        # 2. Create and inspect the dataset with augmentations
        logger.info("\n2. Creating and inspecting dataset with augmentations...")
        # 3. Create dataset with augmentations for visualization
        try:
            # Create image processor
            image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b5")
            logger.info("Using default image processor (nvidia/mit-b5).")
            
            # Create a config with augmentations enabled for training
            aug_config = Config({"augmentation": {"apply": True, "p": 1.0}})
            
            # Create the training dataset with augmentations
            train_dataset = TCDDataset(
                dataset=dataset_dict["train"],
                image_processor=image_processor,
                config=aug_config,
                split="train"
            )
            logger.info("Training dataset with augmentations created successfully.")
            
            # 4. Visualize augmented samples if transform is available
            if hasattr(train_dataset, 'transform') and train_dataset.transform is not None:
                logger.info("\n3. Visualizing augmented samples...")
                aug_output_dir = os.path.join(output_dir, "augmented_samples")
                os.makedirs(aug_output_dir, exist_ok=True)
                
                # Get a sample from the training dataset
                sample = train_dataset[0]  # Get first sample
                
                # Visualize multiple augmented versions of the same sample
                from inspect_dataset import plot_augmented_samples
                plot_augmented_samples(
                    original_sample={"image": sample["pixel_values"], "mask": sample["labels"]},
                    transform=train_dataset.transform,
                    num_augmented=4,
                    save_path=os.path.join(aug_output_dir, "augmentation_examples.png")
                )
                logger.info(f"Augmentation examples saved to: {aug_output_dir}")
            else:
                logger.info("\n3. Skipping augmentation visualization - no transform available")
            
        except Exception as e:
            logger.error(f"Failed to create training dataset or visualize augmentations: {e}")
            logger.warning("Skipping augmentation visualization due to error.")
            
        # 5. Inspect dataset samples
        logger.info("\n4. Inspecting dataset samples...")
        inspect_output_dir = os.path.join(output_dir, "sample_inspections")
        os.makedirs(inspect_output_dir, exist_ok=True)
        
        # Create a dataset without augmentations for inspection
        inspect_dataset_samples(
            dataset_or_dict=dataset_dict,
            image_processor=image_processor,
            save_dir=inspect_output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            enhanced_vis=args.enhanced_vis,
            logger=logger
        )
        logger.info(f"Sample inspections saved to: {inspect_output_dir}")

        # 5. Examine dataset statistics
        logger.info("\n5. Examining dataset statistics...")
        stats_output_dir = os.path.join(output_dir, "statistics")
        os.makedirs(stats_output_dir, exist_ok=True)
        
        try:
            # Create a dataset for statistics (without augmentations)
            stats_dataset = TCDDataset(
                dataset=dataset_dict["train"],
                image_processor=image_processor,
                config=config,
                split="train"
            )
            
            stats = examine_dataset_statistics(
                dataset=stats_dataset,
                save_dir=stats_output_dir,
                num_samples=min(100, len(dataset_dict["train"])),
                logger=logger
            )
            logger.info(f"Dataset statistics saved to: {stats_output_dir}")
        except Exception as e:
            logger.error(f"Error calculating dataset statistics: {e}", exc_info=True)
            logger.warning("Skipping dataset statistics due to error.")
        
        logger.info("\nDataset inspection completed successfully!")
        logger.info(f"All results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error during dataset inspection: {e}", exc_info=True)
        raise

    # Update config with any dataset-specific information
    if args.dataset_name: 
        config["dataset_name"] = args.dataset_name
        
    # Log completion
    logger.info("\nDataset inspection completed successfully!")
    logger.info(f"All results saved to: {output_dir}")

def handle_verify_tiling(args: argparse.Namespace, logger: logging.Logger, config: Config, device: torch.device): # config and device loaded in main
    """Handles the 'verify-tiling' subcommand."""
    logger.info("Mode: Verify Training Tiling")
    # Config is already loaded in main based on args.config_path

    # Call the verification function
    verification_passed = verify_training_tiling(
        config=config, # Pass the loaded config
        num_samples=args.num_samples,
        visualize=args.visualize,
        logger=logger,
        is_notebook=False
    )

    if verification_passed:
        logger.info("Tiling verification successful.")
        sys.exit(0) # Exit with success code
    else:
        logger.error("Tiling verification failed.")
        sys.exit(1) # Exit with failure code


# --- Main Execution ---

from inspect_dataset import verify_training_tiling # Import the function - Already imported above, ensure it's there

def main():
    """Main entry point for the script."""
    parent_parser = create_parent_parser()
    parser = argparse.ArgumentParser(description="TCD-SegFormer: Train, Evaluate, Predict, Inspect, Verify Tiling")
    subparsers = parser.add_subparsers(title="Available Commands", dest="command", required=True)

    # Setup parsers for each command, passing the parent
    setup_train_parser(subparsers, parent_parser)
    setup_predict_parser(subparsers, parent_parser)
    setup_evaluate_parser(subparsers, parent_parser)
    setup_inspect_parser(subparsers, parent_parser)
    setup_verify_tiling_parser(subparsers, parent_parser) # Add the new command parser

    # Parse arguments
    args = parser.parse_args()

    # --- Centralized Setup ---
    config = None
    logger = None
    output_dir = "." # Default

    try:
        # 1. Load Configuration (if applicable)
        # For verify-tiling, predict, evaluate, config_path is required by their parsers
        if args.command in ['predict', 'evaluate', 'verify-tiling']:
             # config_path is already checked by argparse 'required=True'
             config = Config.load(args.config_path)
             # Apply common CLI overrides if they exist in args
             if hasattr(args, 'seed') and args.seed is not None: config["seed"] = args.seed
             # output_dir handled below
        elif args.command == 'train':
             # Train loads config and merges args internally
             config = load_config_from_file_and_args(args.config_path, args)
        # 'inspect' doesn't load a main config file by default

        # 2. Determine Output Directory
        output_dir = _determine_output_dir(args, config)
        args.output_dir = output_dir # Store final output_dir back into args namespace for handlers

        # 3. Setup Logging
        os.makedirs(output_dir, exist_ok=True)
        log_level = logging.DEBUG if hasattr(args, 'verbose') and args.verbose else logging.INFO
        # Use a command-specific log file name
        log_file_name = f"{args.command}.log"
        logger = setup_logging(output_dir, log_level=log_level, filename=log_file_name)

        logger.info(f"Executing command: {args.command}")
        logger.info(f"Arguments: {vars(args)}")
        if config:
            logger.debug(f"Loaded/Merged Config: {config.to_dict()}") # Log full config only in debug

        # 4. Set Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # 5. Execute Command Handler
        args.func(args, logger, config, device) # Pass common objects

    except (ConfigurationError, FileNotFoundError, DatasetError) as e:
        # Log errors using the logger if available, otherwise print
        if logger:
            logger.error(f"Pipeline error: {e}", exc_info=True)
        else:
            print(f"ERROR: Pipeline error: {e}")
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        else:
            print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
