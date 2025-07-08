#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model definition and configuration for TCD-SegFormer model.
"""

import torch
import torch.nn as nn
import os # Added for os.path.exists
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, AutoConfig, SchedulerType as HfSchedulerType
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.optimization import get_scheduler

from typing import Dict, Tuple, Optional, Union, List, Any
import logging
from enum import Enum

# Import from refactored modules
from config import Config
from utils import get_logger

# Setup module logger
logger = get_logger()

class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"

class TrueResSegformer(SegformerForSemanticSegmentation):
    """
    Extended SegFormer model that outputs predictions at the same resolution as the input images.
    Includes support for weighted loss to handle class imbalance.
    """
    def __init__(self, config, project_config: Config, class_weights=None, apply_class_weights=True):
        super().__init__(config) # config here is the HF SegformerConfig
        self.project_config = project_config # Store the project's Config object
        
        # Explicitly enable gradient checkpointing support
        self.supports_gradient_checkpointing = True
        
        self.output_full_resolution = True
        
        if class_weights is not None:
            logger.info(f"TrueResSegformer: Using class weights: {class_weights.tolist() if class_weights is not None else 'None'}")
        self.class_weights = class_weights
        self.apply_class_weights = apply_class_weights
        # Focal loss parameters removed as per user request

    def _set_gradient_checkpointing(self, enable=False, gradient_checkpointing_func=None):
        """
        Enable gradient checkpointing for memory efficiency.
        """
        if hasattr(self.segformer, "gradient_checkpointing"):
            self.segformer.gradient_checkpointing = enable
        
        if hasattr(self.segformer, "encoder"):
            if hasattr(self.segformer.encoder, "gradient_checkpointing"):
                self.segformer.encoder.gradient_checkpointing = enable
            else:
                setattr(self.segformer.encoder, "gradient_checkpointing", enable)
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = super().forward(
            pixel_values=pixel_values,
            labels=None,  # Don't compute loss in base model
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True  # Always use return_dict=True for the base model
        )
        
        logits = outputs.logits
        
        if self.output_full_resolution and pixel_values is not None:
            input_shape = pixel_values.shape[-2:]
            interp_mode = self.project_config.get("interpolation_mode", "bilinear")
            align_corners = self.project_config.get("interpolation_align_corners", False)
            align_corners_param = align_corners if interp_mode != 'nearest' else None

            upsampled_logits = F.interpolate(
                logits,
                size=input_shape,
                mode=interp_mode,
                align_corners=align_corners_param
            )
        else:
            upsampled_logits = logits
        
        loss = None
        if labels is not None:
            if self.apply_class_weights:
                # Use CrossEntropyLoss with class_weights
                loss_fct = nn.CrossEntropyLoss(
                    weight=self.class_weights.to(upsampled_logits.device) if self.class_weights is not None else None,
                    ignore_index=self.config.semantic_loss_ignore_index
                )
            else:
                # Use standard CrossEntropyLoss without class weights
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=self.config.semantic_loss_ignore_index
                )
            loss = loss_fct(upsampled_logits, labels)
        
        if not return_dict:
            # outputs[0] is the original logits, outputs[1:] are hidden_states, attentions etc.
            # We need to replace outputs.logits with upsampled_logits in the returned tuple.
            output_tuple = (upsampled_logits,) 
            if outputs.hidden_states is not None:
                output_tuple += (outputs.hidden_states,)
            if outputs.attentions is not None:
                output_tuple += (outputs.attentions,)
            
            return ((loss,) + output_tuple) if loss is not None else output_tuple
        
        return SemanticSegmenterOutput(
            loss=loss,
            logits=upsampled_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Union[str, SchedulerType], 
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5, 
    power: float = 1.0,      
    patience: int = 3,       
    factor: float = 0.1,     
    min_lr_val: float = 0,
    config_obj: Optional[Config] = None
) -> Union[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau, None]:
    
    current_scheduler_type_enum: Optional[SchedulerType] = None
    hf_scheduler_type_enum: Optional[HfSchedulerType] = None 

    if isinstance(scheduler_type, str):
        try:
            current_scheduler_type_enum = SchedulerType(scheduler_type.lower())
        except ValueError:
            logger.info(f"Scheduler type '{scheduler_type}' not in custom SchedulerType enum. Trying Hugging Face types.")
            try:
                hf_scheduler_type_enum = HfSchedulerType(scheduler_type.lower())
            except ValueError:
                logger.error(f"Unknown scheduler type string: '{scheduler_type}'. Must be one of {[e.value for e in SchedulerType]} or {[e.value for e in HfSchedulerType]}. Defaulting to Hugging Face linear.")
                # Default to Hugging Face linear if string is not recognized by either enum
                hf_scheduler_type_enum = HfSchedulerType.LINEAR 
    elif isinstance(scheduler_type, SchedulerType): 
        current_scheduler_type_enum = scheduler_type
    elif isinstance(scheduler_type, HfSchedulerType): 
         hf_scheduler_type_enum = scheduler_type
    else:
        logger.error(f"Invalid scheduler_type type: {type(scheduler_type)}. Defaulting to Hugging Face linear.")
        hf_scheduler_type_enum = HfSchedulerType.LINEAR

    if current_scheduler_type_enum == SchedulerType.REDUCE_ON_PLATEAU:
        # For ReduceLROnPlateau, it is expected to monitor a metric where smaller is better (e.g., loss).
        # The actual metric value (e.g., validation_loss) is passed during scheduler.step() in the training loop.
        scheduler_mode = 'min'
        monitored_metric_for_log = "validation_loss" # Clarify in log what it's *intended* for.
        
        # The 'verbose' argument for ReduceLROnPlateau is deprecated in PyTorch 1.10+ 
        # and will be removed in 2.0. PyTorch issues a warning if used.
        # It's better to rely on standard logging for LR updates if using newer PyTorch.
        # If compatibility with PyTorch <1.10 is critical, one might add it conditionally:
        # verbose_kwarg = {'verbose': True} if torch.__version__ < '1.10.0' else {}
        # and then **verbose_kwarg in the constructor call.

        logger.info(f"Creating ReduceLROnPlateau scheduler: patience={patience}, factor={factor}, min_lr={min_lr_val}, mode='{scheduler_mode}' (intended to monitor '{monitored_metric_for_log}')")
        
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=factor,
            patience=patience,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=min_lr_val, 
            eps=1e-8 
        )
    
    # Determine the scheduler name for Hugging Face's get_scheduler
    scheduler_name_for_hf: Union[str, HfSchedulerType]
    if hf_scheduler_type_enum: # Prioritize if it was directly an HfSchedulerType or resolved to one
        scheduler_name_for_hf = hf_scheduler_type_enum
    elif current_scheduler_type_enum: # If it was our custom enum, try to map its value
        try:
            scheduler_name_for_hf = HfSchedulerType(current_scheduler_type_enum.value)
        except ValueError:
            logger.error(f"Custom scheduler type '{current_scheduler_type_enum.value}' not directly mappable to Hugging Face schedulers. Defaulting to Hugging Face linear.")
            scheduler_name_for_hf = HfSchedulerType.LINEAR
    else: # Should not happen if logic above is correct, but as a fallback
        logger.error("Scheduler type determination failed unexpectedly. Defaulting to Hugging Face linear.")
        scheduler_name_for_hf = HfSchedulerType.LINEAR

    # Log warnings if specific parameters might not be used by the chosen scheduler type in transformers
    # Ensure scheduler_name_for_hf is a string for comparison if it's an enum
    scheduler_name_str = scheduler_name_for_hf.value if isinstance(scheduler_name_for_hf, Enum) else str(scheduler_name_for_hf)

    if scheduler_name_str == "cosine_with_restarts" and num_cycles != 0.5: # HfSchedulerType.COSINE_WITH_RESTARTS.value
         logger.warning(f"num_cycles={num_cycles} provided, but the standard transformers get_scheduler might use a default value for cosine_with_restarts.")
    if scheduler_name_str == "polynomial" and power != 1.0: # HfSchedulerType.POLYNOMIAL.value
         logger.warning(f"power={power} provided, but the standard transformers get_scheduler might use a default value for polynomial.")

    try:
        scheduler = get_scheduler(
            name=scheduler_name_for_hf, 
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        log_name = scheduler_name_for_hf.value if isinstance(scheduler_name_for_hf, Enum) else str(scheduler_name_for_hf)
        logger.info(f"Created Hugging Face scheduler: type={log_name}, warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")
        return scheduler
    except Exception as e:
        logger.error(f"Failed to create Hugging Face scheduler '{scheduler_name_for_hf}': {e}. Falling back to Hugging Face Linear.")
        return get_scheduler(
            name=HfSchedulerType.LINEAR, 
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    return None # Should be unreachable if defaults work

def create_model(
    config: Config,
    num_training_steps: int,
    class_weights: Optional[torch.Tensor] = None,
    apply_loss_at_original_resolution: bool = True,
    logger: Optional[logging.Logger] = None  # Added logger parameter
) -> Tuple[nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    if logger is None: # Fallback logger if not provided
        logger = get_logger()

    logger.info(f"--- Model Creation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for model creation and potential checkpoint loading.")

    model_name = config["model_name"]
    logger.info(f"Attempting to load model: {model_name}")

    if class_weights is not None:
        logger.info(f"Class weights enabled. Weights: {class_weights.tolist()}")
    else:
        logger.info("Class weights disabled; proceeding with uniform weighting.")

    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    id2label = config["id2label"]
    label2id = config["label2id"]
    num_labels = len(id2label)

    model_hf_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if config.get("use_true_res_segformer", True):
        logger.info(f"Configuration 'use_true_res_segformer' is True. Initializing TrueResSegformer for '{model_name}'.")
        try:
            base_model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                config=model_hf_config,
                ignore_mismatched_sizes=True,
            )
            logger.info(f"Successfully loaded pre-trained weights for base Segformer '{model_name}'.")
            model = TrueResSegformer(
                config=model_hf_config,
                project_config=config,
                class_weights=class_weights,
                apply_class_weights=class_weights is not None
            )
            model.segformer = base_model.segformer
            model.decode_head = base_model.decode_head
            logger.info(f"Successfully created TrueResSegformer and copied pre-trained weights from '{model_name}'.")
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights for TrueResSegformer base ('{model_name}'): {e}. Initializing TrueResSegformer from scratch.")
            model = TrueResSegformer(
                config=model_hf_config,
                project_config=config,
                class_weights=class_weights,
                apply_class_weights=class_weights is not None
            )
    else:
        logger.info(f"Configuration 'use_true_res_segformer' is False. Initializing standard SegformerForSemanticSegmentation for '{model_name}'.")
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                config=model_hf_config,
                ignore_mismatched_sizes=True,
            )
            logger.info(f"Successfully loaded pre-trained standard SegformerForSemanticSegmentation: '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load standard SegformerForSemanticSegmentation model '{model_name}': {e}. Raising error.")
            raise

        if class_weights is not None and config.get("class_weights_enabled", False):
            logger.info(f"Class weights will be applied to standard Segformer via a custom loss wrapper.")
            original_forward = model.forward
            def forward_with_weighted_loss(*args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                if 'labels' in kwargs and kwargs['labels'] is not None and outputs.loss is not None:
                    labels = kwargs['labels']
                    logits = outputs.logits
                    if logits.shape[-2:] != labels.shape[-2:]:
                        interp_mode = 'bilinear'
                        align_corners = False
                        if hasattr(model, 'project_config'):
                            interp_mode = model.project_config.get('interpolation_mode', 'bilinear')
                            align_corners = model.project_config.get('interpolation_align_corners', False)
                        align_corners_param = align_corners if interp_mode != 'nearest' else None
                        if apply_loss_at_original_resolution:
                            logits_for_loss = F.interpolate(logits, size=labels.shape[-2:], mode=interp_mode, align_corners=align_corners_param)
                            labels_for_loss = labels
                            # logger.info(f"Applying class weights at original resolution: logits upsampled to {logits_for_loss.shape[-2:]}") # Too verbose for every forward pass
                        else:
                            logits_for_loss = logits
                            labels_for_loss = F.interpolate(labels.float().unsqueeze(1), size=logits.shape[-2:], mode='nearest').long().squeeze(1)
                            # logger.info(f"Applying class weights at model resolution: labels downsampled to {labels_for_loss.shape[-2:]}") # Too verbose
                    else:
                        logits_for_loss = logits
                        labels_for_loss = labels
                    loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device), ignore_index=model.config.semantic_loss_ignore_index)
                    weighted_loss = loss_fct(logits_for_loss, labels_for_loss)
                    outputs.loss = weighted_loss
                return outputs
            model.forward = forward_with_weighted_loss
            logger.info("Applied custom weighted loss wrapper to standard Segformer.")
        else:
            logger.info("No class weights provided or class weights disabled for standard Segformer. Standard loss calculation will apply.")

    model.to(device) # Move model to device before optimizer creation and checkpoint loading

    # --- Checkpoint Loading (Model Weights) ---
    if config.get("resume_from_checkpoint", False):
        model_weights_path = config.get("resume_from_checkpoint_path")
        if model_weights_path and os.path.exists(model_weights_path):
            logger.info(f"Attempting to load model weights from: {model_weights_path}")
            try:
                state_dict = torch.load(model_weights_path, map_location=device)
                # Handle potential nested state_dict (e.g. if saved with 'model_state_dict' key)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict: # another common key
                    state_dict = state_dict['state_dict']
                
                model.load_state_dict(state_dict, strict=True) # strict=True for fine-tuning same architecture
                logger.info(f"Successfully loaded model weights from {model_weights_path}.")
            except Exception as e:
                logger.error(f"Error loading model weights from {model_weights_path}: {e}. Model will use initial/pretrained weights.")
        else:
            logger.warning(f"resume_from_checkpoint is True, but path '{model_weights_path}' is invalid or not found. Model will use initial/pretrained weights.")
    else:
        logger.info(f"Not resuming model weights from checkpoint. Model uses initial/pretrained weights for {model_name}.")


    optimizer = create_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    logger.info(f"Optimizer created with initial LR: {learning_rate}, Weight Decay: {weight_decay}")

    # --- Optimizer State Loading ---
    if config.get("resume_from_checkpoint", False):
        optimizer_state_path = config.get("resume_optimizer_path") # Get path from config
        if optimizer_state_path and os.path.exists(optimizer_state_path): # Check if path is not None/empty and exists
            logger.info(f"Attempting to load optimizer state from: {optimizer_state_path}")
            try:
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
                logger.info(f"Successfully loaded optimizer state from {optimizer_state_path}.")
                
                # CRITICAL: Apply the new learning rate from the fine-tuning config
                logger.info(f"Applying new learning rate ({config.get('learning_rate', 'KEY_NOT_FOUND')}) to all parameter groups in the loaded optimizer.")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config["learning_rate"]
                logger.info(f"New learning rate {config.get('learning_rate', 'KEY_NOT_FOUND')} applied to optimizer.")
            except Exception as e:
                logger.error(f"Failed to load optimizer state from {optimizer_state_path}: {e}. Optimizer will use initial state with new LR.")
        else:
            logger.warning(f"Optimizer state file not found at {optimizer_state_path}. Optimizer will use initial state with new LR: {config.get('learning_rate', 'KEY_NOT_FOUND')}.")


    # --- Scheduler Instantiation (New Scheduler) ---
    scheduler = None
    scheduler_type_name = config.get("scheduler_type", "linear").lower()
    logger.info(f"Attempting to create scheduler of type: '{scheduler_type_name}'.")

    if scheduler_type_name == "reduce_on_plateau":
        scheduler_monitor_metric = config.get("scheduler_monitor", "val_loss") # Default to val_loss if not specified
        scheduler_mode = config.get("scheduler_mode", "min") # Default to min if not specified
        
        logger.info(f"Creating ReduceLROnPlateau scheduler: "
                    f"Monitoring '{scheduler_monitor_metric}' in '{scheduler_mode}' mode. "
                    f"Patience={config.get('scheduler_patience', 10)}, Factor={config.get('scheduler_factor', 0.1)}, MinLR={config.get('min_lr_scheduler', 0)}.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=config.get('scheduler_factor', 0.1),
            patience=config.get('scheduler_patience', 10),
            min_lr=config.get('min_lr_scheduler', 0),
            verbose=True # Logs when LR changes
        )
    else:
        # For other schedulers, use the existing Hugging Face get_scheduler logic
        num_warmup_steps = int(config.get("warmup_ratio", 0.0) * num_training_steps) # Default warmup_ratio to 0.0 if not present
        logger.info(f"Using Hugging Face get_scheduler for '{scheduler_type_name}' with {num_warmup_steps} warmup steps for {num_training_steps} total training steps.")
        scheduler = create_scheduler( # Call the existing helper for HF schedulers
            optimizer=optimizer,
            scheduler_type=scheduler_type_name, # Pass the string name
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=config.get("num_cycles", 0.5),
            power=config.get("power", 1.0),
            # These are not directly used by HF get_scheduler but are passed to our create_scheduler helper
            patience=config.get("scheduler_patience", 3),
            factor=config.get("scheduler_factor", 0.1),
            min_lr_val=config.get("min_lr_scheduler", 0),
            config_obj=config
        )

    if config.get("gradient_checkpointing", False):
        logger.info("Configuration 'gradient_checkpointing' is True. Attempting to enable...")
        try:
            if hasattr(model, 'supports_gradient_checkpointing') and model.supports_gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logger.info(f"Called model.gradient_checkpointing_enable() for {model.__class__.__name__}.")
                elif hasattr(model, '_set_gradient_checkpointing'): 
                     model._set_gradient_checkpointing(enable=True)
                     logger.info(f"Called model._set_gradient_checkpointing(enable=True) for {model.__class__.__name__}.")
                elif hasattr(model, 'segformer') and hasattr(model.segformer, 'encoder'): 
                    model.segformer.encoder.gradient_checkpointing = True
                    logger.info("Manually set model.segformer.encoder.gradient_checkpointing = True.")
                else:
                    logger.warning(f"Model {model.__class__.__name__} claims support but no known method to enable gradient checkpointing was found.")
            elif isinstance(model, SegformerForSemanticSegmentation) and not isinstance(model, TrueResSegformer): 
                if hasattr(model, 'segformer') and hasattr(model.segformer, 'encoder'):
                    model.segformer.encoder.gradient_checkpointing = True
                    logger.info("Manually set model.segformer.encoder.gradient_checkpointing = True for standard Segformer.")
                else:
                    logger.warning(f"Standard Segformer {model_name} does not have .segformer.encoder. Cannot enable gradient checkpointing.")
            else: 
                 logger.warning(f"Gradient checkpointing not configured for model type: {model.__class__.__name__}. Check model's `supports_gradient_checkpointing` or structure.")
        except Exception as e:
            logger.error(f"Error during gradient checkpointing setup: {e}", exc_info=True)
    else:
        logger.info("Configuration 'gradient_checkpointing' is False. Skipping gradient checkpointing setup.")
    logger.info(f"--- Model Creation Complete ---")
    return model, optimizer, scheduler

def create_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
) -> torch.optim.Optimizer:
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )
    
    return optimizer

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, module in model.named_modules():
        for param_name, _ in module.named_parameters():
            if param_name == "weight" and not any(isinstance(module, layer_type) for layer_type in forbidden_layer_types):
                if name:
                    result.append(f"{name}.{param_name}")
                else:
                    result.append(param_name)
    return result
