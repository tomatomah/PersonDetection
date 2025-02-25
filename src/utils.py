import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    """
    if not os.path.isfile(config_path):
        print(f"'{config_path}' does not exist.")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(e)
        sys.exit(1)

    return config


def set_device(gpu_id: int) -> torch.device:
    """
    Check if the specified GPU ID is available
    and return the appropriate device (GPU or CPU).
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if gpu_id < 0:
        return torch.device("cpu")

    if torch.cuda.device_count() <= gpu_id:
        return torch.device("cpu")

    return torch.device(f"cuda:{gpu_id}")


def set_params(model: nn.Module, decay: float) -> list[dict[str, list | float]]:
    """
    Separate model parameters into two groups for applying weight decay.
    """

    # Parameters without weight decay
    no_decay_params = []

    # Parameters with weight decay
    decay_params = []

    # Get all normalization layer types (BatchNorm, LayerNorm etc.)
    norm_types = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)

    # Iterate through all model modules
    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            # Skip parameters that don't require gradients
            if not param.requires_grad:
                continue

            # Group parameters based on type
            if param_name == "bias":
                # Bias parameters don't get weight decay
                no_decay_params.append(param)
            elif param_name == "weight" and isinstance(module, norm_types):
                # Normalization layer weights don't get weight decay
                no_decay_params.append(param)
            else:
                # All other weights get weight decay
                decay_params.append(param)

    # Return parameter groups with their decay settings
    return [{"params": no_decay_params, "weight_decay": 0.00}, {"params": decay_params, "weight_decay": decay}]


class LinearLR(object):
    """
    Linear Learning Rate Scheduler with warmup and decay phases.
    """

    def __init__(self, max_lr: float, min_lr: float, warmup_epochs: int, total_epochs: int, num_steps: int) -> None:
        warmup_steps = int(max(warmup_epochs * num_steps, 100))  # minimum 100 steps
        decay_steps = int(total_epochs * num_steps - warmup_steps)

        warmup_lr = np.linspace(start=min_lr, stop=max_lr, num=int(warmup_steps), endpoint=False)
        decay_lr = np.linspace(start=max_lr, stop=min_lr, num=decay_steps)

        self.total_lr = np.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer) -> None:
        """
        Update learning rate for current step.
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.total_lr[step]


class CosineLR(object):
    """
    Cosine Learning Rate Scheduler with warmup and decay phases.
    """

    def __init__(self, max_lr: float, min_lr: float, warmup_epochs: int, total_epochs: int, num_steps: int) -> None:
        warmup_steps = int(max(warmup_epochs * num_steps, 100))  # minimum 100 steps
        decay_steps = int(total_epochs * num_steps - warmup_steps)

        warmup_lr = np.linspace(start=min_lr, stop=max_lr, num=int(warmup_steps))
        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = np.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer) -> None:
        """
        Update learning rate for current step.
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.total_lr[step]


class EMA(object):
    """
    Updated Exponential Moving Average (EMA) implementation for model parameter tracking.
    Maintains a moving average of model parameters to provide more stable predictions.
    """

    def __init__(self, model: nn.Module, decay=0.9999, tau=2000, updates=0) -> None:
        # Create a copy of the model and set it to evaluation mode
        self.ema = copy.deepcopy(model).eval()

        # Track number of EMA updates
        self.updates = updates

        # Define decay rate calculation function
        # Effective decay rate approaches 'decay' as x increases
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

        # Disable gradient computation for EMA model parameters
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        # Perform update without gradient computation
        with torch.no_grad():
            # Increment update counter
            self.updates += 1

            # Calculate current decay rate based on update count
            current_decay_rate = self.decay(self.updates)

            # Get current model state
            current_model_params = model.state_dict()

            # Update each EMA model parameter
            for param_name, ema_param in self.ema.state_dict().items():
                # Only update floating point parameters
                if ema_param.dtype.is_floating_point:
                    # EMA update formula: new_value = decay * old_value + (1 - decay) * current_value
                    # Apply decay to old value
                    ema_param.mul_(current_decay_rate)
                    # Incorporate current value with weighted average
                    ema_param.add_(current_model_params[param_name].detach(), alpha=1 - current_decay_rate)
