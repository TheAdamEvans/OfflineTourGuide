"""
Activation recording module for capturing model activations during forward passes.

This module provides hooks to record activations from transformer models,
which are then used for activation-based pruning analysis.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import numpy as np
from pathlib import Path
import json
import pickle


class ActivationRecorder:
    """
    Records activations from model layers during forward passes.
    
    This class registers hooks on specified layers and collects activation
    statistics for pruning analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
        save_dir: Optional[str] = None
    ):
        """
        Initialize activation recorder.
        
        Args:
            model: PyTorch model to record activations from
            target_layers: List of layer names to record (None = all transformer layers)
            save_dir: Directory to save recorded activations
        """
        self.model = model
        self.target_layers = target_layers
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.is_recording = False
        
    def _get_layer_names(self) -> List[str]:
        """Get list of layer names to record activations from."""
        if self.target_layers:
            return self.target_layers
        
        # Auto-detect transformer layers (common patterns)
        layer_names = []
        for name, module in self.model.named_modules():
            # Look for attention, MLP, or feed-forward layers
            if any(keyword in name.lower() for keyword in 
                   ['attn', 'attention', 'mlp', 'feed_forward', 'ffn', 'gate', 'up_proj', 'down_proj']):
                layer_names.append(name)
        
        return layer_names
    
    def _activation_hook(self, name: str) -> Callable:
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if self.is_recording:
                # Store activation statistics
                if isinstance(output, torch.Tensor):
                    # Detach and move to CPU to save memory
                    self.activations[name].append(output.detach().cpu())
                elif isinstance(output, tuple):
                    # Handle tuple outputs (e.g., attention outputs)
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            self.activations[f"{name}_output_{i}"].append(out.detach().cpu())
        return hook
    
    def register_hooks(self):
        """Register forward hooks on target layers."""
        layer_names = self._get_layer_names()
        
        for name, module in self.model.named_modules():
            if name in layer_names or any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(self._activation_hook(name))
                self.hooks.append(hook)
        
        print(f"Registered hooks on {len(self.hooks)} layers")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def start_recording(self):
        """Start recording activations."""
        self.is_recording = True
        self.activations.clear()
    
    def stop_recording(self):
        """Stop recording activations."""
        self.is_recording = False
    
    def record_forward_pass(
        self,
        inputs: torch.Tensor,
        layer_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Record activations from a single forward pass.
        
        Args:
            inputs: Input tensor (tokenized text)
            layer_name: Optional specific layer to record
        
        Returns:
            Dictionary of layer names to activation tensors
        """
        self.start_recording()
        
        with torch.no_grad():
            _ = self.model(inputs)
        
        self.stop_recording()
        
        # Convert lists to tensors
        recorded = {}
        for name, activation_list in self.activations.items():
            if activation_list:
                # Stack all activations from this layer
                recorded[name] = torch.cat(activation_list, dim=0)
        
        return recorded
    
    def save_activations(self, filename: str = "activations.pkl"):
        """Save recorded activations to disk."""
        if not self.save_dir:
            raise ValueError("save_dir not set. Cannot save activations.")
        
        filepath = self.save_dir / filename
        
        # Convert tensors to numpy for serialization
        activations_np = {}
        for name, activation_list in self.activations.items():
            if activation_list:
                # Stack and convert to numpy
                stacked = torch.cat(activation_list, dim=0)
                activations_np[name] = {
                    'data': stacked.numpy(),
                    'shape': list(stacked.shape),
                    'dtype': str(stacked.dtype)
                }
        
        with open(filepath, 'wb') as f:
            pickle.dump(activations_np, f)
        
        print(f"Saved activations to {filepath}")
        return filepath
    
    def load_activations(self, filename: str = "activations.pkl") -> Dict[str, np.ndarray]:
        """Load activations from disk."""
        if not self.save_dir:
            raise ValueError("save_dir not set. Cannot load activations.")
        
        filepath = self.save_dir / filename
        
        with open(filepath, 'rb') as f:
            activations_np = pickle.load(f)
        
        # Convert back to tensors
        activations = {}
        for name, data_dict in activations_np.items():
            activations[name] = torch.from_numpy(data_dict['data'])
        
        return activations


def load_model_for_activation_recording(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",  # Use smaller model for testing
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> nn.Module:
    """
    Load a model for activation recording.
    
    For Qwen3-32B, you'll need to adjust the model_name and potentially
    use model parallelism or quantization.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        torch_dtype: Data type for model weights
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
    
    Returns:
        Loaded model ready for activation recording
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with appropriate settings
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if device == "cuda" else None,
        "trust_remote_code": True
    }
    
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit
            )
            model_kwargs["quantization_config"] = quantization_config
        except ImportError:
            print("Warning: bitsandbytes not available. Skipping quantization.")
            print("Install with: pip install bitsandbytes (Linux/Windows only)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded on {device}")
    return model, tokenizer


def record_activations_from_dataset(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    batch_size: int = 1,
    max_length: int = 512,
    save_dir: Optional[str] = None
) -> ActivationRecorder:
    """
    Record activations from a dataset of texts.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer for the model
        texts: List of input texts
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        save_dir: Directory to save activations
    
    Returns:
        ActivationRecorder with recorded activations
    """
    recorder = ActivationRecorder(model, save_dir=save_dir)
    recorder.register_hooks()
    recorder.start_recording()
    
    print(f"Recording activations from {len(texts)} texts...")
    
    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0:
            print(f"Processing text {i+1}/{len(texts)}")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(next(model.parameters()).device)
        
        # Forward pass (activations recorded via hooks)
        with torch.no_grad():
            _ = model(**inputs)
    
    recorder.stop_recording()
    
    if save_dir:
        recorder.save_activations()
    
    return recorder

