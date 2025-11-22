"""
Activation recording module for capturing model activations during forward passes.

This module provides hooks to record activations from transformer models,
which are then used for activation-based pruning analysis.
"""

import gc
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import numpy as np
from pathlib import Path
import pickle


DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "samples"


def _sanitize_sample_id(sample_id: str) -> str:
    """Create a filesystem-safe sample identifier."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-+_")
    sanitized = "".join(ch if ch in allowed else "_" for ch in sample_id)
    sanitized = sanitized.strip("_")
    return sanitized or "sample"


def _load_sample_texts(sample_dir: Optional[str]) -> List[Dict[str, Optional[str]]]:
    """
    Load real tour stop samples from disk.
    
    Args:
        sample_dir: Directory containing .txt samples. Defaults to repo /samples.
    
    Returns:
        List of dicts with id, path, and text fields.
    """
    directory = Path(sample_dir or DEFAULT_SAMPLE_DIR)
    if not directory.exists():
        raise FileNotFoundError(f"Sample directory {directory} does not exist.")
    
    samples: List[Dict[str, Optional[str]]] = []
    for sample_file in sorted(directory.glob("*.txt")):
        text = sample_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        samples.append({
            "id": sample_file.stem,
            "path": str(sample_file),
            "text": text
        })
    
    if not samples:
        raise ValueError(f"No .txt samples found in {directory}")
    
    print(f"Loaded {len(samples)} tour stop samples from {directory}")
    return samples


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
        self.input_tokens: List[torch.Tensor] = []
        self.output_tokens: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.is_recording = False
        
    def _get_layer_names(self) -> List[str]:
        """Get list of layer names to record activations from."""
        if self.target_layers:
            return self.target_layers
        
        # Auto-detect transformer layers (common patterns)
        layer_names = []
        
        # First, try to find Qwen-specific layer structure (model.layers.X)
        # Qwen models typically have: model.layers.0, model.layers.1, ..., model.layers.63
        qwen_layers = []
        for name, module in self.model.named_modules():
            # Qwen layer pattern: model.layers.N (where N is layer index)
            if 'model.layers.' in name and name.count('.') >= 2:
                parts = name.split('.')
                # Check if it's a layer number (e.g., "model.layers.0")
                try:
                    int(parts[2])  # Validate it's a number
                    layer_base = '.'.join(parts[:3])  # e.g., "model.layers.0"
                    if layer_base not in qwen_layers:
                        qwen_layers.append(layer_base)
                except (ValueError, IndexError):
                    pass
        
        if qwen_layers:
            # For Qwen, we want to record post-attention and post-FFN outputs
            # Record from the layer modules themselves (which output post-attention/post-FFN)
            layer_names = sorted(qwen_layers)
            print(f"Detected {len(layer_names)} Qwen transformer layers")
        else:
            # Fallback: Look for attention, MLP, or feed-forward layers
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
                    # Store mean over batch dimension to handle variable lengths
                    # Shape: [batch, seq_len, hidden_dim] -> [seq_len, hidden_dim]
                    if len(output.shape) >= 2:
                        # Average over batch dimension if present
                        if output.shape[0] == 1:
                            # Single batch, remove batch dim
                            self.activations[name].append(output.squeeze(0).detach().cpu())
                        else:
                            # Multiple batches, average them
                            self.activations[name].append(output.mean(dim=0).detach().cpu())
                    else:
                        self.activations[name].append(output.detach().cpu())
                elif isinstance(output, tuple):
                    # Handle tuple outputs (e.g., attention outputs)
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            if len(out.shape) >= 2:
                                if out.shape[0] == 1:
                                    self.activations[f"{name}_output_{i}"].append(out.squeeze(0).detach().cpu())
                                else:
                                    self.activations[f"{name}_output_{i}"].append(out.mean(dim=0).detach().cpu())
                            else:
                                self.activations[f"{name}_output_{i}"].append(out.detach().cpu())
        return hook
    
    def register_hooks(self):
        """Register forward hooks on target layers."""
        layer_names = self._get_layer_names()
        
        # For Qwen models, we want to hook the layer modules directly
        # to capture post-attention and post-FFN outputs
        hooked_modules = set()
        
        for name, module in self.model.named_modules():
            # Check if this is a target layer
            should_hook = False
            
            if name in layer_names:
                should_hook = True
            elif any(layer_name in name for layer_name in layer_names):
                # For Qwen: hook the layer module itself (e.g., "model.layers.0")
                # This captures the full layer output (post-attention + post-FFN)
                if 'model.layers.' in name:
                    # Extract base layer name (e.g., "model.layers.0" from "model.layers.0.self_attn")
                    parts = name.split('.')
                    if len(parts) >= 3:
                        try:
                            int(parts[2])  # Validate it's a number
                            layer_base = '.'.join(parts[:3])
                            if layer_base in layer_names and layer_base not in hooked_modules:
                                should_hook = True
                                name = layer_base  # Use base layer name for consistency
                        except (ValueError, IndexError):
                            pass
            
            if should_hook and name not in hooked_modules:
                hook = module.register_forward_hook(self._activation_hook(name))
                self.hooks.append(hook)
                hooked_modules.add(name)
        
        print(f"Registered hooks on {len(self.hooks)} layers/modules")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def start_recording(self):
        """Start recording activations."""
        self.is_recording = True
        self.activations.clear()
        self.input_tokens.clear()
        self.output_tokens.clear()
    
    def stop_recording(self):
        """Stop recording activations."""
        self.is_recording = False
    
    def clear_recorded_data(self, clear_cuda_cache: bool = True):
        """Release recorded tensors to prevent memory growth."""
        self.activations.clear()
        self.input_tokens.clear()
        self.output_tokens.clear()
        gc.collect()
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def record_forward_pass(
        self,
        inputs: torch.Tensor,
        layer_name: Optional[str] = None,
        tokenizer = None
    ) -> Dict[str, torch.Tensor]:
        """
        Record activations from a single forward pass.
        
        Args:
            inputs: Input tensor (tokenized text) or dict with input_ids
            layer_name: Optional specific layer to record
            tokenizer: Optional tokenizer to decode tokens
        
        Returns:
            Dictionary of layer names to activation tensors
        """
        self.start_recording()
        
        # Store input tokens
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            if input_ids is not None:
                self.input_tokens.append(input_ids.detach().cpu() if isinstance(input_ids, torch.Tensor) else input_ids)
        elif isinstance(inputs, torch.Tensor):
            self.input_tokens.append(inputs.detach().cpu())
        
        with torch.no_grad():
            if isinstance(inputs, dict):
                outputs = self.model(**inputs)
            else:
                outputs = self.model(inputs)
        
        # Store output tokens (logits -> token IDs)
        if hasattr(outputs, 'logits'):
            output_ids = outputs.logits.argmax(dim=-1)
            self.output_tokens.append(output_ids.detach().cpu())
        elif isinstance(outputs, torch.Tensor):
            output_ids = outputs.argmax(dim=-1) if len(outputs.shape) > 1 else outputs
            self.output_tokens.append(output_ids.detach().cpu())
        
        self.stop_recording()
        
        # Convert lists to tensors
        recorded = {}
        for name, activation_list in self.activations.items():
            if activation_list:
                # For single forward pass, just return the first (and likely only) activation
                if len(activation_list) == 1:
                    recorded[name] = activation_list[0]
                elif len(activation_list) > 1:
                    # If multiple, stack them (they should be same size from same forward pass)
                    try:
                        recorded[name] = torch.stack(activation_list, dim=0)
                    except RuntimeError:
                        # If sizes don't match, just take the first one
                        recorded[name] = activation_list[0]
        
        return recorded
    
    def save_activations(
        self,
        filename: str = "activations.pkl",
        metadata: Optional[Dict[str, Optional[str]]] = None
    ):
        """
        Save recorded activations to disk in Phase 2 format: {input_tokens, layer_outputs[64], output_tokens}
        """
        if not self.save_dir:
            raise ValueError("save_dir not set. Cannot save activations.")
        
        filepath = self.save_dir / filename
        
        print(f"Processing and saving activations...")
        print(f"Found {len(self.activations)} layers with activations")
        
        # Convert tensors to numpy for serialization
        activations_np = {}
        
        # Store layer outputs (all 64 layers or whatever was recorded)
        layer_outputs = {}
        layer_names = list(self.activations.keys())
        for idx, name in enumerate(layer_names):
            activation_list = self.activations[name]
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"Processing layer {idx+1}/{len(layer_names)}: {name}")
            if activation_list:
                # Handle variable-length sequences by padding to max length
                # or computing statistics
                if len(activation_list) > 0:
                    # Get max sequence length
                    max_seq_len = max(act.shape[0] if len(act.shape) >= 1 else 1 
                                     for act in activation_list if act.numel() > 0)
                    
                    # Pad all activations to same length and stack
                    padded_list = []
                    for act in activation_list:
                        if len(act.shape) == 0:
                            continue
                        seq_len = act.shape[0]
                        if seq_len < max_seq_len:
                            # Pad with zeros
                            padding_size = max_seq_len - seq_len
                            if len(act.shape) == 1:
                                padding = torch.zeros(padding_size, dtype=act.dtype)
                                padded = torch.cat([act, padding], dim=0)
                            else:
                                padding_shape = (padding_size,) + act.shape[1:]
                                padding = torch.zeros(padding_shape, dtype=act.dtype)
                                padded = torch.cat([act, padding], dim=0)
                            padded_list.append(padded)
                        else:
                            padded_list.append(act)
                    
                    if padded_list:
                        # Stack all padded activations
                        stacked = torch.stack(padded_list, dim=0)
                        layer_outputs[name] = {
                            'data': stacked.numpy(),
                            'shape': list(stacked.shape),
                            'dtype': str(stacked.dtype),
                            'original_lengths': [act.shape[0] for act in activation_list if len(act.shape) >= 1]
                        }
        
        # Store input tokens
        input_tokens_data = None
        if self.input_tokens:
            # Stack input tokens if multiple
            if len(self.input_tokens) == 1:
                tok = self.input_tokens[0]
                input_tokens_data = tok.numpy() if isinstance(tok, torch.Tensor) else tok
            else:
                # Pad and stack multiple input sequences
                max_len = max(tok.shape[-1] if isinstance(tok, torch.Tensor) else len(tok) for tok in self.input_tokens)
                padded_inputs = []
                for tok in self.input_tokens:
                    if isinstance(tok, torch.Tensor):
                        if tok.shape[-1] < max_len:
                            padding = torch.zeros((*tok.shape[:-1], max_len - tok.shape[-1]), dtype=tok.dtype)
                            padded = torch.cat([tok, padding], dim=-1)
                        else:
                            padded = tok
                        padded_inputs.append(padded)
                    else:
                        padded = tok + [0] * (max_len - len(tok))
                        padded_inputs.append(torch.tensor(padded))
                input_tokens_data = torch.stack(padded_inputs, dim=0).numpy()
        
        # Store output tokens
        output_tokens_data = None
        if self.output_tokens:
            # Stack output tokens if multiple
            if len(self.output_tokens) == 1:
                tok = self.output_tokens[0]
                output_tokens_data = tok.numpy() if isinstance(tok, torch.Tensor) else tok
            else:
                # Pad and stack multiple output sequences
                max_len = max(tok.shape[-1] if isinstance(tok, torch.Tensor) else len(tok) for tok in self.output_tokens)
                padded_outputs = []
                for tok in self.output_tokens:
                    if isinstance(tok, torch.Tensor):
                        if tok.shape[-1] < max_len:
                            padding = torch.zeros((*tok.shape[:-1], max_len - tok.shape[-1]), dtype=tok.dtype)
                            padded = torch.cat([tok, padding], dim=-1)
                        else:
                            padded = tok
                        padded_outputs.append(padded)
                    else:
                        padded = tok + [0] * (max_len - len(tok))
                        padded_outputs.append(torch.tensor(padded))
                output_tokens_data = torch.stack(padded_outputs, dim=0).numpy()
        
        # Format according to Phase 2 spec: {input_tokens, layer_outputs[64], output_tokens}
        activations_np = {
            'input_tokens': input_tokens_data,
            'layer_outputs': layer_outputs,  # Dictionary of all layer outputs
            'output_tokens': output_tokens_data,
            'num_layers': len(layer_outputs),
            'layer_names': list(layer_outputs.keys()),
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(activations_np, f)
        
        print(f"Saved activations to {filepath} (format: input_tokens, layer_outputs[{len(layer_outputs)}], output_tokens)")
        return filepath
    
    def load_activations(self, filename: str = "activations.pkl") -> Dict[str, np.ndarray]:
        """Load activations from disk."""
        if not self.save_dir:
            raise ValueError("save_dir not set. Cannot load activations.")
        
        filepath = self.save_dir / filename
        
        with open(filepath, 'rb') as f:
            activations_np = pickle.load(f)
        
        layer_outputs = activations_np.get('layer_outputs', {})
        activations = {}
        for name, data_dict in layer_outputs.items():
            activations[name] = torch.from_numpy(data_dict['data'])
        
        return activations


def load_model_for_activation_recording(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",  # Use smaller model for testing
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    cache_dir: Optional[str] = None
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
        cache_dir: Directory to cache downloaded models. Defaults to /workspace/downloads
                   if it exists, otherwise uses HuggingFace default (~/.cache/huggingface)
    
    Returns:
        Loaded model ready for activation recording
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    
    # Determine cache directory
    if cache_dir is None:
        # Default to /workspace/downloads for RunPod/workspace environments
        default_cache = "/workspace/downloads"
        # Create directory if it doesn't exist
        try:
            os.makedirs(default_cache, exist_ok=True)
            cache_dir = default_cache
            print(f"Using cache directory: {cache_dir}")
        except (OSError, PermissionError):
            # Fall back to HuggingFace default if we can't create /workspace/downloads
            cache_dir = None
            print("Using HuggingFace default cache directory (~/.cache/huggingface)")
    
    print(f"Loading model: {model_name}")
    if cache_dir:
        print(f"Model files will be cached in: {cache_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # Load model with appropriate settings
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if device == "cuda" else None,
        "trust_remote_code": True,
        "cache_dir": cache_dir
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
    texts: Optional[List[str]] = None,
    batch_size: int = 1,
    max_length: int = 512,
    save_dir: Optional[str] = None,
    sample_dir: Optional[str] = None,
    pad_to_max_length: bool = True
) -> ActivationRecorder:
    """
    Record activations from each tour stop sample and persist them per file.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer for the model
        texts: Optional list of custom texts. If None, .txt files are loaded from sample_dir.
        batch_size: Retained for API compatibility (recording is always sequential/per-sample).
        max_length: Maximum sequence length for tokenization
        save_dir: Directory to save activation files (one file per sample)
        sample_dir: Directory containing .txt samples (defaults to repo /samples)
        pad_to_max_length: Whether to pad to max_length for consistent tensor shapes
    
    Returns:
        ActivationRecorder used for capturing hooks (hooks removed before return)
    """
    if save_dir is None:
        raise ValueError("save_dir must be provided to persist per-sample activations.")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if texts:
        dataset = [
            {"id": f"custom_{i:04d}", "path": None, "text": text.strip()}
            for i, text in enumerate(texts)
            if text and text.strip()
        ]
        print(f"Using {len(dataset)} custom text samples for activation recording.")
    else:
        dataset = _load_sample_texts(sample_dir)
    
    if not dataset:
        raise ValueError("No texts available for activation recording.")
    
    if batch_size != 1:
        print("Warning: batch_size > 1 is not supported for sequential sample recording. "
              "Proceeding with batch_size=1 behavior.")
    
    recorder = ActivationRecorder(model, save_dir=str(save_path))
    recorder.register_hooks()
    
    device = next(model.parameters()).device
    
    try:
        for idx, sample in enumerate(dataset, start=1):
            sample_text = sample["text"]
            print(f"[Activation] Processing sample {idx}/{len(dataset)}: {sample['id']}")
            
            encoded = tokenizer(
                sample_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length" if pad_to_max_length else False
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            recorder.record_forward_pass(encoded)
            
            metadata = {
                "sample_id": sample["id"],
                "source_file": sample["path"],
                "text_length": len(sample_text),
                "max_length": max_length
            }
            filename = f"{_sanitize_sample_id(sample['id'])}.pkl"
            recorder.save_activations(filename=filename, metadata=metadata)
            recorder.clear_recorded_data()
            print(f"[Activation] Saved layer outputs to {save_path / filename}")
    finally:
        recorder.stop_recording()
        recorder.remove_hooks()
    
    return recorder

