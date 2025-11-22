"""
Model pruning module for structured pruning based on activation analysis.

This module implements structured pruning (layer/neuron removal) based on
activation analysis results.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import copy
from pathlib import Path


class StructuredPruner:
    """
    Performs structured pruning on transformer models based on activation analysis.
    
    Supports:
    - Layer pruning (removing entire layers)
    - Neuron pruning (removing neurons from MLP/FFN layers)
    - Attention head pruning (removing attention heads)
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize pruner with a model.
        
        Args:
            model: PyTorch model to prune
        """
        self.model = model
        self.original_state = copy.deepcopy(model.state_dict())
        
    def prune_layers(
        self,
        layers_to_prune: List[str],
        keep_ratio: float = 0.7
    ) -> nn.Module:
        """
        Prune entire layers from the model.
        
        Args:
            layers_to_prune: List of layer names/indices to prune
            keep_ratio: Ratio of layers to keep (alternative to explicit list)
        
        Returns:
            Pruned model
        """
        # This is a simplified version - full implementation would require
        # model architecture knowledge to properly remove layers
        print(f"Pruning {len(layers_to_prune)} layers: {layers_to_prune}")
        
        # For transformer models, we typically prune by:
        # 1. Removing layers from the transformer block list
        # 2. Adjusting layer indices in forward pass
        
        # This is model-specific and would need to be adapted for Qwen architecture
        pruned_model = copy.deepcopy(self.model)
        
        # Example: If model has model.layers attribute (common in HuggingFace)
        if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'layers'):
            layers = pruned_model.model.layers
            num_layers = len(layers)
            
            if isinstance(layers_to_prune[0], int):
                # Prune by indices
                indices_to_keep = [i for i in range(num_layers) if i not in layers_to_prune]
            else:
                # Prune by keep_ratio
                num_to_keep = int(num_layers * keep_ratio)
                indices_to_keep = list(range(0, num_layers, num_layers // num_to_keep))[:num_to_keep]
            
            # Create new layer list
            new_layers = nn.ModuleList([layers[i] for i in sorted(indices_to_keep)])
            pruned_model.model.layers = new_layers
            
            print(f"Pruned from {num_layers} to {len(new_layers)} layers")
        
        return pruned_model
    
    def prune_neurons(
        self,
        layer_name: str,
        neuron_indices: List[int],
        in_features: Optional[int] = None,
        out_features: Optional[int] = None
    ) -> nn.Module:
        """
        Prune specific neurons from a layer.
        
        Args:
            layer_name: Name of the layer to prune (e.g., "mlp.gate_proj")
            neuron_indices: Indices of neurons to remove
            in_features: Input feature size (if None, inferred from weight)
            out_features: Output feature size (if None, inferred from weight)
        
        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(self.model)
        
        # Get the layer
        module = pruned_model
        for part in layer_name.split('.'):
            module = getattr(module, part)
        
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            raise ValueError(f"Layer {layer_name} is not a Linear or Conv2d layer")
        
        # Get weight and bias
        weight = module.weight.data
        bias = module.bias.data if module.bias is not None else None
        
        # Determine which dimension to prune
        if isinstance(module, nn.Linear):
            # For Linear layers, prune output neurons (first dimension)
            out_dim = 0
            in_dim = 1
        else:
            # For Conv2d, prune output channels
            out_dim = 0
            in_dim = 1
        
        # Create mask
        num_neurons = weight.shape[out_dim]
        mask = torch.ones(num_neurons, dtype=torch.bool)
        mask[neuron_indices] = False
        
        # Prune weights and create new module
        if isinstance(module, nn.Linear):
            pruned_weight = weight[mask, :]
            if bias is not None:
                pruned_bias = bias[mask]
            else:
                pruned_bias = None
            
            # Create new layer
            new_module = nn.Linear(
                weight.shape[in_dim],
                pruned_weight.shape[0],
                bias=pruned_bias is not None
            )
            new_module.weight.data = pruned_weight
            if pruned_bias is not None:
                new_module.bias.data = pruned_bias
        else:
            # For Conv2d or other layers, would need similar logic
            raise NotImplementedError(f"Pruning not implemented for {type(module)}")
        
        # Replace module
        parent = pruned_model
        parts = layer_name.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
        
        print(f"Pruned {len(neuron_indices)} neurons from {layer_name}")
        
        return pruned_model
    
    def prune_mlp_layers(
        self,
        pruning_plan: Dict[str, List[int]],
        keep_ratio: float = 0.7
    ) -> nn.Module:
        """
        Prune neurons from MLP/FFN layers based on activation analysis.
        
        Args:
            pruning_plan: Dictionary mapping layer names to neuron indices to prune
            keep_ratio: Alternative: keep this ratio of neurons per layer
        
        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(self.model)
        
        # Find all MLP layers
        mlp_layers = []
        for name, module in pruned_model.named_modules():
            if any(keyword in name.lower() for keyword in 
                   ['mlp', 'feed_forward', 'ffn', 'gate_proj', 'up_proj', 'down_proj']):
                mlp_layers.append(name)
        
        print(f"Found {len(mlp_layers)} MLP layers")
        
        # Prune each layer
        for layer_name in mlp_layers:
            if layer_name in pruning_plan:
                neuron_indices = pruning_plan[layer_name]
            else:
                # Use keep_ratio to determine which neurons to prune
                module = pruned_model
                for part in layer_name.split('.'):
                    module = getattr(module, part)
                
                if isinstance(module, nn.Linear):
                    num_neurons = module.out_features
                    num_to_keep = int(num_neurons * keep_ratio)
                    # Prune neurons with lowest activation (would need activation data)
                    # For now, prune evenly
                    step = num_neurons // (num_neurons - num_to_keep)
                    neuron_indices = list(range(0, num_neurons, step))[:num_neurons - num_to_keep]
                else:
                    continue
            
            pruned_model = self.prune_neurons(layer_name, neuron_indices)
        
        return pruned_model
    
    def get_model_size(self, model: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Calculate model size in parameters and MB.
        
        Args:
            model: Model to analyze (default: self.model)
        
        Returns:
            Dictionary with size metrics
        """
        if model is None:
            model = self.model
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate size in MB (assuming float32 = 4 bytes)
        size_mb = total_params * 4 / (1024 ** 2)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'size_mb': size_mb
        }
    
    def save_pruned_model(self, model: nn.Module, save_path: str):
        """Save pruned model to disk."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.config if hasattr(model, 'config') else None
        }, save_path)
        print(f"Saved pruned model to {save_path}")


def create_pruning_plan_from_analysis(
    analyzer,
    pruning_ratio: float = 0.3,
    strategy: str = "importance",
    target_layers: Optional[List[str]] = None
) -> Dict[str, List[int]]:
    """
    Create a pruning plan from activation analysis results.
    
    Args:
        analyzer: ActivationAnalyzer with computed metrics
        pruning_ratio: Fraction of neurons to prune per layer
        strategy: Pruning strategy ("importance", "sparsity", "variance")
        target_layers: Optional list of specific layers to prune
    
    Returns:
        Dictionary mapping layer names to neuron indices to prune
    """
    pruning_plan = {}
    
    # Get candidates from analyzer
    candidates = analyzer.get_pruning_candidates(pruning_ratio, strategy)
    
    # For each layer, determine which neurons to prune
    for layer_name, score in candidates:
        if target_layers and layer_name not in target_layers:
            continue
        
        # Get neuron-level statistics if available
        metrics = analyzer.metrics.get(layer_name, {})
        
        # For now, use a simple strategy: prune neurons with lowest variance
        # In practice, you'd want to analyze per-neuron activations
        # This would require storing per-neuron statistics in the analyzer
        
        # Placeholder: would need per-neuron activation data
        # For now, return empty plan - user needs to implement per-neuron analysis
        pass
    
    return pruning_plan
