"""
Activation analysis module for analyzing recorded activations to determine pruning targets.

This module computes various metrics on activations to identify which layers/neurons
are least important and can be pruned.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict


class ActivationAnalyzer:
    """
    Analyzes recorded activations to determine pruning targets.
    
    Computes metrics like:
    - Activation magnitude (L1, L2 norms)
    - Activation variance
    - Activation sparsity (percentage of near-zero activations)
    - Importance scores based on multiple metrics
    """
    
    def __init__(self, activations: Dict[str, torch.Tensor]):
        """
        Initialize analyzer with recorded activations.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
        """
        self.activations = activations
        self.metrics: Dict[str, Dict[str, float]] = {}
        
    def compute_magnitude_metrics(self, layer_name: str) -> Dict[str, float]:
        """
        Compute magnitude-based metrics for a layer.
        
        Args:
            layer_name: Name of the layer to analyze
        
        Returns:
            Dictionary with magnitude metrics
        """
        if layer_name not in self.activations:
            return {}
        
        activations = self.activations[layer_name]
        
        # Compute various norms
        l1_norm = torch.mean(torch.abs(activations)).item()
        l2_norm = torch.mean(activations ** 2).item() ** 0.5
        max_abs = torch.max(torch.abs(activations)).item()
        mean_abs = torch.mean(torch.abs(activations)).item()
        
        # Per-neuron statistics (assuming last dimension is neurons)
        if len(activations.shape) >= 2:
            # Average over batch and sequence dimensions
            neuron_activations = activations.view(-1, activations.shape[-1])
            neuron_l1 = torch.mean(torch.abs(neuron_activations), dim=0)
            neuron_l2 = torch.mean(neuron_activations ** 2, dim=0) ** 0.5
            
            neuron_mean_l1 = torch.mean(neuron_l1).item()
            neuron_mean_l2 = torch.mean(neuron_l2).item()
            neuron_std_l1 = torch.std(neuron_l1).item()
            neuron_std_l2 = torch.std(neuron_l2).item()
        else:
            neuron_mean_l1 = neuron_mean_l2 = neuron_std_l1 = neuron_std_l2 = 0.0
        
        return {
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'max_abs': max_abs,
            'mean_abs': mean_abs,
            'neuron_mean_l1': neuron_mean_l1,
            'neuron_mean_l2': neuron_mean_l2,
            'neuron_std_l1': neuron_std_l1,
            'neuron_std_l2': neuron_std_l2
        }
    
    def compute_variance_metrics(self, layer_name: str) -> Dict[str, float]:
        """
        Compute variance-based metrics for a layer.
        
        Higher variance indicates more important activations.
        
        Args:
            layer_name: Name of the layer to analyze
        
        Returns:
            Dictionary with variance metrics
        """
        if layer_name not in self.activations:
            return {}
        
        activations = self.activations[layer_name]
        
        # Overall variance
        variance = torch.var(activations).item()
        std = torch.std(activations).item()
        
        # Per-neuron variance
        if len(activations.shape) >= 2:
            neuron_activations = activations.view(-1, activations.shape[-1])
            neuron_variance = torch.var(neuron_activations, dim=0)
            
            mean_neuron_variance = torch.mean(neuron_variance).item()
            std_neuron_variance = torch.std(neuron_variance).item()
            min_neuron_variance = torch.min(neuron_variance).item()
            max_neuron_variance = torch.max(neuron_variance).item()
        else:
            mean_neuron_variance = std_neuron_variance = 0.0
            min_neuron_variance = max_neuron_variance = 0.0
        
        return {
            'variance': variance,
            'std': std,
            'mean_neuron_variance': mean_neuron_variance,
            'std_neuron_variance': std_neuron_variance,
            'min_neuron_variance': min_neuron_variance,
            'max_neuron_variance': max_neuron_variance
        }
    
    def compute_sparsity_metrics(self, layer_name: str, threshold: float = 1e-6) -> Dict[str, float]:
        """
        Compute sparsity metrics for a layer.
        
        Sparsity indicates how many activations are near-zero.
        Higher sparsity = more neurons can potentially be pruned.
        
        Args:
            layer_name: Name of the layer to analyze
            threshold: Threshold below which activation is considered "zero"
        
        Returns:
            Dictionary with sparsity metrics
        """
        if layer_name not in self.activations:
            return {}
        
        activations = self.activations[layer_name]
        
        # Overall sparsity
        abs_activations = torch.abs(activations)
        sparsity = (abs_activations < threshold).float().mean().item()
        
        # Per-neuron sparsity
        if len(activations.shape) >= 2:
            neuron_activations = activations.view(-1, activations.shape[-1])
            neuron_abs = torch.abs(neuron_activations)
            neuron_sparsity = (neuron_abs < threshold).float().mean(dim=0)
            
            mean_neuron_sparsity = torch.mean(neuron_sparsity).item()
            std_neuron_sparsity = torch.std(neuron_sparsity).item()
            
            # Count of neurons with high sparsity (>90%)
            high_sparsity_count = (neuron_sparsity > 0.9).sum().item()
            total_neurons = neuron_sparsity.shape[0]
            high_sparsity_ratio = high_sparsity_count / total_neurons if total_neurons > 0 else 0.0
        else:
            mean_neuron_sparsity = std_neuron_sparsity = 0.0
            high_sparsity_count = high_sparsity_ratio = 0.0
        
        return {
            'sparsity': sparsity,
            'mean_neuron_sparsity': mean_neuron_sparsity,
            'std_neuron_sparsity': std_neuron_sparsity,
            'high_sparsity_count': high_sparsity_count,
            'high_sparsity_ratio': high_sparsity_ratio
        }
    
    def compute_importance_scores(
        self,
        layer_name: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute composite importance scores for a layer.
        
        Combines multiple metrics to determine overall importance.
        Lower scores indicate layers/neurons that can be pruned.
        
        Args:
            layer_name: Name of the layer to analyze
            weights: Optional weights for different metrics
        
        Returns:
            Dictionary with importance scores
        """
        if weights is None:
            weights = {
                'magnitude': 0.3,
                'variance': 0.4,
                'sparsity': 0.3  # Higher sparsity = lower importance
            }
        
        magnitude_metrics = self.compute_magnitude_metrics(layer_name)
        variance_metrics = self.compute_variance_metrics(layer_name)
        sparsity_metrics = self.compute_sparsity_metrics(layer_name)
        
        if not all([magnitude_metrics, variance_metrics, sparsity_metrics]):
            return {}
        
        # Normalize metrics to [0, 1] range (relative to all layers)
        # For now, use raw values - normalization should be done across all layers
        magnitude_score = magnitude_metrics.get('l2_norm', 0.0)
        variance_score = variance_metrics.get('variance', 0.0)
        sparsity_score = 1.0 - sparsity_metrics.get('sparsity', 0.0)  # Invert sparsity
        
        # Composite importance score
        importance = (
            weights['magnitude'] * magnitude_score +
            weights['variance'] * variance_score +
            weights['sparsity'] * sparsity_score
        )
        
        return {
            'importance': importance,
            'magnitude_score': magnitude_score,
            'variance_score': variance_score,
            'sparsity_score': sparsity_score
        }
    
    def analyze_all_layers(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze all layers and compute all metrics.
        
        Returns:
            Dictionary mapping layer names to their metrics
        """
        results = {}
        
        for layer_name in self.activations.keys():
            metrics = {
                **self.compute_magnitude_metrics(layer_name),
                **self.compute_variance_metrics(layer_name),
                **self.compute_sparsity_metrics(layer_name),
                **self.compute_importance_scores(layer_name)
            }
            results[layer_name] = metrics
        
        self.metrics = results
        return results
    
    def get_pruning_candidates(
        self,
        pruning_ratio: float = 0.3,
        strategy: str = "importance"
    ) -> List[Tuple[str, float]]:
        """
        Get list of layers/neurons to prune based on analysis.
        
        Args:
            pruning_ratio: Fraction of layers/neurons to prune (0.0 to 1.0)
            strategy: Strategy for selecting candidates ("importance", "sparsity", "variance")
        
        Returns:
            List of (layer_name, score) tuples, sorted by pruning priority
        """
        if not self.metrics:
            self.analyze_all_layers()
        
        candidates = []
        
        for layer_name, metrics in self.metrics.items():
            if strategy == "importance":
                score = metrics.get('importance', 0.0)
            elif strategy == "sparsity":
                score = metrics.get('sparsity', 0.0)  # Higher sparsity = better to prune
            elif strategy == "variance":
                score = -metrics.get('variance', 0.0)  # Lower variance = better to prune
            else:
                score = metrics.get('importance', 0.0)
            
            candidates.append((layer_name, score))
        
        # Sort by score (ascending for importance/variance, descending for sparsity)
        if strategy == "sparsity":
            candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            candidates.sort(key=lambda x: x[1])
        
        # Return top N candidates based on pruning ratio
        num_to_prune = int(len(candidates) * pruning_ratio)
        return candidates[:num_to_prune]
    
    def save_analysis(self, filepath: str):
        """Save analysis results to JSON file."""
        # Convert tensors to lists for JSON serialization
        serializable_metrics = {}
        for layer_name, metrics in self.metrics.items():
            serializable_metrics[layer_name] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"Saved analysis to {filepath}")


def analyze_activations_from_file(
    activations_file: str,
    output_file: Optional[str] = None
) -> ActivationAnalyzer:
    """
    Load activations from file and analyze them.
    
    Args:
        activations_file: Path to saved activations (pickle file)
        output_file: Optional path to save analysis results
    
    Returns:
        ActivationAnalyzer with computed metrics
    """
    import pickle
    
    # Load activations
    with open(activations_file, 'rb') as f:
        activations_data = pickle.load(f)
    
    # Convert back to tensors
    activations = {}
    for name, data_dict in activations_data.items():
        activations[name] = torch.from_numpy(data_dict['data'])
    
    # Analyze
    analyzer = ActivationAnalyzer(activations)
    analyzer.analyze_all_layers()
    
    if output_file:
        analyzer.save_analysis(output_file)
    
    return analyzer

