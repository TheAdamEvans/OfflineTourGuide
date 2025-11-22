"""
Example script for running activation-based pruning.

This is a simplified example that demonstrates the pruning pipeline.
For production use, see the full pipeline in pruning_pipeline.py
"""

import torch
from pathlib import Path

# Import pruning modules
from model import (
    load_model_for_activation_recording,
    record_activations_from_dataset,
    analyze_activations_from_file,
    StructuredPruner
)


def main():
    """Run a simple pruning example."""
    
    print("=" * 60)
    print("Activation-Based Pruning Example")
    print("=" * 60)
    print("\nNOTE: This script is designed to run on RunPod with GPU and sufficient disk space.")
    print("For local testing, ensure you have at least 2GB free disk space.\n")
    
    # Configuration
    # Use a very small model for quick testing
    # For RunPod, you'll use: "Qwen/Qwen3-32B" or "Qwen/Qwen2.5-7B-Instruct"
    model_name = "Qwen/Qwen3-32B"  # Very small model for testing (~1GB)
    device = "cuda"
    output_dir = Path("pruning_example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Prepare activation texts
    # In practice, use diverse texts from your tour guide domain
    activation_texts = [
        "You are an engaging, extremely knowledgeable tour guide.",
        "Generate a detailed tour guide description for Sydney Opera House.",
        "This tour group is interested in architecture and history.",
        "Create a brief tour guide description for Bondi Beach.",
        "The tour group wants to learn about local culture and food.",
    ] * 20  # Repeat for more data points
    
    print(f"\n[Step 1] Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Step 2: Load model (with quantization to save memory)
    try:
        # On macOS, bitsandbytes is not available, so skip quantization
        # On RunPod with GPU, you can enable quantization to save memory
        print(f"Loading model (this may take a few minutes for first download)...")
        result = load_model_for_activation_recording(
            model_name=model_name,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            load_in_8bit=False,  # Set to True on RunPod with GPU to save memory
            load_in_4bit=False   # Set to True on RunPod for even more memory savings
        )
        model, tokenizer = result
        print(f"✓ Model loaded successfully")
    except Exception as e:
        error_msg = str(e)
        print(f"Error loading model: {error_msg}")
        
        if "No space left on device" in error_msg or "free disk space" in error_msg:
            print("\n⚠️  Insufficient disk space to download model.")
            print("\nThis script is designed to run on RunPod where you have:")
            print("  - GPU with sufficient VRAM")
            print("  - At least 50GB+ free disk space")
            print("  - Linux environment (for bitsandbytes quantization)")
            print("\nTo run on RunPod:")
            print("  1. Deploy a GPU pod with sufficient resources")
            print("  2. Install dependencies: pip install -e .")
            print("  3. Run: python -m model.example_pruning")
            print("\nFor local testing, free up disk space or use a pre-downloaded model.")
        else:
            print("\nNote: You may need to:")
            print("  1. Install bitsandbytes (Linux/Windows only): pip install bitsandbytes")
            print("  2. Use a smaller model for testing")
            print("  3. Ensure you have enough memory and disk space")
            print("  4. On macOS, quantization is not available - using FP16/FP32 instead")
        
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Record activations
    print(f"\n[Step 2] Recording activations from {len(activation_texts)} texts...")
    try:
        _recorder = record_activations_from_dataset(
            model=model,
            tokenizer=tokenizer,
            texts=activation_texts,
            batch_size=1,
            max_length=512,
            save_dir=str(output_dir / "activations")
        )
        
        activation_file = output_dir / "activations" / "activations.pkl"
        print(f"✓ Activations saved to {activation_file}")
    except Exception as e:
        print(f"Error recording activations: {e}")
        return
    
    # Step 4: Analyze activations
    print(f"\n[Step 3] Analyzing activations...")
    try:
        analyzer = analyze_activations_from_file(
            activations_file=str(activation_file),
            output_file=str(output_dir / "analysis.json")
        )
        
        # Get pruning candidates
        candidates = analyzer.get_pruning_candidates(
            pruning_ratio=0.3,
            strategy="importance"
        )
        
        print(f"✓ Found {len(candidates)} pruning candidates")
        print("\nTop 10 candidates for pruning:")
        for i, (layer_name, score) in enumerate(candidates[:10], 1):
            print(f"  {i}. {layer_name}: {score:.4f}")
    except Exception as e:
        print(f"Error analyzing activations: {e}")
        return
    
    # Step 5: Prune model
    print(f"\n[Step 4] Pruning model...")
    try:
        pruner = StructuredPruner(model)
        
        # Get model size before
        size_before = pruner.get_model_size(model)
        print(f"Before pruning: {size_before['total_parameters']:,} parameters "
              f"({size_before['size_mb']:.2f} MB)")
        
        # Prune layers (simplified - would need model-specific logic)
        # For now, just demonstrate the structure
        print("\nNote: Actual pruning requires model-specific implementation.")
        print("See PRUNING_GUIDE.md for details on adapting to your model architecture.")
        
        # Save analysis for later use
        print(f"\n✓ Analysis saved to {output_dir / 'analysis.json'}")
        print(f"✓ You can use this analysis to prune your model")
        
    except Exception as e:
        print(f"Error during pruning: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review analysis: {output_dir / 'analysis.json'}")
    print(f"2. Adapt pruning logic for your model architecture")
    print(f"3. Fine-tune the pruned model with your training data")
    print(f"\nSee PRUNING_GUIDE.md for detailed instructions.")


if __name__ == "__main__":
    main()

