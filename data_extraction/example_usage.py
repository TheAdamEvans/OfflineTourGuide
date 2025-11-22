"""
Example usage of the data extraction pipeline with Qwen3-14B.
"""

import openlocationcode as olc
from extract import (
    load_qwen_model,
    generate_training_dataset,
    generate_description_from_plus_code
)


def example_single_generation():
    """Example: Generate a single description."""
    print("Loading Qwen3-14B model...")
    model, tokenizer = load_qwen_model(device="cuda")
    
    # Generate description for a specific Plus Code
    result = generate_description_from_plus_code(
        plus_code="4RRH+Q8 Sydney",
        interests=["architecture", "history"],
        style="detailed",
        model=model,
        tokenizer=tokenizer,
        device="cuda"
    )
    
    print("\nGenerated Description:")
    print(f"Plus Code: {result['plus_code']}")
    print(f"Interests: {result['interests']}")
    print(f"Style: {result['style']}")
    print(f"\n{result['response']}")


def example_batch_generation():
    """Example: Generate training dataset with text file output."""
    print("Loading Qwen3-14B model...")
    model, tokenizer = load_qwen_model(device="cuda")
    
    # Generate Plus Codes for Sydney locations
    sydney_locations = [
        (-33.8688, 151.2093, "Sydney Opera House"),
        (-33.8615, 151.2120, "Sydney Harbour Bridge"),
        (-33.8705, 151.2071, "Circular Quay"),
    ]
    
    plus_codes = [olc.encode(lat, lon, codeLength=10) for lat, lon, _ in sydney_locations]
    
    # Generate training dataset
    generate_training_dataset(
        plus_codes=plus_codes,
        interests_list=[["architecture"], ["history", "culture"], ["nature"]],
        styles=["brief", "stimulate", "detailed"],
        model=model,
        tokenizer=tokenizer,
        output_file="training_data.jsonl",
        text_output_dir="output_texts",  # Saves individual .txt files
        device="cuda"
    )
    
    print("\nGeneration complete!")
    print("Check 'training_data.jsonl' for JSONL output")
    print("Check 'output_texts/' for individual text files")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        example_batch_generation()
    else:
        example_single_generation()

