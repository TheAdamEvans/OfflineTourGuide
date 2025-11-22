"""
Main entry point for OfflineTourGuide data extraction.
"""

import openlocationcode as olc
from data_extraction import (
    load_qwen_model,
    generate_training_dataset,
    generate_description_from_plus_code
)


def main():
    print("OfflineTourGuide - Data Extraction with Qwen3-14B")
    print("=" * 60)
    
    # Example: Generate Plus Codes for Sydney locations
    sydney_locations = [
        (-33.8688, 151.2093, "Sydney Opera House"),
        (-33.8615, 151.2120, "Sydney Harbour Bridge"),
        (-33.8705, 151.2071, "Circular Quay"),
    ]
    
    print("\nGenerating Plus Codes for Sydney locations:")
    plus_codes = []
    for lat, lon, name in sydney_locations:
        plus_code = olc.encode(lat, lon, codeLength=10)
        plus_codes.append(plus_code)
        print(f"{name}: {plus_code}")
    
    print("\n" + "="*60)
    print("To use with Qwen3-14B:")
    print("="*60)
    print("""
    # Load model
    model, tokenizer = load_qwen_model(device="cuda")
    
    # Generate description:
    result = generate_description_from_plus_code(
        plus_code="4RRH+Q8 Sydney",
        interests=["architecture", "history"],
        style="detailed",
        model=model,
        tokenizer=tokenizer,
        device="cuda"
    )
    
    # Generate training dataset with text file output:
    generate_training_dataset(
        plus_codes=plus_codes,
        interests_list=[["architecture"], ["history", "culture"]],
        styles=["brief", "stimulate", "detailed"],
        model=model,
        tokenizer=tokenizer,
        output_file="training_data.jsonl",
        text_output_dir="output_texts",  # Saves individual .txt files
        device="cuda"
    )
    """)


if __name__ == "__main__":
    main()
