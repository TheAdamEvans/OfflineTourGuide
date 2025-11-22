"""
Main entry point for OfflineTourGuide data extraction.
"""

import openlocationcode.openlocationcode as olc

from data_extraction import (generate_description_from_plus_code,
                             generate_training_dataset)


def main():
    print("OfflineTourGuide - Data Extraction with Qwen3-32B FP8 using vLLM on RunPod")
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
    print("To use with Qwen3-32B FP8 using vLLM on RunPod:")
    print("="*60)
    print("""
    # Setup:
    # 1. Deploy a vLLM template on RunPod (serverless endpoint)
    # 2. Get your endpoint ID from RunPod dashboard
    # 3. Set these in your .env file:
    #    RUNPOD_API_KEY=your_api_key_here
    #    RUNPOD_ENDPOINT=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1
    #    VLLM_MODEL_NAME=qwen3-32b  # (optional, defaults to 'qwen3-32b')
    #
    # Note: You can also just provide the endpoint ID:
    #    RUNPOD_ENDPOINT=YOUR_ENDPOINT_ID
    
    # Generate description:
    result = generate_description_from_plus_code(
        plus_code="4RRH+Q8 Sydney",
        interests=["architecture", "history"],
        style="detailed"
    )
    
    # Generate training dataset with text file output:
    generate_training_dataset(
        plus_codes=plus_codes,
        interests_list=[["architecture"], ["history", "culture"]],
        styles=["brief", "stimulate", "detailed"],
        output_file="training_data.jsonl",
        text_output_dir="output_texts"  # Saves individual .txt files
    )
    """)


if __name__ == "__main__":
    main()
