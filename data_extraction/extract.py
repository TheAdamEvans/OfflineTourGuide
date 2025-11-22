"""
Data extraction pipeline for getting location data and Plus Codes from Qwen3-14B.

Two approaches:
1. Generate descriptions FROM Plus Codes (primary workflow)
2. Extract location data TO create Plus Codes (alternative workflow)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import openlocationcode as olc
import torch

# ============================================================================
# APPROACH 1: Generate descriptions FROM Plus Codes (Primary Workflow)
# ============================================================================

def generate_description_from_plus_code(
    plus_code: str,
    interests: List[str],
    style: str,
    model,  # Qwen model instance
    tokenizer,  # Qwen tokenizer instance
    device: str = "cuda"
) -> Dict:
    """
    Query Qwen3-14B to generate a tour guide description for a Plus Code.
    
    This is the main data generation approach from Phase 1.3.
    
    Args:
        plus_code: Plus Code string (e.g., "4RRH+Q8 Sydney")
        interests: List of interest tags (e.g., ["architecture", "history"])
        style: Style tag ("brief", "stimulate", "detailed")
        model: Qwen model instance
        tokenizer: Qwen tokenizer instance
        device: Device to run inference on
    
    Returns:
        Dictionary with plus_code, interests, style, and generated response
    """
    
    # Convert Plus Code back to approximate coordinates for context
    # (Plus Codes can be decoded to get lat/long)
    try:
        code_area = olc.decode(plus_code)
        lat = code_area.latitudeCenter
        lon = code_area.longitudeCenter
        location_context = f"approximately {lat:.4f}, {lon:.4f}"
    except:
        location_context = plus_code
    
    # Build prompt
    interests_str = ", ".join(interests)
    prompt = f"""You are a tour guide for Sydney, Australia. Generate a {style} tour guide description for the location with Plus Code: {plus_code} ({location_context}).

Focus on these interests: {interests_str}

Requirements:
- Be accurate and factual about this specific location
- Match the {style} style (brief = 2-3 sentences, stimulate = engaging paragraph, detailed = comprehensive description)
- Include specific details about points of interest, architecture, history, or cultural significance
- End naturally when complete (no filler text)

Tour guide description:"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set max tokens based on style
    max_new_tokens = 100 if style == "brief" else 300 if style == "stimulate" else 600
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return {
        "plus_code": plus_code,
        "interests": interests,
        "style": style,
        "response": generated_text.strip()
    }


# ============================================================================
# APPROACH 2: Extract location data TO create Plus Codes (Alternative)
# ============================================================================

def extract_poi_list_from_qwen(
    area: str,
    model,
    tokenizer,
    device: str = "cuda"
) -> List[Dict]:
    """
    Query Qwen3-14B to get a list of POIs in an area, then convert to Plus Codes.
    
    This is useful for discovering locations before generating descriptions.
    
    Args:
        area: Area name (e.g., "Sydney CBD", "Bondi Beach")
        model: Qwen model instance
        tokenizer: Qwen tokenizer instance
        device: Device to run inference on
    
    Returns:
        List of dictionaries with name, lat, lon, plus_code, description
    """
    
    prompt = f"""List 10-15 notable points of interest in {area}, Sydney, Australia.

For each POI, provide:
- Name
- Approximate latitude and longitude (as decimal degrees)
- Brief 1-sentence description
- Category (architecture, culture, nature, food, history, etc.)

Format as JSON array with keys: name, latitude, longitude, description, category"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Parse JSON (may need cleaning)
    try:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            pois = json.loads(json_match.group())
        else:
            # Fallback: try parsing entire response
            pois = json.loads(result)
    except:
        print(f"Failed to parse JSON. Raw response: {result}")
        return []
    
    # Convert to Plus Codes
    results = []
    for poi in pois:
        lat = poi.get('latitude')
        lon = poi.get('longitude')
        if lat and lon:
            plus_code = olc.encode(lat, lon, codeLength=10)
            results.append({
                "name": poi.get('name'),
                "latitude": lat,
                "longitude": lon,
                "plus_code": plus_code,
                "description": poi.get('description'),
                "category": poi.get('category')
            })
    
    return results


# ============================================================================
# Batch Generation Pipeline
# ============================================================================

def generate_training_dataset(
    plus_codes: List[str],
    interests_list: List[List[str]],
    styles: List[str],
    model,
    tokenizer,
    output_file: str = "training_data.jsonl",
    text_output_dir: Optional[str] = None,
    device: str = "cuda"
) -> None:
    """
    Generate training dataset by querying Qwen3-14B for each Plus Code combination.
    
    This implements Phase 1.3 of the plan.
    
    Args:
        plus_codes: List of Plus Code strings
        interests_list: List of interest tag combinations (e.g., [["architecture"], ["history", "culture"]])
        styles: List of style tags
        model: Qwen model instance
        tokenizer: Qwen tokenizer instance
        output_file: Output JSONL file path
        text_output_dir: Optional directory to save individual text files
        device: Device to run inference on
    """
    
    # Create text output directory if specified
    if text_output_dir:
        Path(text_output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for idx, plus_code in enumerate(plus_codes):
            for interests in interests_list:
                for style in styles:
                    print(f"Generating: {plus_code} | {interests} | {style}")
                    
                    try:
                        result = generate_description_from_plus_code(
                            plus_code=plus_code,
                            interests=interests,
                            style=style,
                            model=model,
                            tokenizer=tokenizer,
                            device=device
                        )
                        
                        # Write to JSONL
                        f.write(json.dumps(result) + '\n')
                        f.flush()  # Save incrementally
                        
                        # Write to text file if directory specified
                        if text_output_dir:
                            # Create a safe filename from the plus code and parameters
                            safe_plus_code = plus_code.replace('+', '_').replace('/', '_')
                            interests_str = '_'.join(interests)
                            filename = f"{idx:04d}_{safe_plus_code}_{interests_str}_{style}.txt"
                            text_file_path = Path(text_output_dir) / filename
                            
                            with open(text_file_path, 'w', encoding='utf-8') as txt_file:
                                txt_file.write(f"Plus Code: {result['plus_code']}\n")
                                txt_file.write(f"Interests: {', '.join(result['interests'])}\n")
                                txt_file.write(f"Style: {result['style']}\n")
                                txt_file.write(f"\n{result['response']}\n")
                        
                    except Exception as e:
                        print(f"Error generating {plus_code}: {e}")
                        continue


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_qwen_model(model_name: str = "Qwen/Qwen3-14B", device: str = "cuda"):
    """
    Load Qwen3-14B model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer

