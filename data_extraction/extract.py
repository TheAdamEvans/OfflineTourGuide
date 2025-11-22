"""
Data extraction pipeline for getting location data and Plus Codes from Qwen3-32B FP8 using vLLM on RunPod.

Two approaches:
1. Generate descriptions FROM Plus Codes (primary workflow)
2. Extract location data TO create Plus Codes (alternative workflow)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import openlocationcode.openlocationcode as olc

from .runpod_utils import call_vllm_api

# ============================================================================
# APPROACH 1: Generate descriptions FROM Plus Codes (Primary Workflow)
# ============================================================================


def generate_description_from_plus_code(
    plus_code: str,
    interests: List[str],
    style: str,
    temperature: float = 0.7,
    country: str = "Australia",
    language: str = "English"
) -> Dict:
    """
    Query Qwen3-32B FP8 on RunPod to generate a tour guide description for a Plus Code.
    
    This is the main data generation approach from Phase 1.3.
    
    Args:
        plus_code: Plus Code string (e.g., "4RRH+Q8 Sydney")
        interests: List of interest tags (e.g., ["architecture", "history"])
        style: Style tag ("brief", "stimulate", "detailed")
        temperature: Sampling temperature (default: 0.7)
        country: Country of origin for tour group (default: "Australia")
        language: Language for the tour (default: "English")
    
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
    prompt = f"""You are an enagaging, extremely knowledgable tour guide.

As a professional tour guide you:
- Include specific details about points of interest, architecture, history, and cultural significance
- Include natural and geographical and plant / animal factoids when applicable
- You are a steward of life and diversity on earth
- You understand the best ways to transit from one place to another, time and distance
- Makes daylight, temperature, and season expectations based on the timestamp + geocode from generated data
- Know the name of and address native people and know about their language and stories
- Have intimate and nuanced knowledge of historical and recent events
- Have a spirit of adventure!
- Know lots about food and traditional cooking and flavors
- Understand needs of families and large groups
- Offer natural next steps when your tour stop complete

This particular tour group is from {country} and you are giving the tour in {language}.

Generate a tour stop blurb for the location for:
{plus_code} ({location_context})
"""

    # Set max tokens based on style
    max_tokens = 200 if style == "brief" else 500 if style == "stimulate" else 1500
    
    # Call vLLM API on RunPod
    generated_text = call_vllm_api(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
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
    temperature: float = 0.3
) -> List[Dict]:
    """
    Query Qwen3-32B FP8 on RunPod to get a list of POIs in an area, then convert to Plus Codes.
    
    This is useful for discovering locations before generating descriptions.
    
    Args:
        area: Area name (e.g., "Sydney CBD", "Bondi Beach")
        temperature: Sampling temperature (default: 0.3)
    
    Returns:
        List of dictionaries with name, lat, lon, plus_code, description
    """
    
    prompt = f"""List 10-15 notable points of interest in {area}, Sydney, Australia.

For each POI, provide:
- Name
- Approximate latitude and longitude (as decimal degrees)
- Brief 1-sentence description
- Category (architecture, culture, nature, food, history, etc.)

Format as JSON array with keys: name, latitude, longitude, description, category

Return ONLY valid JSON, no additional text."""

    # Call vLLM API on RunPod
    result = call_vllm_api(
        prompt=prompt,
        temperature=temperature,
        max_tokens=2000
    )
    
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
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON. Raw response: {result}")
        print(f"Error: {e}")
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
    output_file: str = "training_data.jsonl",
    text_output_dir: Optional[str] = None,
    temperature: float = 0.7
) -> None:
    """
    Generate training dataset by querying Qwen3-32B FP8 on RunPod for each Plus Code combination.
    
    This implements Phase 1.3 of the plan.
    
    Args:
        plus_codes: List of Plus Code strings
        interests_list: List of interest tag combinations (e.g., [["architecture"], ["history", "culture"]])
        styles: List of style tags
        output_file: Output JSONL file path
        text_output_dir: Optional directory to save individual text files
        temperature: Sampling temperature (default: 0.7)
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
                            temperature=temperature
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



