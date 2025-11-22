"""
Data extraction pipeline for getting location data and Plus Codes from ChatGPT/Claude.

Two approaches:
1. Generate descriptions FROM Plus Codes (primary workflow)
2. Extract location data TO create Plus Codes (alternative workflow)
"""

import json
import openlocationcode as olc
from typing import List, Dict, Optional
import os


# ============================================================================
# APPROACH 1: Generate descriptions FROM Plus Codes (Primary Workflow)
# ============================================================================

def generate_description_from_plus_code(
    plus_code: str,
    interests: List[str],
    style: str,
    api_client,  # OpenAI or Anthropic client
    model: str = "gpt-4"
) -> Dict:
    """
    Query ChatGPT/Claude to generate a tour guide description for a Plus Code.
    
    This is the main data generation approach from Phase 1.3.
    
    Args:
        plus_code: Plus Code string (e.g., "4RRH+Q8 Sydney")
        interests: List of interest tags (e.g., ["architecture", "history"])
        style: Style tag ("brief", "stimulate", "detailed")
        api_client: OpenAI or Anthropic client instance
        model: Model name to use
    
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

    # Call API (example for OpenAI - adjust for Anthropic)
    if hasattr(api_client, 'chat'):  # OpenAI
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert tour guide for Sydney, Australia."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500 if style == "brief" else 1000 if style == "stimulate" else 2000
        )
        generated_text = response.choices[0].message.content
    else:  # Anthropic Claude
        response = api_client.messages.create(
            model=model,
            max_tokens=500 if style == "brief" else 1000 if style == "stimulate" else 2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        generated_text = response.content[0].text
    
    return {
        "plus_code": plus_code,
        "interests": interests,
        "style": style,
        "response": generated_text.strip()
    }


# ============================================================================
# APPROACH 2: Extract location data TO create Plus Codes (Alternative)
# ============================================================================

def extract_poi_list_from_chatgpt(
    area: str,
    api_client,
    model: str = "gpt-4"
) -> List[Dict]:
    """
    Query ChatGPT/Claude to get a list of POIs in an area, then convert to Plus Codes.
    
    This is useful for discovering locations before generating descriptions.
    
    Args:
        area: Area name (e.g., "Sydney CBD", "Bondi Beach")
        api_client: OpenAI or Anthropic client
        model: Model name
    
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

    # Call API
    if hasattr(api_client, 'chat'):  # OpenAI
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a location data expert. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} if "gpt-4" in model else None,
            temperature=0.3
        )
        result = response.choices[0].message.content
    else:  # Anthropic
        response = api_client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        result = response.content[0].text
    
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
    api_client,
    output_file: str = "training_data.jsonl",
    model: str = "gpt-4"
) -> None:
    """
    Generate training dataset by querying ChatGPT/Claude for each Plus Code combination.
    
    This implements Phase 1.3 of the plan.
    
    Args:
        plus_codes: List of Plus Code strings
        interests_list: List of interest tag combinations (e.g., [["architecture"], ["history", "culture"]])
        styles: List of style tags
        api_client: API client instance
        output_file: Output JSONL file path
        model: Model to use
    """
    
    with open(output_file, 'w') as f:
        for plus_code in plus_codes:
            for interests in interests_list:
                for style in styles:
                    print(f"Generating: {plus_code} | {interests} | {style}")
                    
                    try:
                        result = generate_description_from_plus_code(
                            plus_code=plus_code,
                            interests=interests,
                            style=style,
                            api_client=api_client,
                            model=model
                        )
                        f.write(json.dumps(result) + '\n')
                        f.flush()  # Save incrementally
                    except Exception as e:
                        print(f"Error generating {plus_code}: {e}")
                        continue


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Generate Plus Codes for Sydney locations
    sydney_locations = [
        (-33.8688, 151.2093, "Sydney Opera House"),
        (-33.8615, 151.2120, "Sydney Harbour Bridge"),
        (-33.8705, 151.2071, "Circular Quay"),
    ]
    
    print("Generating Plus Codes for Sydney locations:")
    for lat, lon, name in sydney_locations:
        plus_code = olc.encode(lat, lon, codeLength=10)
        print(f"{name}: {plus_code}")
    
    print("\n" + "="*60)
    print("To use with ChatGPT/Claude API:")
    print("="*60)
    print("""
    # For OpenAI:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # For Anthropic:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Generate description:
    result = generate_description_from_plus_code(
        plus_code="4RRH+Q8 Sydney",
        interests=["architecture", "history"],
        style="detailed",
        api_client=client
    )
    """)

