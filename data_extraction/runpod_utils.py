"""
RunPod vLLM API utilities for connecting to vLLM templates on RunPod.

RunPod vLLM templates expose OpenAI-compatible API endpoints.
You don't execute code on the instance - you just send HTTP requests.
"""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv is required. Install it with: pip install python-dotenv"
    )

import requests

# Load environment variables
load_dotenv()


def get_runpod_api_key() -> str:
    """
    Get RunPod API key from environment variables.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not found in environment
    """
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        raise ValueError(
            "RUNPOD_API_KEY not found. Please set it in your .env file."
        )
    return api_key


def get_runpod_endpoint() -> str:
    """
    Get RunPod vLLM endpoint URL from environment variables.
    
    RunPod vLLM templates use OpenAI-compatible endpoints:
    Format: https://api.runpod.ai/v2/[ENDPOINT_ID]/openai/v1
    
    You can provide either:
    - Full URL: https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1
    - Just endpoint ID: YOUR_ENDPOINT_ID (we'll construct the full URL)
    
    Returns:
        Full endpoint URL string
        
    Raises:
        ValueError: If endpoint is not found in environment
    """
    endpoint = os.getenv('RUNPOD_ENDPOINT')
    if not endpoint:
        raise ValueError(
            "RUNPOD_ENDPOINT not found. Please set it in your .env file.\n"
            "Format: https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1\n"
            "Or just: YOUR_ENDPOINT_ID"
        )
    
    # If it's just an endpoint ID, construct the full URL
    if not endpoint.startswith('http'):
        endpoint = f"https://api.runpod.ai/v2/{endpoint}/openai/v1"
    
    return endpoint.rstrip('/')


def get_model_name() -> str:
    """
    Get model name from environment variables.
    
    This should match the MODEL_NAME you set in your RunPod vLLM template.
    
    Returns:
        Model name string (defaults to "qwen3-32b" if not set)
    """
    return os.getenv('VLLM_MODEL_NAME', 'qwen3-32b')


def call_vllm_api(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> str:
    """
    Call RunPod vLLM API (OpenAI-compatible) to get AI-generated responses.
    
    RunPod vLLM templates expose an OpenAI-compatible API endpoint.
    You don't execute code on the instance - you just send HTTP requests.
    
    Args:
        prompt: The prompt to send to the API
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 2000)
    
    Returns:
        Generated text response from the API
        
    Raises:
        ValueError: If API key or endpoint is not configured
        requests.RequestException: If API call fails
    """
    api_key = get_runpod_api_key()
    base_url = get_runpod_endpoint()
    model_name = get_model_name()
    
    # Use OpenAI-compatible chat completions endpoint
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # OpenAI-compatible payload format
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    
    result = response.json()
    
    # Extract response from OpenAI-compatible format
    if 'choices' in result and len(result['choices']) > 0:
        return result['choices'][0]['message']['content']
    
    # Fallback
    raise ValueError(f"Unexpected response format: {result}")

