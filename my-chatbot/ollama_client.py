"""Client for interacting with Ollama API"""
import requests
import config


OLLAMA_BASE_URL = "http://localhost:11434"


def check_model_exists() -> tuple[bool, str]:
    """
    Check if the configured model exists in Ollama
    Returns: (exists: bool, result: str)
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "connection_error"
        
        data = response.json()
        models = data.get("models", [])
        
        # Check if our model exists
        for model in models:
            model_name = model.get("name", "")
            # Match both "gemma:3b" and "gemma:latest" formats
            if model_name.startswith(config.MODEL.split(":")[0]):
                return True, model_name
        
        return False, "not_found"
    
    except requests.exceptions.RequestException:
        return False, "connection_error"


def generate(prompt: str, system: str = None) -> str:
    """
    Generate a response using Ollama
    """
    payload = {
        "model": config.MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    if system:
        payload["system"] = system
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            return f"Error: Ollama API returned status {response.status_code}"
        
        data = response.json()
        return data.get("response", "No response generated")
    
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
