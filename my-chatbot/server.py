from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import requests
import time
import uuid
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow requests from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# --------------------------------------------------
# KNOWLEDGE BASE - Topic: Yamaha Aerox Motorcycle
# --------------------------------------------------
KNOWLEDGE_BASE = """
You are a knowledgeable assistant specializing in the Yamaha Aerox motorcycle.
Here is what you know:
- The Yamaha Aerox is a sporty underbone/scooter-style motorcycle produced by Yamaha.
- Engine: 155cc liquid-cooled, 4-stroke, SOHC, with Variable Valve Actuation (VVA).
- It produces around 15 horsepower and 14.4 Nm of torque.
- Top speed is approximately 110–120 km/h.
- It features a CVT (Continuously Variable Transmission) — no manual clutch needed.
- Fuel system: Fuel injection (DiASil cylinder for better heat dissipation).
- The Aerox has a sporty MotoGP-inspired design with an aggressive front fairing.
- It comes with a large under-seat storage compartment.
- Features include: LED headlights, digital instrument cluster, and a USB charging port.
- Suspension: Telescopic front fork, unit swing rear suspension.
- Brakes: Disc brake (front and rear), with ABS available on some variants.
- Fuel tank capacity: approximately 5.5 liters.
- Dry weight: approximately 115 kg.
- Common variants include the Aerox S (standard) and Aerox Connected (with Bluetooth/app connectivity).
- Price in the Philippines ranges from approximately ₱115,000 to ₱130,000 depending on variant.
- Recommended for riders who want a fast, fuel-efficient, and stylish commuter bike.
- Maintenance intervals: engine oil change every 3,000 km or 3 months, whichever comes first.
- Compatible with Yamaha's Y-Connect app for ride data and notifications (Connected variant).

Only answer questions related to the Yamaha Aerox motorcycle. If asked something unrelated,
politely say you can only help with Aerox-related questions.
"""

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"  # Supports tool calls and works with OpenCode


class PromptRequest(BaseModel):
    prompt: str




@app.post("/generate")
async def generate(request: PromptRequest):
    full_prompt = f"{KNOWLEDGE_BASE}\n\nUser: {request.prompt}\nAssistant:"

    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()
        return {"response": result.get("response", "No response from model.")}
    except Exception as e:
        return {"error": str(e)}


# OpenAI-compatible endpoint so OpenCode can connect to this FastAPI server
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    
    messages = body.get("messages", [])
    requested_model = body.get("model")
    
    # Extract user prompt - simple approach
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            
            if isinstance(content, str):
                user_prompt = content
                break
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        # Skip system reminders
                        if not text.startswith("<system-reminder>"):
                            parts.append(text)
                user_prompt = " ".join(parts).strip()
                if user_prompt:
                    break
    
    if not user_prompt:
        user_prompt = "Hello! I can answer questions about the Yamaha Aerox motorcycle."
    
    # Simple prompt - no complex directives
    full_prompt = f"""{KNOWLEDGE_BASE}

User: {user_prompt}
Assistant:"""
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": 100,
                    "temperature": 0.3
                }
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()
        
        # Basic cleanup only
        answer = answer.replace("Answer:", "").replace("Response:", "").strip()
        
        if not answer:
            answer = "I can answer questions about the Yamaha Aerox motorcycle. What would you like to know?"
        
        # Return minimal OpenAI format
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model or MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }]
        }
        
    except Exception as e:
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model or MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sorry, there was an error processing your request."
                },
                "finish_reason": "stop"
            }]
        }


# Endpoint to list available models (required by OpenAI-compatible clients)
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    }


@app.get("/")
async def root():
    return {"message": "Chatbot server is running!"}
