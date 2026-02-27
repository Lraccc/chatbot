from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import json
import time
import uuid
import config
import retrieval
import ollama_client
import prompts
from typing import Union, List, Any

retrieval_system: retrieval.RetrievalSystem | None = None
STATIC_DIR = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval_system
    
    print(f"Loading knowledge base from {config.KB_DIR}...")
    retrieval_system = retrieval.get_retrieval_system()
    print(f"Indexed {len(retrieval_system.chunks)} chunks from {len(retrieval_system.documents)} documents")
    
    print(f"Checking Ollama for model {config.MODEL}...")
    exists, result = ollama_client.check_model_exists()
    
    if exists:
        print(f"Model {result} found!")
    elif result == "connection_error":
        print("ERROR: Cannot connect to Ollama. Is Ollama running?")
        print("Start Ollama with: ollama serve")
        exit(1)
    else:
        print(f"ERROR: Model '{config.MODEL}' not found.")
        print(f"\nTo pull the model, run:")
        print(f"  ollama pull {config.MODEL}")
        print(f"\nTo see available models, run:")
        print("  ollama ls")
        exit(1)
    
    yield
    
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

class GenerateRequest(BaseModel):
    prompt: str


class OpenAIChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any], None] = None


class OpenAIChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIChatMessage]
    stream: bool = False


class OpenAICompletionsRequest(BaseModel):
    model: str | None = None
    prompt: str
    stream: bool = False

@app.get("/health")
async def health():
    return {"status": "ok", "model": config.MODEL}

@app.get("/v1")
async def v1_root():
    return {"message": "OpenAI Compatible API"}

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="Retrieval system not initialized")
    
    retrieved = retrieval_system.query(req.prompt)
    
    if not retrieved or retrieved[0]["score"] < config.SIMILARITY_THRESHOLD:
        return {
            "response": prompts.REFUSAL_MESSAGE,
            "sources": []
        }
    
    system_prompt = prompts.build_system_prompt(req.prompt, retrieved)
    
    response = ollama_client.generate(req.prompt, system=system_prompt)
    
    return {
        "response": response,
        "sources": prompts.get_sources(retrieved)
    }


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return " ".join(texts)
    else:
        return ""


def _rag_answer(prompt_text: str) -> tuple[str, list[str]]:
    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="Retrieval system not initialized")

    retrieved = retrieval_system.query(prompt_text)

    if not retrieved or retrieved[0]["score"] < config.SIMILARITY_THRESHOLD:
        return prompts.REFUSAL_MESSAGE, []

    system_prompt = prompts.build_system_prompt(prompt_text, retrieved)
    response = ollama_client.generate(prompt_text, system=system_prompt)
    return response, prompts.get_sources(retrieved)


@app.get("/v1/models")
async def v1_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "dota2-rag",
                "object": "model",
                "created": created,
                "owned_by": "local"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: OpenAIChatCompletionsRequest):
    user_messages = [m for m in req.messages if m.role == "user" and extract_text(m.content or "").strip()]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    prompt_text = extract_text(user_messages[-1].content or "").strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="No user message provided")
    content, sources = _rag_answer(prompt_text)

    created = int(time.time())
    resp_id = f"chatcmpl-{uuid.uuid4().hex}"
    model = req.model or "dota2-rag"

    if req.stream:
        def event_stream():
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "sources": sources
    }


@app.post("/v1/completions")
async def v1_completions(req: OpenAICompletionsRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    prompt_text = req.prompt.strip()
    content, sources = _rag_answer(prompt_text)

    created = int(time.time())
    resp_id = f"cmpl-{uuid.uuid4().hex}"
    model = req.model or "dota2-rag"

    if req.stream:
        def event_stream():
            chunk = {
                "id": resp_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": content,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": resp_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": content,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "sources": sources
    }
