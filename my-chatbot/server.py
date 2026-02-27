from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Union, List, Any, Dict, Tuple
from pathlib import Path
import json
import time
import uuid
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "tinyllama:latest"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3
HOST = "127.0.0.1"
PORT = 5005
OLLAMA_BASE_URL = "http://localhost:11434"
KB_DIR = Path(__file__).parent / "knowledge_base"

# ============================================================================
# OLLAMA CLIENT FUNCTIONS
# ============================================================================

def check_model_exists() -> Tuple[bool, str]:
    """Check if the configured model exists in Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "connection_error"
        
        data = response.json()
        models = data.get("models", [])
        
        for model in models:
            model_name = model.get("name", "")
            if model_name.startswith(MODEL.split(":")[0]):
                return True, model_name
        
        return False, "not_found"
    
    except requests.exceptions.RequestException:
        return False, "connection_error"


def generate_ollama(prompt: str, system: str = None) -> str:
    """Generate a response using Ollama"""
    payload = {
        "model": MODEL,
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

# ============================================================================
# PROMPT FUNCTIONS
# ============================================================================

REFUSAL_MESSAGE = """I apologize, but I don't have enough information in my knowledge base to answer that question about the Yamaha Aerox. Please ask me something related to the Yamaha Aerox motorcycle, such as specifications, maintenance, features, or riding tips."""


def build_system_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """Build a system prompt with retrieved context"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Source {i} - {chunk['source']}]\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    system_prompt = f"""You are a helpful assistant specializing in the Yamaha Aerox motorcycle. Answer the user's question based ONLY on the provided context below. 

If the context doesn't contain enough information to answer the question, politely say you don't have that information.

Be concise, accurate, and friendly. Focus on providing practical and useful information about the Yamaha Aerox.

CONTEXT:
{context}

Remember: Only use information from the context above. Do not make up information."""

    return system_prompt


def get_sources(retrieved_chunks: List[Dict]) -> List[str]:
    """Extract unique source filenames from retrieved chunks"""
    sources = []
    seen = set()
    
    for chunk in retrieved_chunks:
        source = chunk.get('source', '')
        if source and source not in seen:
            sources.append(source)
            seen.add(source)
    
    return sources

# ============================================================================
# RETRIEVAL SYSTEM
# ============================================================================

class RetrievalSystem:
    def __init__(self, documents: List[Dict[str, str]], chunks: List[Dict[str, str]]):
        self.documents = documents
        self.chunks = chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Generating embeddings...")
        chunk_texts = [c['text'] for c in chunks]
        self.embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        
    def query(self, query_text: str, top_k: int = None) -> List[Dict]:
        """Query the retrieval system and return top-k relevant chunks"""
        if top_k is None:
            top_k = TOP_K
            
        query_embedding = self.model.encode([query_text])[0]
        
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                **self.chunks[idx],
                'score': float(scores[idx])
            })
        
        return results


def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 500) -> List[Dict[str, str]]:
    """Split documents into chunks"""
    chunks = []
    
    for doc in documents:
        content = doc['content']
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'source': doc['filename']
                    })
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'source': doc['filename']
            })
    
    return chunks


def load_documents_from_files() -> List[Dict[str, str]]:
    """Load all text documents from the knowledge base directory"""
    documents = []
    
    if not KB_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KB_DIR}")
    
    for file_path in KB_DIR.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'filename': file_path.name,
                'content': content
            })
    
    if not documents:
        raise ValueError(f"No .txt files found in {KB_DIR}")
    
    return documents


def get_retrieval_system() -> RetrievalSystem:
    """Initialize and return the retrieval system"""
    documents = load_documents_from_files()
    chunks = chunk_documents(documents)
    return RetrievalSystem(documents, chunks)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

retrieval_system: RetrievalSystem | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval_system
    
    print(f"Loading knowledge base from {KB_DIR}...")
    retrieval_system = get_retrieval_system()
    print(f"Indexed {len(retrieval_system.chunks)} chunks from {len(retrieval_system.documents)} documents")
    
    print(f"Checking Ollama for model {MODEL}...")
    exists, result = check_model_exists()
    
    if exists:
        print(f"Model {result} found!")
    elif result == "connection_error":
        print("ERROR: Cannot connect to Ollama. Is Ollama running?")
        print("Start Ollama with: ollama serve")
        exit(1)
    else:
        print(f"ERROR: Model '{MODEL}' not found.")
        print(f"\nTo pull the model, run:")
        print(f"  ollama pull {MODEL}")
        print(f"\nTo see available models, run:")
        print("  ollama list")
        exit(1)
    
    yield
    
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

    if not retrieved or retrieved[0]["score"] < SIMILARITY_THRESHOLD:
        return REFUSAL_MESSAGE, []

    system_prompt = build_system_prompt(prompt_text, retrieved)
    response = generate_ollama(prompt_text, system=system_prompt)
    return response, get_sources(retrieved)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return HTMLResponse(content=HTML_CONTENT)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="Retrieval system not initialized")
    
    retrieved = retrieval_system.query(req.prompt)
    
    if not retrieved or retrieved[0]["score"] < SIMILARITY_THRESHOLD:
        return {
            "response": REFUSAL_MESSAGE,
            "sources": []
        }
    
    system_prompt = build_system_prompt(req.prompt, retrieved)
    response = generate_ollama(req.prompt, system=system_prompt)
    
    return {
        "response": response,
        "sources": get_sources(retrieved)
    }


@app.get("/v1")
async def v1_root():
    return {"message": "OpenAI Compatible API"}


@app.get("/v1/models")
async def v1_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "aerox-rag",
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
    model = req.model or "aerox-rag"

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
    model = req.model or "aerox-rag"

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
