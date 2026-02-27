"""Configuration for the Yamaha Aerox RAG chatbot"""
from pathlib import Path

# Model configuration
MODEL = "llama3.2:3b"

# Knowledge base directory
KB_DIR = Path(__file__).parent / "knowledge_base"

# Retrieval settings
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3

# Server settings
HOST = "127.0.0.1"
PORT = 5005
