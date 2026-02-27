"""Retrieval system for RAG using sentence embeddings"""
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import config


class RetrievalSystem:
    def __init__(self, documents: List[Dict[str, str]], chunks: List[Dict[str, str]]):
        self.documents = documents
        self.chunks = chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for all chunks
        print("Generating embeddings...")
        chunk_texts = [c['text'] for c in chunks]
        self.embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        
    def query(self, query_text: str, top_k: int = None) -> List[Dict]:
        """
        Query the retrieval system and return top-k relevant chunks
        """
        if top_k is None:
            top_k = config.TOP_K
            
        # Generate query embedding
        query_embedding = self.model.encode([query_text])[0]
        
        # Calculate cosine similarity
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return chunks with scores
        results = []
        for idx in top_indices:
            results.append({
                **self.chunks[idx],
                'score': float(scores[idx])
            })
        
        return results


def load_documents(kb_dir: Path) -> List[Dict[str, str]]:
    """Load all text documents from the knowledge base directory"""
    documents = []
    
    if not kb_dir.exists():
        print(f"Warning: Knowledge base directory {kb_dir} does not exist")
        return documents
    
    for file_path in kb_dir.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'filename': file_path.name,
                'content': content
            })
    
    return documents


def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 500) -> List[Dict[str, str]]:
    """Split documents into chunks"""
    chunks = []
    
    for doc in documents:
        content = doc['content']
        # Simple chunking by splitting on double newlines and sentences
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


def get_retrieval_system() -> RetrievalSystem:
    """Initialize and return the retrieval system"""
    documents = load_documents(config.KB_DIR)
    
    if not documents:
        raise RuntimeError(f"No documents found in {config.KB_DIR}")
    
    chunks = chunk_documents(documents)
    
    return RetrievalSystem(documents, chunks)
