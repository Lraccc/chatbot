"""Prompt templates for the RAG chatbot"""
from typing import List, Dict


REFUSAL_MESSAGE = """I apologize, but I don't have enough information in my knowledge base to answer that question about the Yamaha Aerox. Please ask me something related to the Yamaha Aerox motorcycle, such as specifications, maintenance, features, or riding tips."""


def build_system_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build a system prompt with retrieved context
    """
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
    """
    Extract unique source filenames from retrieved chunks
    """
    sources = []
    seen = set()
    
    for chunk in retrieved_chunks:
        source = chunk.get('source', '')
        if source and source not in seen:
            sources.append(source)
            seen.add(source)
    
    return sources
