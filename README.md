# Yamaha Aerox RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot specialized in answering questions about the Yamaha Aerox motorcycle. The chatbot uses Ollama with the Gemma model and provides accurate answers based on a curated knowledge base.

## Features

- 🏍️ Specialized knowledge about Yamaha Aerox motorcycles
- 🔍 RAG-based retrieval for accurate, source-backed answers
- 🤖 Powered by Ollama and Gemma language model
- 🌐 Web interface for easy interaction
- 🔌 OpenAI-compatible API endpoints
- 📚 Comprehensive knowledge base covering specs, maintenance, and tips

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running ([Download Ollama](https://ollama.ai))
3. **Gemma model** (installation instructions below)

## Installation

### 1. Install Python Dependencies

```bash
# Activate your virtual environment (if you have one)
# On Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Install Ollama Model

The Gemma model is currently downloading. If the download was interrupted, run:

```bash
ollama pull gemma
```

To check if the model is installed:

```bash
ollama list
```

You should see `gemma` in the list of models.

### 3. Verify Ollama is Running

Make sure Ollama is running on your system:

```bash
# On Windows, Ollama usually runs as a service
# You can check by running:
ollama list
```

If Ollama is not running, start it with:

```bash
ollama serve
```

## Running the Chatbot

### Start the Server

```bash
python server.py
```

You should see output like:
```
Loading knowledge base from knowledge_base...
Generating embeddings...
Indexed X chunks from Y documents
Checking Ollama for model gemma...
Model gemma found!
```

### Access the Chatbot

Open your web browser and navigate to:

```
http://127.0.0.1:5005
```

You'll see a friendly chat interface where you can ask questions about the Yamaha Aerox!

## Usage Examples

Try asking questions like:

- "What are the engine specifications?"
- "How often should I change the oil?"
- "What is the fuel tank capacity?"
- "What are the key features of the Aerox?"
- "How do I maintain my Aerox in wet weather?"
- "What's the recommended tire pressure?"
- "Tell me about the VVA system"

## Project Structure

```
my-chatbot/
├── server.py              # FastAPI server
├── config.py              # Configuration settings
├── retrieval.py           # RAG retrieval system
├── ollama_client.py       # Ollama API client
├── prompts.py             # Prompt templates
├── requirements.txt       # Python dependencies
├── knowledge_base/        # Knowledge base documents
│   ├── overview.txt
│   ├── specifications.txt
│   ├── maintenance.txt
│   └── features_and_tips.txt
└── static/
    └── index.html         # Web interface
```

## API Endpoints

### Standard Endpoint

**POST /generate**
```json
{
  "prompt": "What is the engine displacement?"
}
```

Response:
```json
{
  "response": "The Yamaha Aerox 155 has an engine displacement of 155cc...",
  "sources": ["specifications.txt"]
}
```

### OpenAI-Compatible Endpoints

The chatbot also provides OpenAI-compatible endpoints:

**POST /v1/chat/completions**
```json
{
  "model": "dota2-rag",
  "messages": [
    {"role": "user", "content": "What is the fuel capacity?"}
  ]
}
```

**POST /v1/completions**
```json
{
  "model": "dota2-rag",
  "prompt": "What is the fuel capacity?"
}
```

**GET /v1/models**
Lists available models.

## Configuration

Edit `config.py` to customize:

- `MODEL`: Ollama model to use (default: "gemma:3b")
- `KB_DIR`: Knowledge base directory
- `TOP_K`: Number of chunks to retrieve (default: 3)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.3)
- `HOST`: Server host (default: "127.0.0.1")
- `PORT`: Server port (default: 5005)

## Customizing the Knowledge Base

To add more information:

1. Create a new `.txt` file in the `knowledge_base/` directory
2. Add your content (plain text)
3. Restart the server

The system will automatically:
- Load all `.txt` files from the knowledge base
- Chunk the documents
- Generate embeddings
- Make the content searchable

## Troubleshooting

### "Cannot connect to Ollama"

- Make sure Ollama is running: `ollama serve`
- Check if Ollama is accessible: `ollama list`

### "Model not found"

- Pull the model: `ollama pull gemma`
- Verify installation: `ollama list`

### "No documents found"

- Ensure the `knowledge_base/` directory exists
- Check that it contains `.txt` files
- Verify file permissions

### Server Won't Start

- Check if port 5005 is already in use
- Try changing the port in `config.py`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Performance Notes

- First startup takes longer (generating embeddings)
- Response time: 2-10 seconds depending on:
  - Query complexity
  - Model size
  - Hardware capabilities
- The Gemma model requires ~4-8 GB of RAM

## Future Enhancements

Potential improvements:
- Add more knowledge base content
- Implement conversation history
- Add file upload for user documentation
- Create mobile-responsive UI improvements
- Add voice input/output
- Support for images and diagrams

## License

This project is for educational and personal use.

## Credits

- FastAPI for the web framework
- Ollama for local LLM inference
- Sentence Transformers for embeddings
- Yamaha Motor Company for the Aerox motorcycle

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all prerequisites are met
3. Check server logs for error messages

Happy riding! 🏍️
