# Aviation RAG System

A production-ready Retrieval-Augmented Generation (RAG) system designed specifically for aviation professionals. This system combines FAISS-based dense vector search with OpenAI's GPT-4 to provide accurate, real-time answers to aviation-related questions.

## Features

- Dense vector search using FAISS for efficient document retrieval
- Integration with OpenAI's GPT-4 for high-quality responses
- Real-time question answering
- Aviation-specific knowledge base
- RESTful API interface
- Scalable architecture

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## Project Structure

```
.
├── main.py              # FastAPI application entry point
├── vector_store.py      # FAISS vector store implementation
├── data_processor.py    # Data processing and embedding generation
├── config.py           # Configuration settings
└── requirements.txt    # Project dependencies
```

## API Endpoints

- `POST /api/query`: Submit a question and receive an answer
- `POST /api/ingest`: Add new documents to the knowledge base

## Usage

1. Start the server
2. Send POST requests to `/api/query` with your aviation-related questions
3. Receive detailed, context-aware responses

## Security

- API key authentication required
- Rate limiting implemented
- Input validation and sanitization

## Performance

- Optimized vector search with FAISS
- Caching for frequently accessed information
- Efficient document chunking and embedding

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

MIT License 