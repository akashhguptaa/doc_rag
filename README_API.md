# Financial Data RAG System - API Documentation

## Overview

This is a production-ready RAG (Retrieval-Augmented Generation) system for financial data analysis with WebSocket support for real-time conversations.

## Architecture

### 1. **vector_search.py** - Vector Search Engine

- FAISS-based semantic search
- Cross-encoder reranking for improved accuracy
- Handles both holdings and trades data
- Optimized context formatting for LLM

### 2. **rag_agent.py** - LangGraph RAG Agent

- Query classification (RAG vs general queries)
- Conversation history management
- Conditional routing based on query type
- Google Gemini integration

### 3. **main.py** - FastAPI WebSocket Server

- Real-time WebSocket connections
- REST API endpoints
- Conversation thread management
- Health checks and monitoring

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8000
HOST=0.0.0.0
```

### 3. Ensure Data Files Exist

Make sure these files are in the project directory:

- `financial_data.index` (FAISS index)
- `financial_data_metadata.pkl` (metadata)

### 4. Start the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### REST API

#### Health Check

```
GET /health
```

Returns service health status

#### Stats

```
GET /stats
```

Returns system statistics (vector count, active connections, etc.)

#### Query (REST)

```
POST /query
Content-Type: application/json

{
  "query": "How many IBM trades were executed?",
  "thread_id": "user123"
}
```

Response:

```json
{
  "response": "Based on the data...",
  "requires_rag": true,
  "timestamp": "2026-01-29T10:30:00",
  "thread_id": "user123"
}
```

#### Get Conversation History

```
GET /conversation/{thread_id}
```

#### Clear Conversation History

```
DELETE /conversation/{thread_id}
```

### WebSocket API

#### Connect

```
ws://localhost:8000/ws/{client_id}
```

#### Send Message

```json
{
  "query": "Your question here",
  "thread_id": "optional_thread_id"
}
```

#### Receive Response

```json
{
  "type": "response",
  "query": "Your question",
  "response": "AI response",
  "requires_rag": true,
  "num_sources": 5,
  "timestamp": "2026-01-29T10:30:00",
  "thread_id": "thread123"
}
```

## Usage Examples

### Python WebSocket Client

```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8000/ws/myclient"
    async with websockets.connect(uri) as websocket:
        # Send query
        await websocket.send(json.dumps({
            "query": "How many IBM trades?",
            "thread_id": "demo"
        }))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(data["response"])

asyncio.run(chat())
```

### JavaScript WebSocket Client

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/myclient");

ws.onopen = () => {
  ws.send(
    JSON.stringify({
      query: "How many IBM trades?",
      thread_id: "demo",
    }),
  );
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.response);
};
```

### cURL (REST API)

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many IBM trades?",
    "thread_id": "demo"
  }'
```

## Running the Example Client

### Demo Mode (Automated)

```bash
python client_example.py demo
```

### Interactive Mode

```bash
python client_example.py interactive
```

## Features

### RAG Agent (LangGraph)

- **Query Classification**: Automatically determines if a query needs data retrieval
- **Conversation History**: Maintains context across multiple turns
- **Conditional Routing**: Routes to RAG or general response based on query type
- **Memory Management**: Keeps last 20 messages per thread

### Vector Search

- **FAISS Indexing**: Fast similarity search
- **Reranking**: Cross-encoder reranking for improved accuracy
- **Dataset Filtering**: Filter by holdings or trades
- **Context Formatting**: Optimized for LLM consumption

### WebSocket Server

- **Real-time Communication**: Instant responses
- **Multiple Connections**: Handle concurrent clients
- **Thread Management**: Isolated conversation threads
- **Error Handling**: Graceful error recovery

## System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- FAISS index and metadata files
- Google Gemini API key

## Performance Notes

- Initial startup takes 5-10 seconds to load models
- Query processing: 1-3 seconds per query
- WebSocket: Real-time (<100ms latency)
- Reranking adds ~200ms per query

## Security Considerations

1. **API Key**: Store GEMINI_API_KEY in environment variables
2. **CORS**: Configure allowed origins in production
3. **Rate Limiting**: Implement rate limiting for production
4. **Authentication**: Add authentication for production use

## Troubleshooting

### "Search engine not initialized"

- Ensure `financial_data.index` and `financial_data_metadata.pkl` exist
- Check file paths in configuration

### "GEMINI_API_KEY not set"

- Set environment variable or add to .env file

### WebSocket connection failed

- Check if server is running
- Verify port is not in use
- Check firewall settings

## Monitoring

Access these endpoints for monitoring:

- `/health` - Service health
- `/stats` - System statistics
- `/docs` - Interactive API documentation (Swagger)

## Production Deployment

For production deployment:

1. Use environment variables for configuration
2. Enable HTTPS/WSS
3. Add authentication middleware
4. Implement rate limiting
5. Set up logging and monitoring
6. Use a reverse proxy (nginx)
7. Configure CORS properly

## License

MIT License
