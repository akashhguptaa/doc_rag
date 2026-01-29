"""
FastAPI WebSocket Server for Financial Data Chatbot
Provides real-time conversational interface with RAG capabilities
"""

import os
import json
import logging
from typing import Dict, List
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from vector_search import create_search_engine
from rag_agent import create_rag_agent
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Data RAG Chatbot API",
    description="WebSocket-based conversational AI for financial data analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
search_engine = None
rag_agent = None
active_connections: Dict[str, WebSocket] = {}
conversation_histories: Dict[str, List[dict]] = {}


class QueryRequest(BaseModel):
    """Request model for REST API queries"""

    query: str
    thread_id: str = "default"


class QueryResponse(BaseModel):
    """Response model for queries"""

    response: str
    requires_rag: bool
    timestamp: str
    thread_id: str


# Startup event - Initialize models
@app.on_event("startup")
async def startup_event():
    """Initialize search engine and RAG agent on startup"""
    global search_engine, rag_agent

    try:
        logger.info("Initializing vector search engine...")
        search_engine = create_search_engine(
            index_path="financial_data.index",
            metadata_path="financial_data_metadata.pkl",
        )
        logger.info(f"Search engine ready: {search_engine.get_stats()}")

        logger.info("Initializing RAG agent...")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not set. Agent may not function properly.")

        rag_agent = create_rag_agent(
            search_engine=search_engine, gemini_api_key=gemini_api_key
        )
        logger.info("RAG agent initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    active_connections.clear()
    conversation_histories.clear()


@app.get("/health")
async def health_check():
    """Check if the service is healthy"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "search_engine": "ready" if search_engine else "not initialized",
        "rag_agent": "ready" if rag_agent else "not initialized",
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    stats = search_engine.get_stats()
    stats["active_connections"] = len(active_connections)
    stats["conversation_threads"] = len(conversation_histories)

    return stats


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    REST API endpoint for queries

    Args:
        request: Query request with question and thread_id

    Returns:
        Query response with answer and metadata
    """
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG agent not initialized")

    try:
        # Get conversation history for this thread
        chat_history = conversation_histories.get(request.thread_id, [])

        # Process query
        result = await rag_agent.aquery(
            user_query=request.query,
            chat_history=chat_history,
            thread_id=request.thread_id,
        )

        # Update conversation history
        if request.thread_id not in conversation_histories:
            conversation_histories[request.thread_id] = []

        conversation_histories[request.thread_id].append(
            {
                "role": "user",
                "content": request.query,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        conversation_histories[request.thread_id].append(
            {
                "role": "assistant",
                "content": result["response"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Keep only last 20 messages
        if len(conversation_histories[request.thread_id]) > 20:
            conversation_histories[request.thread_id] = conversation_histories[
                request.thread_id
            ][-20:]

        return QueryResponse(
            response=result["response"],
            requires_rag=result["requires_rag"],
            timestamp=datetime.utcnow().isoformat(),
            thread_id=request.thread_id,
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time conversational interface

    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await websocket.accept()
    active_connections[client_id] = websocket

    logger.info(
        f"Client {client_id} connected. Active connections: {len(active_connections)}"
    )

    # Send welcome message
    await websocket.send_json(
        {
            "type": "system",
            "message": "Connected to Financial Data RAG Chatbot",
            "timestamp": datetime.utcnow().isoformat(),
            "client_id": client_id,
        }
    )

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            query = message_data.get("query", "")
            thread_id = message_data.get("thread_id", client_id)

            if not query:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Query cannot be empty",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                continue

            logger.info(f"Received query from {client_id}: {query[:50]}...")

            # Send acknowledgment
            await websocket.send_json(
                {
                    "type": "processing",
                    "message": "Processing your query...",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            try:
                chat_history = conversation_histories.get(thread_id, [])

                # Process query through RAG agent
                result = await rag_agent.aquery(
                    user_query=query, chat_history=chat_history, thread_id=thread_id
                )

                # Update conversation history
                if thread_id not in conversation_histories:
                    conversation_histories[thread_id] = []

                conversation_histories[thread_id].append(
                    {
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                conversation_histories[thread_id].append(
                    {
                        "role": "assistant",
                        "content": result["response"],
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                # Keep only last 20 messages
                if len(conversation_histories[thread_id]) > 20:
                    conversation_histories[thread_id] = conversation_histories[
                        thread_id
                    ][-20:]

                # Send response
                await websocket.send_json(
                    {
                        "type": "response",
                        "query": query,
                        "response": result["response"],
                        "requires_rag": result["requires_rag"],
                        "num_sources": len(result.get("search_results", [])),
                        "timestamp": datetime.utcnow().isoformat(),
                        "thread_id": thread_id,
                    }
                )

                logger.info(f"Sent response to {client_id}")

            except Exception as e:
                logger.error(f"Error processing query for {client_id}: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Error processing query: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        active_connections.pop(client_id, None)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        active_connections.pop(client_id, None)
        try:
            await websocket.close()
        except:
            pass


@app.delete("/conversation/{thread_id}")
async def clear_conversation(thread_id: str):
    """
    Clear conversation history for a specific thread

    Args:
        thread_id: Thread ID to clear

    Returns:
        Confirmation message
    """
    if thread_id in conversation_histories:
        del conversation_histories[thread_id]
        return {"message": f"Conversation history cleared for thread {thread_id}"}
    else:
        return {"message": f"No conversation history found for thread {thread_id}"}


@app.get("/conversation/{thread_id}")
async def get_conversation(thread_id: str):
    """
    Get conversation history for a specific thread

    Args:
        thread_id: Thread ID to retrieve

    Returns:
        Conversation history
    """
    history = conversation_histories.get(thread_id, [])
    return {"thread_id": thread_id, "message_count": len(history), "messages": history}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Financial Data RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "query": "/query (POST)",
            "websocket": "/ws/{client_id}",
            "conversation": "/conversation/{thread_id} (GET/DELETE)",
        },
        "documentation": "/docs",
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")
