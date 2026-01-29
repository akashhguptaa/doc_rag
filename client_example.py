"""
Example WebSocket Client for Financial Data Chatbot
Demonstrates how to connect and interact with the WebSocket endpoint
"""

import asyncio
import websockets
import json
from datetime import datetime


async def chat_client(client_id: str = "test_client"):
    """
    Example WebSocket client for the chatbot

    Args:
        client_id: Unique identifier for this client
    """
    uri = f"ws://localhost:8000/ws/{client_id}"

    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Server: {welcome}\n")

        # Example queries to test the system
        test_queries = [
            "How many IBM trades were executed?",
            "Which portfolio has the best performance?",
            "What is the weather like today?",  # General query
            "Show me all bond transactions",
            "Compare HoldCo 1 and ClientA portfolios",
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"You: {query}")
            print(f"{'='*60}")

            # Send query
            message = {"query": query, "thread_id": "demo_thread"}
            await websocket.send(json.dumps(message))

            # Receive processing acknowledgment
            processing_msg = await websocket.recv()
            processing_data = json.loads(processing_msg)
            if processing_data["type"] == "processing":
                print(f"Status: {processing_data['message']}")

            # Receive response
            response_msg = await websocket.recv()
            response_data = json.loads(response_msg)

            if response_data["type"] == "response":
                print(f"\nAssistant: {response_data['response']}")
                print(f"\nMetadata:")
                print(
                    f"  - Type: {'RAG-based' if response_data['requires_rag'] else 'General'}"
                )
                print(f"  - Sources: {response_data['num_sources']}")
                print(f"  - Timestamp: {response_data['timestamp']}")
            elif response_data["type"] == "error":
                print(f"\nError: {response_data['message']}")

            # Wait a bit between queries
            await asyncio.sleep(2)

        print(f"\n{'='*60}")
        print("Demo completed!")


async def interactive_client(client_id: str = "interactive_client"):
    """
    Interactive WebSocket client for manual testing

    Args:
        client_id: Unique identifier for this client
    """
    uri = f"ws://localhost:8000/ws/{client_id}"

    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Server: {json.loads(welcome)['message']}\n")

        print("Type your questions (or 'quit' to exit):\n")

        while True:
            # Get user input
            query = input("You: ").strip()

            if query.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            if not query:
                continue

            # Send query
            message = {"query": query, "thread_id": "interactive_thread"}
            await websocket.send(json.dumps(message))

            # Receive processing acknowledgment
            processing_msg = await websocket.recv()
            processing_data = json.loads(processing_msg)
            if processing_data["type"] == "processing":
                print(f"[{processing_data['message']}]")

            # Receive response
            response_msg = await websocket.recv()
            response_data = json.loads(response_msg)

            if response_data["type"] == "response":
                print(f"\nAssistant: {response_data['response']}\n")
            elif response_data["type"] == "error":
                print(f"\nError: {response_data['message']}\n")


if __name__ == "__main__":
    import sys

    # Choose mode based on command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "interactive":
        print("Starting interactive client...")
        asyncio.run(interactive_client())
    else:
        print("Starting demo client...")
        asyncio.run(chat_client())
