"""
RAG Agent with LangGraph for Financial Data Analysis
Implements conversational RAG with history and routing logic
"""

import os
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from google import genai
import logging

from vector_search import VectorSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the state structure
class AgentState(TypedDict):
    """State for the RAG agent"""

    query: str
    chat_history: List[dict]
    context: str
    response: str
    requires_rag: bool
    search_results: List[dict]


class FinancialRAGAgent:
    """
    LangGraph-based RAG agent for financial data queries
    Handles both RAG-based and general queries with conversation history
    """

    def __init__(
        self,
        vector_search_engine: VectorSearchEngine,
        gemini_api_key: str = None,
        model_name: str = "gemini-1.5-flash",
    ):
        """
        Initialize the RAG agent

        Args:
            vector_search_engine: Initialized VectorSearchEngine instance
            gemini_api_key: Google Gemini API key
            model_name: Gemini model name to use
        """
        self.search_engine = vector_search_engine
        self.model_name = model_name

        # Initialize Gemini client
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")

        self.client = genai.Client(api_key=api_key)
        logger.info(f"Initialized RAG Agent with model: {model_name}")

        # Build the LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow

        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_rag_response", self._generate_rag_response)
        workflow.add_node("generate_general_response", self._generate_general_response)

        # Add edges
        workflow.set_entry_point("classify_query")

        # Conditional routing based on query classification
        workflow.add_conditional_edges(
            "classify_query",
            self._route_query,
            {"rag": "retrieve_context", "general": "generate_general_response"},
        )

        workflow.add_edge("retrieve_context", "generate_rag_response")
        workflow.add_edge("generate_rag_response", END)
        workflow.add_edge("generate_general_response", END)

        # Compile with memory for conversation history
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _classify_query(self, state: AgentState) -> AgentState:
        """
        Classify whether the query requires RAG or can be answered generally

        Args:
            state: Current agent state

        Returns:
            Updated state with classification
        """
        query = state["query"].lower()

        # Keywords that indicate financial data queries
        financial_keywords = [
            "trade",
            "portfolio",
            "holding",
            "stock",
            "bond",
            "security",
            "transaction",
            "buy",
            "sell",
            "investment",
            "fund",
            "p&l",
            "profit",
            "loss",
            "cash",
            "value",
            "performance",
            "ibm",
            "meta",
            "holdings",
            "trades",
            "custodian",
            "strategy",
        ]

        # Check if query contains financial keywords
        requires_rag = any(keyword in query for keyword in financial_keywords)

        # Also check for question patterns about specific data
        question_patterns = [
            "how many",
            "show me",
            "what is",
            "which",
            "analyze",
            "compare",
        ]
        if any(pattern in query for pattern in question_patterns):
            requires_rag = True

        state["requires_rag"] = requires_rag
        logger.info(f"Query classified: requires_rag={requires_rag}")

        return state

    def _route_query(self, state: AgentState) -> Literal["rag", "general"]:
        """
        Route to appropriate processing node based on classification

        Args:
            state: Current agent state

        Returns:
            Node name to route to
        """
        return "rag" if state["requires_rag"] else "general"

    def _retrieve_context(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant context from vector database

        Args:
            state: Current agent state

        Returns:
            Updated state with search results and context
        """
        query = state["query"]

        # Perform vector search with reranking
        search_results = self.search_engine.search(
            query=query, top_k=7, enable_reranking=True
        )

        # Format results as context
        context = self.search_engine.format_search_results_for_context(search_results)

        state["search_results"] = search_results
        state["context"] = context

        logger.info(f"Retrieved {len(search_results)} relevant documents")
        return state

    def _generate_rag_response(self, state: AgentState) -> AgentState:
        """
        Generate response using RAG (retrieved context + LLM)

        Args:
            state: Current agent state

        Returns:
            Updated state with generated response
        """
        query = state["query"]
        context = state["context"]
        chat_history = state.get("chat_history", [])

        # Build conversation history string
        history_str = self._format_chat_history(chat_history)

        # Create prompt with context and history
        prompt = f"""You are a financial data analyst assistant. Answer the user's question based on the provided financial data context and conversation history.

IMPORTANT RULES:
1. Use ONLY the information provided in the financial data context
2. If the answer cannot be found in the provided data, say "Sorry, I cannot find the answer in the provided files."
3. Be accurate with numbers and calculations
4. Reference conversation history when relevant
5. Do not make up or assume information

CONVERSATION HISTORY:
{history_str}

FINANCIAL DATA CONTEXT:
{context}

USER QUESTION: {query}

Please provide a helpful and accurate answer:"""

        try:
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )

            if response and response.text:
                answer = response.text.strip()
            else:
                answer = "Sorry, I cannot find the answer in the provided files."

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            answer = f"Error generating response. Please try again."

        state["response"] = answer
        return state

    def _generate_general_response(self, state: AgentState) -> AgentState:
        """
        Generate response for general queries (without RAG)

        Args:
            state: Current agent state

        Returns:
            Updated state with generated response
        """
        query = state["query"]
        chat_history = state.get("chat_history", [])

        # Build conversation history string
        history_str = self._format_chat_history(chat_history)

        # Create prompt for general conversation
        prompt = f"""You are a helpful financial assistant. The user is asking a general question that doesn't require specific data analysis.

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {query}

Please provide a helpful response:"""

        try:
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )

            if response and response.text:
                answer = response.text.strip()
            else:
                answer = "I'm here to help with financial data analysis. Could you please ask a specific question about the data?"

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            answer = "I apologize, but I encountered an error. Please try again."

        state["response"] = answer
        return state

    def _format_chat_history(self, chat_history: List[dict]) -> str:
        """
        Format chat history for inclusion in prompts

        Args:
            chat_history: List of chat messages

        Returns:
            Formatted history string
        """
        if not chat_history:
            return "No previous conversation."

        history_parts = []
        for msg in chat_history[-5:]:  # Keep last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_parts.append(f"{role.upper()}: {content}")

        return "\n".join(history_parts)

    def query(
        self,
        user_query: str,
        chat_history: List[dict] = None,
        thread_id: str = "default",
    ) -> dict:
        """
        Process a user query through the RAG agent

        Args:
            user_query: The user's question
            chat_history: Previous conversation history
            thread_id: Thread ID for conversation persistence

        Returns:
            Dictionary with response and metadata
        """
        # Initialize state
        initial_state = {
            "query": user_query,
            "chat_history": chat_history or [],
            "context": "",
            "response": "",
            "requires_rag": False,
            "search_results": [],
        }

        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.graph.invoke(initial_state, config)

        # Return response with metadata
        return {
            "response": final_state["response"],
            "requires_rag": final_state["requires_rag"],
            "search_results": final_state.get("search_results", []),
            "query": user_query,
        }

    async def aquery(
        self,
        user_query: str,
        chat_history: List[dict] = None,
        thread_id: str = "default",
    ) -> dict:
        """
        Async version of query method

        Args:
            user_query: The user's question
            chat_history: Previous conversation history
            thread_id: Thread ID for conversation persistence

        Returns:
            Dictionary with response and metadata
        """
        # Initialize state
        initial_state = {
            "query": user_query,
            "chat_history": chat_history or [],
            "context": "",
            "response": "",
            "requires_rag": False,
            "search_results": [],
        }

        # Run the graph asynchronously
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.graph.ainvoke(initial_state, config)

        # Return response with metadata
        return {
            "response": final_state["response"],
            "requires_rag": final_state["requires_rag"],
            "search_results": final_state.get("search_results", []),
            "query": user_query,
        }


# Convenience function
def create_rag_agent(
    search_engine: VectorSearchEngine, gemini_api_key: str = None
) -> FinancialRAGAgent:
    """Create and return a FinancialRAGAgent instance"""
    return FinancialRAGAgent(
        vector_search_engine=search_engine, gemini_api_key=gemini_api_key
    )
