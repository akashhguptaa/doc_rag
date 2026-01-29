"""
Vector Search and Reranking Module
Handles FAISS-based semantic search with reranking capabilities
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """
    Advanced vector search engine with FAISS indexing and cross-encoder reranking
    """

    def __init__(
        self,
        index_path: str = "financial_data.index",
        metadata_path: str = "financial_data_metadata.pkl",
        embedding_model_name: str = "all-mpnet-base-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize the vector search engine

        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata pickle file
            embedding_model_name: Name of the embedding model
            reranker_model_name: Name of the cross-encoder reranking model
        """
        logger.info("Initializing Vector Search Engine...")

        # Load FAISS index
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Load metadata
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.texts = data["texts"]
            self.holdings_count = data["holdings_count"]
            self.trades_count = data["trades_count"]
            self.model_name = data.get("model_name", embedding_model_name)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded embedding model: {self.model_name}")

        # Initialize reranker model
        try:
            self.reranker = CrossEncoder(reranker_model_name)
            self.reranking_enabled = True
            logger.info(f"Loaded reranker model: {reranker_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Reranking disabled.")
            self.reranking_enabled = False

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into an embedding vector

        Args:
            query: Text query to encode

        Returns:
            Normalized embedding vector
        """
        embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(embedding)
        return embedding.astype("float32")

    def search(
        self,
        query: str,
        top_k: int = 10,
        dataset_filter: Optional[str] = None,
        enable_reranking: bool = True,
    ) -> List[Dict]:
        """
        Perform semantic search with optional reranking

        Args:
            query: Search query
            top_k: Number of results to return
            dataset_filter: Filter by dataset type ('holdings' or 'trades')
            enable_reranking: Whether to apply reranking

        Returns:
            List of search results with scores and metadata
        """
        # Encode query
        query_embedding = self.encode_query(query)

        # Perform initial FAISS search (retrieve more for reranking)
        search_k = (
            top_k * 3 if enable_reranking and self.reranking_enabled else top_k * 2
        )
        scores, indices = self.index.search(query_embedding, search_k)

        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break

            metadata = self.metadata[idx]

            # Apply dataset filter if specified
            if dataset_filter and metadata["dataset_type"] != dataset_filter:
                continue

            result = {
                "score": float(score),
                "dataset_type": metadata["dataset_type"],
                "row_index": metadata["row_index"],
                "text": self.texts[idx],
                "metadata": metadata,
                "original_data": metadata["original_data"],
                "index": int(idx),
            }
            results.append(result)

        # Apply reranking if enabled
        if enable_reranking and self.reranking_enabled and len(results) > 0:
            results = self._rerank_results(query, results, top_k)
        else:
            results = results[:top_k]

        logger.info(
            f"Search completed: {len(results)} results for query: '{query[:50]}...'"
        )
        return results

    def _rerank_results(
        self, query: str, results: List[Dict], top_k: int
    ) -> List[Dict]:
        """
        Rerank search results using cross-encoder

        Args:
            query: Original search query
            results: Initial search results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return results

        # Prepare query-document pairs
        query_doc_pairs = [(query, result["text"]) for result in results]

        # Get reranking scores
        rerank_scores = self.reranker.predict(query_doc_pairs)

        # Update scores and sort
        for result, rerank_score in zip(results, rerank_scores):
            result["original_score"] = result["score"]
            result["rerank_score"] = float(rerank_score)
            result["score"] = float(rerank_score)  # Use rerank score as primary

        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(f"Reranking completed: top score {results[0]['rerank_score']:.4f}")
        return results[:top_k]

    def get_related_records(self, record_index: int, top_k: int = 5) -> List[Dict]:
        """
        Find records similar to a specific record

        Args:
            record_index: Index of the record to find similar items for
            top_k: Number of similar records to return

        Returns:
            List of similar records
        """
        if record_index >= len(self.texts):
            raise ValueError(f"Record index {record_index} out of range")

        # Use the embedding of the specified record as query
        query_vector = self.index.reconstruct(record_index).reshape(1, -1)
        scores, indices = self.index.search(query_vector, top_k + 1)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == record_index:  # Skip the original record
                continue
            if idx == -1:
                break

            metadata = self.metadata[idx]
            result = {
                "score": float(score),
                "dataset_type": metadata["dataset_type"],
                "row_index": metadata["row_index"],
                "text": self.texts[idx],
                "metadata": metadata,
                "original_data": metadata["original_data"],
            }
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def format_search_results_for_context(
        self, results: List[Dict], max_context_length: int = 5000
    ) -> str:
        """
        Format search results into a context string for LLM

        Args:
            results: Search results to format
            max_context_length: Maximum length of context string

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for i, result in enumerate(results):
            data = result["original_data"]
            dataset_type = result["dataset_type"]

            # Create structured context
            if dataset_type == "trades":
                context_part = f"TRADE RECORD {i+1}:\n"
                context_part += f"- Trade Type: {data.get('TradeTypeName', 'N/A')}\n"
                context_part += f"- Security: {data.get('Name', 'N/A')}\n"
                context_part += f"- Security Type: {data.get('SecurityType', 'N/A')}\n"
                context_part += f"- Portfolio: {data.get('PortfolioName', 'N/A')}\n"
                context_part += f"- Quantity: {data.get('Quantity', 'N/A')}\n"
                context_part += f"- Price: {data.get('Price', 'N/A')}\n"
                context_part += f"- Total Cash: {data.get('TotalCash', 'N/A')}\n"
                context_part += f"- Trade Date: {data.get('TradeDate', 'N/A')}\n"
            else:  # holdings
                context_part = f"HOLDING RECORD {i+1}:\n"
                context_part += f"- Security: {data.get('Name', 'N/A')}\n"
                context_part += f"- Portfolio: {data.get('PortfolioName', 'N/A')}\n"
                # Add other available holding fields
                for key, value in data.items():
                    if key not in ["Name", "PortfolioName"] and value is not None:
                        context_part += f"- {key}: {value}\n"

            # Check if adding this would exceed max length
            if current_length + len(context_part) > max_context_length:
                break

            context_parts.append(context_part)
            current_length += len(context_part)

        return "\n".join(context_parts)

    def get_stats(self) -> Dict:
        """Get statistics about the search engine"""
        return {
            "total_vectors": self.index.ntotal,
            "holdings_count": self.holdings_count,
            "trades_count": self.trades_count,
            "embedding_model": self.model_name,
            "reranking_enabled": self.reranking_enabled,
        }


# Convenience function for quick initialization
def create_search_engine(**kwargs) -> VectorSearchEngine:
    """Create and return a VectorSearchEngine instance"""
    return VectorSearchEngine(**kwargs)
