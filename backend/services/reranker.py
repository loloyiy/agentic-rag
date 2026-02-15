"""
Reranker abstraction layer supporting multiple backends.

Feature: Pluggable reranker architecture
- Cohere: Cloud-based reranking via API (requires API key)
- Local: Offline reranking using CrossEncoder (no API key needed)

Usage:
    from services.reranker import RerankerFactory

    reranker = RerankerFactory.create(mode='cohere', api_key='...')
    results = reranker.rerank(query, documents, top_k=5)
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Reranker:
    """Base interface for rerankers."""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of dicts with 'index' and 'relevance_score' keys
        """
        raise NotImplementedError


class CohereReranker(Reranker):
    """Cohere cloud-based reranker."""

    COHERE_API_URL = "https://api.cohere.com/v1/rerank"
    DEFAULT_MODEL = "rerank-english-v3.0"

    def __init__(self, api_key: str):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key

        Raises:
            ValueError: If API key is empty or too short
        """
        if not api_key or len(api_key) < 10:
            raise ValueError("Invalid Cohere API key")
        self.api_key = api_key
        logger.info("Cohere reranker initialized")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Call Cohere rerank API via httpx.

        Fix #195: Use httpx directly instead of Cohere SDK to avoid
        Python 3.14 + Pydantic V1 incompatibility.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of dicts with 'index' and 'relevance_score' keys

        Raises:
            Exception: On API errors (401 Unauthorized, network errors, etc.)
        """
        import httpx  # Import here to avoid dependency if Cohere not used

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.DEFAULT_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_k,
            "return_documents": False  # We don't need the text back
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.COHERE_API_URL,
                    headers=headers,
                    json=payload
                )

                if response.status_code == 401:
                    raise Exception("Invalid Cohere API key (401 Unauthorized)")
                elif response.status_code != 200:
                    raise Exception(
                        f"Cohere API error: {response.status_code} - {response.text}"
                    )

                data = response.json()
                return data.get("results", [])
        except httpx.RequestError as e:
            raise Exception(f"Cohere API connection error: {str(e)}")


class LocalReranker(Reranker):
    """Local offline reranker using CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize local reranker with CrossEncoder model.

        Args:
            model_name: HuggingFace model identifier
                       (auto-downloaded on first use)
        """
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder model: {model_name}")
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info("Local reranker initialized successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using local CrossEncoder model.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of dicts with 'index' and 'relevance_score' keys
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            # If we have fewer documents than top_k, return all with scores
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
            return [
                {"index": i, "relevance_score": float(score)}
                for i, score in enumerate(scores)
            ]

        # Create (query, document) pairs for scoring
        pairs = [(query, doc) for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        # Create results with index and relevance_score
        results_with_scores = [
            {"index": i, "relevance_score": float(score)}
            for i, score in enumerate(scores)
        ]

        # Sort by relevance_score descending and take top_k
        results_with_scores.sort(
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        return results_with_scores[:top_k]


class RerankerFactory:
    """Factory for creating reranker instances."""

    @staticmethod
    def create(
        mode: str = "cohere",
        api_key: Optional[str] = None,
        cross_encoder_model: Optional[str] = None
    ) -> Reranker:
        """
        Create a reranker instance based on mode.

        Args:
            mode: "cohere" or "local"
            api_key: Cohere API key (required for Cohere mode)
            cross_encoder_model: CrossEncoder model name (optional, uses default for local)

        Returns:
            Reranker instance

        Raises:
            ValueError: If mode is invalid or required parameters are missing
        """
        mode = mode.lower().strip()

        if mode == "cohere":
            if not api_key:
                raise ValueError("api_key required for Cohere reranker")
            return CohereReranker(api_key)
        elif mode == "local":
            if cross_encoder_model:
                return LocalReranker(cross_encoder_model)
            else:
                return LocalReranker()  # Uses default model
        else:
            raise ValueError(
                f"Unknown reranker mode: {mode}. "
                "Choose 'cohere' or 'local'"
            )

    @staticmethod
    def get_available_modes() -> List[str]:
        """Get list of available reranker modes."""
        return ["cohere", "local"]
