"""
Ensemble reranker for combining multiple reranking strategies.

Combines results from multiple rerankers using voting/averaging strategies
to produce more robust ranking decisions.

Feature: Multi-model ensemble ranking
- Voting-based ensemble (majority vote on ranking)
- Score-based ensemble (average normalized scores)
- Weighted ensemble (custom weights per reranker)
- Confidence-based weighting

Usage:
    from services.ensemble_reranker import EnsembleReranker
    from services.reranker import RerankerFactory

    # Create ensemble with multiple rerankers
    cohere = RerankerFactory.create(mode='cohere', api_key='...')
    local = RerankerFactory.create(mode='local')

    ensemble = EnsembleReranker([cohere, local], strategy='score_average')
    results = ensemble.rerank(query, documents, top_k=5)
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import numpy, fallback if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Provide simple fallback implementations
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        @staticmethod
        def isclose(a, b, rtol=1e-5, atol=1e-8):
            return abs(a - b) <= atol + rtol * abs(b)


class EnsembleReranker:
    """Combines multiple rerankers using ensemble strategies."""

    VALID_STRATEGIES = ['voting', 'score_average', 'score_weighted', 'borda_count']

    def __init__(
        self,
        rerankers: List[Any],
        strategy: str = 'score_average',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize EnsembleReranker.

        Args:
            rerankers: List of reranker instances
            strategy: Ensemble strategy ('voting', 'score_average', 'score_weighted', 'borda_count')
            weights: Optional weights for weighted strategies (must sum to 1.0)

        Raises:
            ValueError: If strategy is invalid or weights don't sum to 1.0
        """
        if not rerankers:
            raise ValueError("At least one reranker required")

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Valid options: {self.VALID_STRATEGIES}"
            )

        self.rerankers = rerankers
        self.strategy = strategy
        self.num_rerankers = len(rerankers)

        # Setup weights for weighted strategy
        if weights:
            if len(weights) != len(rerankers):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of rerankers ({len(rerankers)})"
                )
            if not np.isclose(sum(weights), 1.0):
                raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
            self.weights = weights
        else:
            # Equal weights by default
            self.weights = [1.0 / len(rerankers)] * len(rerankers)

        logger.info(
            f"EnsembleReranker initialized with {len(rerankers)} rerankers "
            f"using strategy: {strategy}"
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using ensemble of multiple rerankers.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return

        Returns:
            List of reranked results sorted by ensemble score
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            # If we have fewer docs than top_k, return all
            return [
                {
                    "index": i,
                    "relevance_score": 1.0,
                    "ensemble_score": 1.0,
                    "strategy": self.strategy
                }
                for i in range(len(documents))
            ]

        # Get rankings from each reranker
        reranker_results = []
        for i, reranker in enumerate(self.rerankers):
            try:
                results = reranker.rerank(query, documents, top_k=len(documents))
                reranker_results.append(results)
                logger.info(
                    f"[Ensemble] Reranker {i+1}/{len(self.rerankers)} "
                    f"returned {len(results)} results"
                )
            except Exception as e:
                logger.warning(
                    f"[Ensemble] Reranker {i+1} failed: {e}. "
                    f"Skipping this reranker."
                )
                reranker_results.append([])

        # Combine results using selected strategy
        if self.strategy == 'voting':
            final_results = self._voting_ensemble(reranker_results, top_k)
        elif self.strategy == 'score_average':
            final_results = self._score_average_ensemble(reranker_results, top_k)
        elif self.strategy == 'score_weighted':
            final_results = self._score_weighted_ensemble(reranker_results, top_k)
        elif self.strategy == 'borda_count':
            final_results = self._borda_count_ensemble(reranker_results, top_k)
        else:
            # Fallback (shouldn't happen due to validation)
            final_results = self._score_average_ensemble(reranker_results, top_k)

        logger.info(
            f"[Ensemble] Final ranking produced {len(final_results)} results "
            f"using {self.strategy} strategy"
        )

        return final_results

    def _voting_ensemble(
        self,
        reranker_results: List[List[Dict[str, Any]]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Voting-based ensemble: rank by how many rerankers put in top-k.

        Args:
            reranker_results: Results from each reranker
            top_k: Number of results to return

        Returns:
            Ensemble-ranked results
        """
        # Count votes for each document index
        votes = defaultdict(int)
        position_scores = defaultdict(lambda: [])

        for reranker_idx, results in enumerate(reranker_results):
            if not results:
                continue

            for rank, result in enumerate(results[:top_k * 2]):  # Consider top-2k for voting
                doc_idx = result.get("index", 0)
                votes[doc_idx] += 1
                # Inverse position score (earlier position = higher score)
                position_scores[doc_idx].append(1.0 / (rank + 1))

        # Create final results sorted by vote count, then by average position score
        final_results = []
        for doc_idx in sorted(votes.keys(), key=lambda x: (-votes[x], -np.mean(position_scores[x]))):
            if len(final_results) >= top_k:
                break

            avg_position_score = np.mean(position_scores[doc_idx])
            final_results.append({
                "index": doc_idx,
                "relevance_score": votes[doc_idx] / len(self.rerankers),  # Normalized vote count
                "ensemble_score": avg_position_score,
                "strategy": "voting",
                "votes": votes[doc_idx]
            })

        return final_results

    def _score_average_ensemble(
        self,
        reranker_results: List[List[Dict[str, Any]]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Score-average ensemble: average normalized scores from all rerankers.

        Args:
            reranker_results: Results from each reranker
            top_k: Number of results to return

        Returns:
            Ensemble-ranked results
        """
        # Collect scores for each document from all rerankers
        doc_scores = defaultdict(list)

        for reranker_idx, results in enumerate(reranker_results):
            if not results:
                continue

            # Normalize scores to 0-1 range
            scores = [r.get("relevance_score", 0) for r in results]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score

                for result in results:
                    doc_idx = result.get("index", 0)
                    raw_score = result.get("relevance_score", 0)

                    if score_range > 0:
                        normalized = (raw_score - min_score) / score_range
                    elif max_score > 0:
                        normalized = 0.5  # All same score
                    else:
                        normalized = 0.0

                    doc_scores[doc_idx].append(normalized)

        # Calculate average scores
        final_results = []
        for doc_idx in sorted(
            doc_scores.keys(),
            key=lambda x: np.mean(doc_scores[x]),
            reverse=True
        ):
            if len(final_results) >= top_k:
                break

            avg_score = np.mean(doc_scores[doc_idx])
            final_results.append({
                "index": doc_idx,
                "relevance_score": avg_score,
                "ensemble_score": avg_score,
                "strategy": "score_average",
                "num_votes": len(doc_scores[doc_idx])
            })

        return final_results

    def _score_weighted_ensemble(
        self,
        reranker_results: List[List[Dict[str, Any]]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Weighted ensemble: weighted average of normalized scores.

        Args:
            reranker_results: Results from each reranker
            top_k: Number of results to return

        Returns:
            Ensemble-ranked results
        """
        doc_scores = defaultdict(lambda: [])

        for reranker_idx, results in enumerate(reranker_results):
            if not results:
                continue

            # Normalize scores
            scores = [r.get("relevance_score", 0) for r in results]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score

                for result in results:
                    doc_idx = result.get("index", 0)
                    raw_score = result.get("relevance_score", 0)

                    if score_range > 0:
                        normalized = (raw_score - min_score) / score_range
                    elif max_score > 0:
                        normalized = 0.5
                    else:
                        normalized = 0.0

                    # Apply weight
                    weighted_score = normalized * self.weights[reranker_idx]
                    doc_scores[doc_idx].append(weighted_score)

        # Calculate weighted sums
        final_results = []
        for doc_idx in sorted(
            doc_scores.keys(),
            key=lambda x: sum(doc_scores[x]),
            reverse=True
        ):
            if len(final_results) >= top_k:
                break

            weighted_sum = sum(doc_scores[doc_idx])
            final_results.append({
                "index": doc_idx,
                "relevance_score": weighted_sum,
                "ensemble_score": weighted_sum,
                "strategy": "score_weighted",
                "num_votes": len(doc_scores[doc_idx])
            })

        return final_results

    def _borda_count_ensemble(
        self,
        reranker_results: List[List[Dict[str, Any]]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Borda count: sum of positional rankings across all rerankers.

        Args:
            reranker_results: Results from each reranker
            top_k: Number of results to return

        Returns:
            Ensemble-ranked results
        """
        doc_borda_scores = defaultdict(float)
        num_valid_rerankers = sum(1 for r in reranker_results if r)

        for reranker_idx, results in enumerate(reranker_results):
            if not results:
                continue

            # Borda count: highest position gets n points, lowest gets 1
            n = len(results)
            for rank, result in enumerate(results):
                doc_idx = result.get("index", 0)
                borda_points = (n - rank) / n  # Normalized to 0-1
                doc_borda_scores[doc_idx] += borda_points

        # Average Borda scores
        final_results = []
        for doc_idx in sorted(
            doc_borda_scores.keys(),
            key=lambda x: doc_borda_scores[x],
            reverse=True
        ):
            if len(final_results) >= top_k:
                break

            avg_borda = doc_borda_scores[doc_idx] / num_valid_rerankers
            final_results.append({
                "index": doc_idx,
                "relevance_score": avg_borda,
                "ensemble_score": avg_borda,
                "strategy": "borda_count",
            })

        return final_results
