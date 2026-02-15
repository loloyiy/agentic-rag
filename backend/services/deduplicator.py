"""
Deduplication service for merged search results.

When searching with multiple query variants, results may overlap.
This module provides strategies to deduplicate and merge results.

Feature: Intelligent result deduplication
- Exact ID matching (remove duplicates by chunk_id)
- Semantic deduplication (remove similar content via embeddings)
- Fuzzy matching (remove near-duplicates via string similarity)

Usage:
    from services.deduplicator import Deduplicator

    dedup = Deduplicator(strategy='exact')
    unique = dedup.deduplicate(merged_results)
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class Deduplicator:
    """Deduplicates merged search results from multiple queries."""

    VALID_STRATEGIES = ['exact', 'fuzzy', 'semantic']

    def __init__(self, strategy: str = 'exact', similarity_threshold: float = 0.85):
        """
        Initialize Deduplicator.

        Args:
            strategy: Deduplication strategy ('exact', 'fuzzy', 'semantic')
            similarity_threshold: Threshold for fuzzy matching (0.0-1.0)

        Raises:
            ValueError: If strategy is invalid
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Valid options: {self.VALID_STRATEGIES}"
            )

        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        logger.info(
            f"Deduplicator initialized with strategy: {strategy} "
            f"(threshold={similarity_threshold})"
        )

    def deduplicate(
        self,
        results: List[Dict[str, Any]],
        preserve_order: bool = True,
        keep_highest_score: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate a list of search results.

        Args:
            results: List of result dicts (from multiple queries)
            preserve_order: Keep original order if True, else sort by score
            keep_highest_score: Keep highest score when duplicates found

        Returns:
            Deduplicated list of results
        """
        if not results:
            return []

        if self.strategy == 'exact':
            return self._deduplicate_exact(results, preserve_order, keep_highest_score)
        elif self.strategy == 'fuzzy':
            return self._deduplicate_fuzzy(results, preserve_order, keep_highest_score)
        elif self.strategy == 'semantic':
            return self._deduplicate_semantic(results, preserve_order, keep_highest_score)

    def _deduplicate_exact(
        self,
        results: List[Dict[str, Any]],
        preserve_order: bool,
        keep_highest_score: bool
    ) -> List[Dict[str, Any]]:
        """
        Exact deduplication: by chunk_id or content hash.

        Fast, deterministic. Best for structured data with IDs.
        """
        seen = {}
        deduped = []

        for result in results:
            # Try chunk_id first
            chunk_id = result.get('chunk_id')
            if chunk_id:
                key = chunk_id
            else:
                # Fallback: hash the text content
                text = result.get('text', '') or result.get('content', '')
                key = hashlib.md5(text.encode()).hexdigest()

            if key not in seen:
                seen[key] = result
                deduped.append(result)
            elif keep_highest_score:
                # Update if new result has higher score
                existing_score = seen[key].get('similarity', 0)
                new_score = result.get('similarity', 0)
                if new_score > existing_score:
                    # Find and replace in list
                    for i, r in enumerate(deduped):
                        if (r.get('chunk_id') == chunk_id or
                            hashlib.md5(r.get('text', '').encode()).hexdigest() == key):
                            deduped[i] = result
                            seen[key] = result
                            break

        logger.info(
            f"[Deduplicator] Exact deduplication: "
            f"{len(results)} -> {len(deduped)} results"
        )

        if not preserve_order:
            deduped.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return deduped

    def _deduplicate_fuzzy(
        self,
        results: List[Dict[str, Any]],
        preserve_order: bool,
        keep_highest_score: bool
    ) -> List[Dict[str, Any]]:
        """
        Fuzzy deduplication: by text similarity.

        Removes near-duplicates based on string similarity.
        Slower but catches rephrased content.
        """
        deduped = []
        threshold = self.similarity_threshold

        for result in results:
            text = result.get('text', '') or result.get('content', '')
            if not text:
                deduped.append(result)
                continue

            # Check similarity with existing results
            is_duplicate = False
            for existing in deduped:
                existing_text = existing.get('text', '') or existing.get('content', '')
                if not existing_text:
                    continue

                similarity = self._text_similarity(text, existing_text)
                if similarity >= threshold:
                    is_duplicate = True
                    if keep_highest_score:
                        # Replace if new score is higher
                        if result.get('similarity', 0) > existing.get('similarity', 0):
                            idx = deduped.index(existing)
                            deduped[idx] = result
                    break

            if not is_duplicate:
                deduped.append(result)

        logger.info(
            f"[Deduplicator] Fuzzy deduplication: "
            f"{len(results)} -> {len(deduped)} results "
            f"(threshold={threshold})"
        )

        if not preserve_order:
            deduped.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return deduped

    def _deduplicate_semantic(
        self,
        results: List[Dict[str, Any]],
        preserve_order: bool,
        keep_highest_score: bool
    ) -> List[Dict[str, Any]]:
        """
        Semantic deduplication: using embeddings similarity.

        Most accurate but requires embeddings in results.
        Falls back to fuzzy if embeddings not available.
        """
        # Check if results have embeddings
        has_embeddings = any(
            'embedding' in r or 'embedding_vector' in r or 'vector' in r
            for r in results
        )

        if not has_embeddings:
            logger.warning(
                "[Deduplicator] Embeddings not available, "
                "falling back to fuzzy deduplication"
            )
            return self._deduplicate_fuzzy(
                results,
                preserve_order,
                keep_highest_score
            )

        # Semantic dedup using embeddings
        deduped = []
        threshold = self.similarity_threshold

        for result in results:
            embedding = result.get('embedding') or result.get('embedding_vector')
            if not embedding:
                deduped.append(result)
                continue

            is_duplicate = False
            for existing in deduped:
                existing_embedding = (
                    existing.get('embedding') or
                    existing.get('embedding_vector')
                )
                if not existing_embedding:
                    continue

                similarity = self._cosine_similarity(embedding, existing_embedding)
                if similarity >= threshold:
                    is_duplicate = True
                    if keep_highest_score:
                        if result.get('similarity', 0) > existing.get('similarity', 0):
                            idx = deduped.index(existing)
                            deduped[idx] = result
                    break

            if not is_duplicate:
                deduped.append(result)

        logger.info(
            f"[Deduplicator] Semantic deduplication: "
            f"{len(results)} -> {len(deduped)} results"
        )

        if not preserve_order:
            deduped.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return deduped

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1, text2).ratio()

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)
