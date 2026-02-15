"""
Query expansion service for semantic RAG.

Expands user queries into multiple semantic variants to improve retrieval coverage.
Uses LLM to generate rephrased queries that capture different semantic aspects.

Feature: Multi-perspective query expansion
- Synonym/paraphrase expansion
- Question reformulation
- Keyword expansion
- Technical term expansion

Usage:
    from services.query_expander import QueryExpander

    expander = QueryExpander(ai_service=ai_service)
    expanded = await expander.expand("What is machine learning?")
    # Returns: ["machine learning definition", "AI learning from data", ...]
"""

import logging
import re
from typing import List, Optional
from services.ai_service import AIService

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expands queries into multiple semantic variants."""

    def __init__(
        self,
        ai_service: AIService,
        max_variants: int = 3,
        enable_expansion: bool = True
    ):
        """
        Initialize QueryExpander.

        Args:
            ai_service: AIService instance for LLM access
            max_variants: Maximum number of query variants to generate (default: 3)
            enable_expansion: Whether to enable expansion (can be disabled via config)
        """
        self.ai_service = ai_service
        self.max_variants = max_variants
        self.enable_expansion = enable_expansion
        logger.info(
            f"QueryExpander initialized (max_variants={max_variants}, "
            f"enabled={enable_expansion})"
        )

    async def expand(self, query: str) -> List[str]:
        """
        Expand a query into multiple semantic variants.

        Args:
            query: Original user query

        Returns:
            List of query variants (including original query as first item)

        Example:
            >>> expander = QueryExpander(ai_service)
            >>> variants = await expander.expand("How do birds fly?")
            >>> # Returns: [
            >>> #   "How do birds fly?",
            >>> #   "Bird flight mechanics and physiology",
            >>> #   "Avian aerodynamics and wing movement",
            >>> #   "How do different bird species achieve flight?"
            >>> # ]
        """
        if not self.enable_expansion or not query or len(query) < 5:
            # Query too short or expansion disabled
            return [query]

        try:
            # Always include original query as first variant
            variants = [query]

            # If max_variants is 1, return just original
            if self.max_variants <= 1:
                return variants

            # Generate additional variants using LLM
            expanded = await self._generate_variants(
                query,
                num_variants=self.max_variants - 1  # -1 because we already have original
            )

            variants.extend(expanded)
            logger.info(
                f"[QueryExpander] Expanded '{query[:50]}...' to {len(variants)} variants"
            )

            return variants

        except Exception as e:
            logger.warning(f"[QueryExpander] Expansion failed, using original: {e}")
            return [query]

    async def _generate_variants(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate query variants using LLM.

        Args:
            query: Original query
            num_variants: Number of variants to generate

        Returns:
            List of alternative query formulations
        """
        prompt = f"""You are a search query optimization expert. Generate {num_variants} alternative search queries that capture different aspects of the user's intent.

Original query: "{query}"

Requirements:
- Each query should be concise (5-15 words)
- Queries should explore different angles/phrasings
- Focus on semantic meaning, not exact keywords
- Return ONLY the queries, one per line, without numbering or bullets
- Do not include the original query in the results

Generate {num_variants} alternative queries:"""

        try:
            # Call LLM to generate variants
            response = await self.ai_service.generate_text(
                prompt=prompt,
                max_tokens=200,
                temperature=0.7  # Some creativity but not too much
            )

            # Parse response into list of queries
            variants = [
                q.strip()
                for q in response.split('\n')
                if q.strip() and len(q.strip()) > 3
            ]

            # Return only requested number of variants
            return variants[:num_variants]

        except Exception as e:
            logger.warning(f"[QueryExpander] LLM variant generation failed: {e}")
            return []

    def expand_sync(self, query: str) -> List[str]:
        """
        Synchronous version of expand (for blocking contexts).

        Args:
            query: Original query

        Returns:
            List of query variants
        """
        if not self.enable_expansion or not query or len(query) < 5:
            return [query]

        # Fallback to simple rule-based expansion if async not available
        return self._simple_expansion(query)

    def _simple_expansion(self, query: str) -> List[str]:
        """
        Simple rule-based expansion (fallback when LLM not available).

        This uses heuristics to create variants without LLM.
        """
        variants = [query]

        # Remove common question words to create statement variant
        statement = re.sub(r'^(what|how|why|when|where|which|who|can|could|would|should|is|are|do|does)\s+', '', query, flags=re.IGNORECASE)
        if statement and statement != query:
            variants.append(statement)

        # Create question variant if not already one
        if not query.rstrip().endswith('?'):
            question = f"What about {query.lower()}?"
            if question != query:
                variants.append(question)

        # Add a synonym-based variant (very basic)
        synonym_pairs = {
            'machine learning': 'artificial intelligence algorithms',
            'bug': 'defect issue error',
            'feature': 'capability functionality',
            'optimize': 'improve speed performance',
        }

        for original, replacement in synonym_pairs.items():
            if original.lower() in query.lower():
                variant = query.replace(original, replacement.split()[0])
                if variant != query and variant not in variants:
                    variants.append(variant)
                break

        return variants[:self.max_variants]

    def get_stats(self) -> dict:
        """Get expansion statistics."""
        return {
            "enabled": self.enable_expansion,
            "max_variants": self.max_variants,
        }
