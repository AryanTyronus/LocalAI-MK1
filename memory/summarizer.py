"""
Extractive summarization for conversation rolling summaries.
Compresses older conversation context while preserving key information.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional
from core.logger import logger
from core.config import (
    KEY_PHRASES_COUNT,
    PRESERVE_UNRESOLVED,
    SUMMARIZATION_MODEL_TYPE
)


class ConversationSummarizer:
    """
    Generates concise summaries of conversation segments.
    Focuses on key facts, decisions, and unresolved topics.
    """

    # Keywords that indicate important information
    KEY_INDICATORS = {
        'facts': ['is', 'are', 'was', 'were', 'my name', 'born', 'year', 'age'],
        'decisions': ['decided', 'will', "won't", 'should', 'must', 'prefer'],
        'unresolved': ['?', 'confused', 'unclear', 'need help', 'how', 'why'],
        'goals': ['want', 'goal', 'plan', 'aim', 'objective', 'aspire']
    }

    def __init__(self):
        """Initialize the summarizer."""
        self.model_type = SUMMARIZATION_MODEL_TYPE
        logger.info(f"Initialized ConversationSummarizer with {self.model_type} mode")

    def summarize_messages(
        self,
        messages: List[Dict[str, str]],
        max_phrases: int = KEY_PHRASES_COUNT
    ) -> Dict[str, any]:
        """
        Summarize a batch of conversation messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_phrases: Maximum key phrases to extract

        Returns:
            Summary dict with text, key phrases, entities, and metadata
        """
        if not messages:
            return self._empty_summary()

        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'message_count': len(messages),
            'key_phrases': [],
            'facts': [],
            'decisions': [],
            'unresolved_topics': [],
            'goals': [],
            'summary_text': ''
        }

        # Extract important content
        self._extract_key_information(messages, summary_data, max_phrases)

        # Generate summary text
        summary_data['summary_text'] = self._generate_summary_text(summary_data)

        return summary_data

    def _extract_key_information(
        self,
        messages: List[Dict],
        summary_data: Dict,
        max_phrases: int
    ) -> None:
        """Extract key information from messages."""
        all_text = ' '.join([m.get('content', '') for m in messages if m.get('role') == 'user'])

        # Extract facts
        for indicator in self.KEY_INDICATORS['facts']:
            sentences = self._extract_sentences_with_keyword(all_text, indicator)
            summary_data['facts'].extend(sentences[:2])

        # Extract decisions
        for indicator in self.KEY_INDICATORS['decisions']:
            sentences = self._extract_sentences_with_keyword(all_text, indicator)
            summary_data['decisions'].extend(sentences[:2])

        # Extract unresolved topics
        if PRESERVE_UNRESOLVED:
            for indicator in self.KEY_INDICATORS['unresolved']:
                sentences = self._extract_sentences_with_keyword(all_text, indicator)
                summary_data['unresolved_topics'].extend(sentences[:2])

        # Extract goals
        for indicator in self.KEY_INDICATORS['goals']:
            sentences = self._extract_sentences_with_keyword(all_text, indicator)
            summary_data['goals'].extend(sentences[:2])

        # Extract key phrases (most important nouns/concepts)
        summary_data['key_phrases'] = self._extract_key_phrases(all_text, max_phrases)

    def _extract_sentences_with_keyword(self, text: str, keyword: str) -> List[str]:
        """Extract sentences containing a specific keyword."""
        sentences = text.split('.')
        matching = []

        for sent in sentences:
            if keyword.lower() in sent.lower():
                cleaned = sent.strip()
                if cleaned and len(cleaned) > 5:
                    matching.append(cleaned)

        return matching

    def _extract_key_phrases(self, text: str, max_count: int = 5) -> List[str]:
        """
        Extract key phrases from text (simple word frequency approach).
        """
        # Simple approach: extract capitalized words and common important terms
        words = text.split()
        phrase_scores = {}

        important_terms = [
            'name', 'age', 'year', 'like', 'love', 'hate', 'enjoy',
            'study', 'learn', 'work', 'goal', 'want', 'prefer',
            'physics', 'math', 'science', 'technology'
        ]

        for word in words:
            word_lower = word.lower().strip('.,!?')
            # Score based on capitalization or importance
            if word[0].isupper() or word_lower in important_terms:
                phrase_scores[word_lower] = phrase_scores.get(word_lower, 0) + 1

        # Return top phrases
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases[:max_count]]

    def _generate_summary_text(self, summary_data: Dict) -> str:
        """Generate human-readable summary text."""
        parts = []

        if summary_data['facts']:
            parts.append("Facts: " + "; ".join(summary_data['facts'][:2]))

        if summary_data['decisions']:
            parts.append("Decisions: " + "; ".join(summary_data['decisions'][:2]))

        if summary_data['goals']:
            parts.append("Goals: " + "; ".join(summary_data['goals'][:1]))

        if summary_data['unresolved_topics']:
            parts.append("Unresolved: " + "; ".join(summary_data['unresolved_topics'][:1]))

        return " | ".join(parts) if parts else "Conversation segment summary"

    def _empty_summary(self) -> Dict:
        """Return an empty summary structure."""
        return {
            'timestamp': datetime.now().isoformat(),
            'message_count': 0,
            'key_phrases': [],
            'facts': [],
            'decisions': [],
            'unresolved_topics': [],
            'goals': [],
            'summary_text': ''
        }

    def merge_summaries(self, summaries: List[Dict]) -> Dict:
        """
        Merge multiple summaries into a single compressed summary.

        Args:
            summaries: List of summary dicts

        Returns:
            Merged summary
        """
        merged = {
            'timestamp': datetime.now().isoformat(),
            'message_count': sum(s.get('message_count', 0) for s in summaries),
            'key_phrases': [],
            'facts': [],
            'decisions': [],
            'unresolved_topics': [],
            'goals': [],
            'summary_text': ''
        }

        # Combine all information
        for summary in summaries:
            merged['key_phrases'].extend(summary.get('key_phrases', []))
            merged['facts'].extend(summary.get('facts', []))
            merged['decisions'].extend(summary.get('decisions', []))
            merged['unresolved_topics'].extend(summary.get('unresolved_topics', []))
            merged['goals'].extend(summary.get('goals', []))

        # Remove duplicates and limit
        merged['key_phrases'] = list(set(merged['key_phrases']))[:KEY_PHRASES_COUNT]
        merged['facts'] = merged['facts'][:3]
        merged['decisions'] = merged['decisions'][:3]
        merged['unresolved_topics'] = merged['unresolved_topics'][:2]
        merged['goals'] = merged['goals'][:2]

        # Generate summary text
        merged['summary_text'] = self._generate_summary_text(merged)

        return merged
