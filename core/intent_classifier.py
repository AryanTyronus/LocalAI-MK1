from typing import Tuple


class IntentClassifier:
    """Lightweight heuristic intent classifier.

    Returns: (intent, confidence)
    intents: chat, research, coding, action
    """

    KEYWORDS = {
        'research': ['research', 'study', 'explain', 'paper', 'article', 'what is', 'why', 'how', '?'],
        'coding': ['code', 'implement', 'debug', 'function', 'library', 'script', 'program', 'refactor'],
        'action': ['open', 'launch', 'run', 'execute', 'install', 'start', 'stop']
    }

    @classmethod
    def classify(cls, text: str) -> Tuple[str, float]:
        if not text:
            return 'chat', 0.0

        t = text.lower()
        scores = {'chat': 0, 'research': 0, 'coding': 0, 'action': 0}
        for intent, keywords in cls.KEYWORDS.items():
            for k in keywords:
                if k in t:
                    scores[intent] += 1

        # Choose highest score
        intent = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = (scores[intent] / total) if total > 0 else 0.0
        if intent == 'action':
            intent = 'agent'
        return intent, float(confidence)
