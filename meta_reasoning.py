# meta_reasoning.py

class MetaReasoning:
    def __init__(self):
        self.performance_history = []

    def evaluate_reasoning(self, outcome, expected):
        """Evaluates reasoning accuracy and logs performance for self-assessment."""
        self.performance_history.append(outcome == expected)
        return "Optimized" if sum(self.performance_history[-5:]) >= 4 else "Needs adjustment"
