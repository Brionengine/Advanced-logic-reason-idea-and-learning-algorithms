# logic_engine.py

class LogicEngine:
    def __init__(self):
        pass
    
    def deductive_reasoning(self, premises, conclusion):
        """Determines if the conclusion logically follows from the premises."""
        if all(premises):
            return conclusion
        return "Undetermined"

    def inductive_reasoning(self, observations):
        """Generates a generalization based on observed instances."""
        if observations:
            return sum(observations) / len(observations)
        return None
