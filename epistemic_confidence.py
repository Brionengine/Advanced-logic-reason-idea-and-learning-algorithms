# epistemic_confidence.py

class EpistemicConfidence:
    def __init__(self):
        pass
    
    def assess_confidence(self, statement, confidence_level):
        """Returns the statement if confidence is above a threshold; otherwise, returns low confidence warning."""
        return statement if confidence_level >= 0.8 else "Confidence too low"
