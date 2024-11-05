# probabilistic_reasoning.py

class ProbabilisticReasoning:
    def __init__(self):
        pass
    
    def bayesian_inference(self, prior, likelihood, evidence):
        """Applies Bayesian update rule to compute the posterior probability."""
        return (likelihood * prior) / evidence if evidence != 0 else 0
