from transformers import pipeline

class InfiniteMind:
    def __init__(self):
        self.requirement_parser = pipeline("text2text-generation", model="t5-small")

    def parse_requirements(self, user_input):
        """
        Parse natural language requirements and translate them into system-specific requirements.
        Example: "I need to optimize memory usage in the object detection module."
        """
        prompt = f"Translate this into a system requirement: {user_input}"
        requirement = self.requirement_parser(prompt)[0]['generated_text']
        return requirement
