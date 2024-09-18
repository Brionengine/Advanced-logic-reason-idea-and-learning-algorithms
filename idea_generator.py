# idea_generator.py
import openai

class IdeaGenerator:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        openai.api_key = 'your-openai-api-key'  # Replace with your OpenAI API key

    def generate_ideas(self):
        # Extract experiences from the knowledge base
        experiences = self._extract_experiences()
        prompt = self._build_prompt(experiences)
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=150,
            n=3,
            stop=None,
            temperature=0.7,
        )
        ideas = [choice.text.strip() for choice in response.choices]
        return ideas

    def _extract_experiences(self):
        # Convert the knowledge graph into a textual format
        experiences = []
        for node in self.knowledge_base.graph.nodes(data=True):
            state = node[1]['state']
            experiences.append(f"State: {state}")
        return experiences

    def _build_prompt(self, experiences):
        experiences_text = "\n".join(experiences[:5])  # Use the first 5 experiences for brevity
        prompt = f"Based on the following experiences:\n{experiences_text}\nGenerate innovative ideas:"
        return prompt
