# InfiniteMind

InfiniteMind is an AI project designed to enable an agent to learn from experiences, build a comprehensive knowledge base, and generate new ideas autonomously. The goal is to contribute towards the development of Artificial General Intelligence (AGI) and Artificial Super Intelligence (ASI) by focusing on the creation of algorithms that facilitate continuous learning and creativity.

## Features

- **Reinforcement Learning Agent**: Learns optimal actions through interactions with an environment.
- **Knowledge Base**: Stores experiences in a scalable graph database using NetworkX.
- **Idea Generation**: Generates new ideas based on accumulated experiences using OpenAI's GPT models.
- **Experience Replay**: Utilizes past experiences to improve learning efficiency.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/InfiniteMind.git
   cd InfiniteMind

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Set Up OpenAI API Key

Replace 'your-openai-api-key' in idea_generator.py with your actual OpenAI API key.

4. Usage
Run the main script to start the training process and generate ideas:
python main.py
