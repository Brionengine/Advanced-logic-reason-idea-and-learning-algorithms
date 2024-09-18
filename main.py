# main.py
import torch
from agent import Agent
from environment import Environment
from knowledge_base import KnowledgeBase
from idea_generator import IdeaGenerator
from experience_replay import ExperienceReplay

def main():
    # Initialize components
    env = Environment()
    agent = Agent(state_size=env.state_size, action_size=env.action_size)
    knowledge_base = KnowledgeBase()
    idea_generator = IdeaGenerator(knowledge_base)
    replay_buffer = ExperienceReplay()
    
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Agent selects action
            action = agent.select_action(state)
            # Environment returns next state and reward
            next_state, reward, done = env.step(action)
            # Store experience
            replay_buffer.push((state, action, reward, next_state, done))
            # Update knowledge base
            knowledge_base.store_experience(state, action, reward, next_state, done)
            # Agent learns from experience
            if len(replay_buffer) > agent.batch_size:
                experiences = replay_buffer.sample(agent.batch_size)
                agent.learn(experiences)
            state = next_state
        print(f"Episode {episode+1}/{num_episodes} completed.")

    # Generate ideas based on accumulated experiences
    ideas = idea_generator.generate_ideas()
    print("Generated Ideas:")
    for idea in ideas:
        print(f"- {idea}")

if __name__ == "__main__":
    main()
