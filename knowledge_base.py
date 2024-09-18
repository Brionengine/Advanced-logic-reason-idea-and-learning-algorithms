# knowledge_base.py
import networkx as nx

class KnowledgeBase:
    def __init__(self):
        self.graph = nx.DiGraph()

    def store_experience(self, state, action, reward, next_state, done):
        state_id = id(state.tobytes())
        next_state_id = id(next_state.tobytes())
        # Add nodes if they don't exist
        if not self.graph.has_node(state_id):
            self.graph.add_node(state_id, state=state)
        if not self.graph.has_node(next_state_id):
            self.graph.add_node(next_state_id, state=next_state)
        # Add edge representing the action taken
        self.graph.add_edge(state_id, next_state_id, action=action, reward=reward, done=done)

    def query(self, state_id):
        # Retrieve information related to a specific state
        return self.graph[state_id]
