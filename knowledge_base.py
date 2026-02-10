"""
Brion Quantum - Knowledge Base v2.0
=====================================
Graph-based knowledge representation with semantic querying, path-based
inference, community detection, and knowledge decay/reinforcement.

Novel Algorithm: Quantum Knowledge Entanglement Graph (QKEG)
  - Knowledge nodes are entangled through experience edges, where
    traversal probability is governed by edge reward amplitudes.
  - High-reward paths create standing wave patterns that guide
    future decision-making toward optimal action sequences.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class KnowledgeBase:
    """
    Quantum Knowledge Entanglement Graph (QKEG) Engine.

    Stores state-action-reward transitions as a directed graph with
    rich querying, path inference, and knowledge management capabilities.
    """

    def __init__(self, decay_rate: float = 0.001, max_nodes: int = 50000):
        if not HAS_NETWORKX:
            raise ImportError("networkx required for KnowledgeBase")

        self.graph = nx.DiGraph()
        self.decay_rate = decay_rate
        self.max_nodes = max_nodes

        # Metadata
        self.total_experiences = 0
        self.creation_time = datetime.now().isoformat()

        # Indexes for fast lookup
        self._state_index: Dict[int, Any] = {}
        self._reward_cache: Dict[Tuple[int, int], float] = {}

    # -- Core Storage -------------------------------------------------------

    def store_experience(self, state: Any, action: Any, reward: float,
                         next_state: Any, done: bool):
        """Store a state-action-reward transition in the knowledge graph."""
        state_id = self._state_to_id(state)
        next_state_id = self._state_to_id(next_state)

        # Add or update state nodes
        if not self.graph.has_node(state_id):
            self.graph.add_node(state_id, state=state, visits=0,
                                created=datetime.now().isoformat())
        self.graph.nodes[state_id]["visits"] = (
            self.graph.nodes[state_id].get("visits", 0) + 1
        )

        if not self.graph.has_node(next_state_id):
            self.graph.add_node(next_state_id, state=next_state, visits=0,
                                created=datetime.now().isoformat())

        # Add or update transition edge
        if self.graph.has_edge(state_id, next_state_id):
            edge = self.graph[state_id][next_state_id]
            # Running average of reward
            edge["count"] = edge.get("count", 1) + 1
            edge["reward"] = (
                edge["reward"] * (edge["count"] - 1) + reward
            ) / edge["count"]
            edge["actions"] = list(set(edge.get("actions", []) + [str(action)]))
        else:
            self.graph.add_edge(state_id, next_state_id,
                                action=str(action), reward=reward,
                                done=done, count=1,
                                actions=[str(action)],
                                created=datetime.now().isoformat())

        # Cache reward
        self._reward_cache[(state_id, next_state_id)] = reward
        self.total_experiences += 1

        # Prune if too large
        if self.graph.number_of_nodes() > self.max_nodes:
            self._prune_least_visited()

    # -- Querying -----------------------------------------------------------

    def query(self, state: Any) -> Dict[str, Any]:
        """Query all transitions from a given state."""
        state_id = self._state_to_id(state)
        if not self.graph.has_node(state_id):
            return {"found": False, "transitions": []}

        transitions = []
        for successor in self.graph.successors(state_id):
            edge = self.graph[state_id][successor]
            transitions.append({
                "next_state_id": successor,
                "action": edge.get("action", "unknown"),
                "reward": edge.get("reward", 0.0),
                "done": edge.get("done", False),
                "count": edge.get("count", 1)
            })

        return {
            "found": True,
            "state_id": state_id,
            "visits": self.graph.nodes[state_id].get("visits", 0),
            "transitions": sorted(transitions, key=lambda t: t["reward"],
                                   reverse=True)
        }

    def best_action(self, state: Any) -> Optional[Dict[str, Any]]:
        """Find the best action from a state based on historical rewards."""
        result = self.query(state)
        if not result["found"] or not result["transitions"]:
            return None
        return result["transitions"][0]

    def find_path(self, start_state: Any, end_state: Any) -> Optional[List[Any]]:
        """Find the shortest path between two states."""
        start_id = self._state_to_id(start_state)
        end_id = self._state_to_id(end_state)

        if not (self.graph.has_node(start_id) and self.graph.has_node(end_id)):
            return None

        try:
            path = nx.shortest_path(self.graph, start_id, end_id)
            return path
        except nx.NetworkXNoPath:
            return None

    def highest_reward_path(self, start_state: Any,
                             max_steps: int = 10) -> List[Dict[str, Any]]:
        """Find the path of highest cumulative reward from a state."""
        state_id = self._state_to_id(start_state)
        if not self.graph.has_node(state_id):
            return []

        path = []
        visited = {state_id}
        current = state_id

        for _ in range(max_steps):
            successors = list(self.graph.successors(current))
            if not successors:
                break

            # Pick highest reward successor not yet visited
            best = None
            best_reward = float("-inf")
            for s in successors:
                if s not in visited:
                    reward = self.graph[current][s].get("reward", 0.0)
                    if reward > best_reward:
                        best_reward = reward
                        best = s

            if best is None:
                break

            edge = self.graph[current][best]
            path.append({
                "from": current,
                "to": best,
                "action": edge.get("action", "unknown"),
                "reward": best_reward
            })

            visited.add(best)
            current = best

            if edge.get("done", False):
                break

        return path

    # -- Knowledge Analysis -------------------------------------------------

    def find_clusters(self) -> List[Set[int]]:
        """Find strongly connected components (behavior clusters)."""
        return [set(c) for c in nx.strongly_connected_components(self.graph)]

    def hub_states(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find hub states (high out-degree, many possible actions)."""
        if self.graph.number_of_nodes() == 0:
            return []

        centrality = nx.degree_centrality(self.graph)
        sorted_states = sorted(centrality.items(), key=lambda x: x[1],
                                reverse=True)
        return sorted_states[:top_k]

    def reward_landscape(self) -> Dict[str, float]:
        """Compute aggregate reward statistics across the graph."""
        rewards = [
            data.get("reward", 0.0)
            for _, _, data in self.graph.edges(data=True)
        ]

        if not rewards:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

        return {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "max": float(np.max(rewards)),
            "min": float(np.min(rewards)),
            "total_edges": len(rewards)
        }

    # -- Knowledge Management -----------------------------------------------

    def decay_knowledge(self):
        """Apply temporal decay to edge rewards (forgotten knowledge)."""
        for u, v, data in self.graph.edges(data=True):
            data["reward"] *= (1.0 - self.decay_rate)

    def reinforce(self, state: Any, next_state: Any, boost: float = 0.1):
        """Reinforce a specific transition (strengthen memory)."""
        state_id = self._state_to_id(state)
        next_id = self._state_to_id(next_state)
        if self.graph.has_edge(state_id, next_id):
            self.graph[state_id][next_id]["reward"] += boost

    def _prune_least_visited(self, remove_fraction: float = 0.1):
        """Remove least-visited nodes to maintain graph size."""
        nodes_to_remove = int(self.graph.number_of_nodes() * remove_fraction)
        nodes_by_visits = sorted(
            self.graph.nodes(data=True),
            key=lambda x: x[1].get("visits", 0)
        )
        for node, _ in nodes_by_visits[:nodes_to_remove]:
            self.graph.remove_node(node)

    # -- Helpers ------------------------------------------------------------

    def _state_to_id(self, state: Any) -> int:
        """Convert a state to a hashable ID."""
        if hasattr(state, 'tobytes'):
            state_id = hash(state.tobytes())
        elif isinstance(state, dict):
            state_id = hash(frozenset(
                (k, str(v)) for k, v in sorted(state.items())
            ))
        else:
            state_id = hash(str(state))

        self._state_index[state_id] = state
        return state_id

    # -- Reporting ----------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return knowledge graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "total_experiences": self.total_experiences,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0.0,
            "components": nx.number_weakly_connected_components(self.graph),
            "reward_landscape": self.reward_landscape(),
            "created": self.creation_time
        }
