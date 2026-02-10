"""
Brion Quantum - Experience Replay Engine v2.0
===============================================
Prioritized experience replay with temporal difference weighting,
importance sampling, and quantum-inspired memory consolidation.

Novel Algorithm: Quantum Priority Memory Consolidation (QPMC)
  - Experiences are stored with quantum amplitude weights that decay
    or amplify based on temporal relevance and surprise factor.
  - High-surprise experiences create stronger memory traces (higher
    amplitude), similar to quantum measurement collapse strengthening
    an eigenstate.

Developed by Brion Quantum AI Team
"""

import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from datetime import datetime


class ExperienceReplay:
    """
    Quantum Priority Memory Consolidation (QPMC) Engine.

    Enhanced experience replay buffer with prioritized sampling,
    temporal decay, and adaptive capacity management.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        # Buffer parameters
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent (how much prioritization)
        self.beta = beta  # Importance sampling correction
        self.beta_increment = beta_increment

        # Experience storage
        self.memory: deque = deque(maxlen=capacity)
        self.priorities: List[float] = []

        # Metadata
        self.total_pushed: int = 0
        self.total_sampled: int = 0

        # Consolidation tracking
        self.consolidation_log: List[Dict[str, Any]] = []

    # -- Core Operations ----------------------------------------------------

    def push(self, experience: Any, priority: Optional[float] = None):
        """
        Add an experience with optional priority.
        If no priority given, assigns max priority (most likely to be sampled next).
        """
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0

        if len(self.memory) >= self.capacity:
            self.priorities.pop(0)

        self.memory.append({
            "data": experience,
            "timestamp": datetime.now().isoformat(),
            "push_index": self.total_pushed,
            "access_count": 0
        })
        self.priorities.append(priority)
        self.total_pushed += 1

    def sample(self, batch_size: int) -> List[Any]:
        """
        Sample a batch using prioritized experience replay.
        Returns raw experience data for backwards compatibility.
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        if not self.memory:
            return []

        indices, weights = self._prioritized_indices(batch_size)

        batch = []
        for idx in indices:
            self.memory[idx]["access_count"] += 1
            batch.append(self.memory[idx]["data"])

        self.total_sampled += batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch

    def prioritized_sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Full prioritized sample with importance weights and indices.
        Suitable for gradient correction in learning algorithms.
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        if not self.memory:
            return {"experiences": [], "weights": [], "indices": []}

        indices, weights = self._prioritized_indices(batch_size)

        experiences = []
        for idx in indices:
            self.memory[idx]["access_count"] += 1
            experiences.append(self.memory[idx]["data"])

        self.total_sampled += batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        return {
            "experiences": experiences,
            "weights": weights.tolist(),
            "indices": indices.tolist()
        }

    def _prioritized_indices(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prioritized sampling indices and importance weights."""
        priorities = np.array(self.priorities[:len(self.memory)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            len(self.memory), size=batch_size,
            replace=False, p=probabilities
        )

        # Importance sampling weights
        n = len(self.memory)
        weights = (n * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        return indices, weights

    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """Update priorities for sampled experiences (e.g., after learning)."""
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = abs(priority) + 1e-6  # Prevent zero priority

    # -- Memory Consolidation -----------------------------------------------

    def consolidate(self, decay_rate: float = 0.01) -> Dict[str, Any]:
        """
        Quantum-inspired memory consolidation:
        - Recent experiences maintain high priority
        - Old experiences decay unless frequently accessed
        - High-surprise (high priority) experiences resist decay
        """
        if not self.memory:
            return {"consolidated": 0, "decayed": 0}

        decayed = 0
        for i in range(len(self.priorities)):
            age = self.total_pushed - self.memory[i]["push_index"]
            access = self.memory[i]["access_count"]

            # Decay factor: older and less accessed = more decay
            decay = np.exp(-decay_rate * age / (1 + access))
            self.priorities[i] *= decay

            if self.priorities[i] < 0.01:
                decayed += 1

        # Remove very low priority experiences
        while self.memory and self.priorities and self.priorities[0] < 0.001:
            self.memory.popleft()
            self.priorities.pop(0)

        result = {
            "consolidated": len(self.memory),
            "decayed": decayed,
            "timestamp": datetime.now().isoformat()
        }
        self.consolidation_log.append(result)
        return result

    # -- Learning (for UnifiedQuantumMind integration) ----------------------

    def learn(self):
        """
        Trigger memory consolidation and priority rebalancing.
        Used by the UnifiedQuantumMind orchestrator.
        """
        self.consolidate()

        # Rebalance priorities to prevent extreme skew
        if self.priorities:
            prio_array = np.array(self.priorities)
            mean_prio = prio_array.mean()
            if mean_prio > 0:
                # Soft normalization: pull toward mean
                self.priorities = (
                    0.9 * prio_array + 0.1 * mean_prio
                ).tolist()

    # -- Analytics ----------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        if not self.priorities:
            return {
                "size": 0, "capacity": self.capacity,
                "utilization": 0.0, "total_pushed": self.total_pushed,
                "total_sampled": self.total_sampled
            }

        prio_array = np.array(self.priorities)
        return {
            "size": len(self.memory),
            "capacity": self.capacity,
            "utilization": len(self.memory) / self.capacity,
            "total_pushed": self.total_pushed,
            "total_sampled": self.total_sampled,
            "priority_mean": float(prio_array.mean()),
            "priority_std": float(prio_array.std()),
            "priority_max": float(prio_array.max()),
            "priority_min": float(prio_array.min()),
            "beta": self.beta,
            "consolidations": len(self.consolidation_log)
        }

    def __len__(self):
        return len(self.memory)


# Backwards compatibility alias for UnifiedQuantumMind integration
MemoryReplay = ExperienceReplay
