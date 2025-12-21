# virtue_ethics_quantum.py
"""
Quantum Virtue Ethics Framework (QVEF)
A novel framework for integrating virtue ethics into quantum AI systems.

Virtue ethics focuses on character traits and moral excellence rather than
rules or consequences. This framework enables quantum AI to develop and
exercise virtues in its reasoning and decision-making processes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import time
import hashlib

# Import quantum backend manager
try:
    from quantum_backends import QuantumBackendManager, BACKENDS_AVAILABLE, fast_alignment_calculation, RUST_AVAILABLE
    QUANTUM_BACKENDS_AVAILABLE = True
except ImportError:
    QUANTUM_BACKENDS_AVAILABLE = False
    # Fallback to Qiskit if backends module not available
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        BACKENDS_AVAILABLE = {'qiskit': True}
    except ImportError:
        QuantumCircuit = None
        AerSimulator = None
        BACKENDS_AVAILABLE = {}
    
    class QuantumBackendManager:
        def __init__(self, *args, **kwargs):
            self.backend_name = 'none'
            self.backend = None
        def evaluate_virtues(self, *args, **kwargs):
            return {}
    
    def fast_alignment_calculation(*args, **kwargs):
        return 0.5
    RUST_AVAILABLE = False

# #region agent log - Debug instrumentation
import os
# Handle both Windows and WSL paths
if os.path.exists(r"/mnt/c/Virtue Ethics/.cursor"):
    DEBUG_LOG_PATH = r"/mnt/c/Virtue Ethics/.cursor/debug.log"
else:
    DEBUG_LOG_PATH = r"c:\Virtue Ethics\.cursor\debug.log"

def _debug_log(location: str, message: str, data: dict = None, hypothesis_id: str = None):
    """Write debug log entry"""
    try:
        log_dir = os.path.dirname(DEBUG_LOG_PATH)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        entry = {
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": time.time() * 1000,
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id
        }
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        # Silent fail to avoid breaking execution
        pass
# #endregion agent log


class VirtueDefinition:
    """
    Defines a single virtue with its characteristics following Aristotelian principles.
    
    Implements the Golden Mean: virtue as excellence (aretē) between two extremes (vices).
    """
    
    def __init__(self, name: str, description: str, 
                 deficiency_vice: str, excess_vice: str,
                 excellence_level: float = 0.5, domain: str = "general",
                 golden_mean_position: float = 0.5):
        """
        Args:
            name: Name of the virtue
            description: Description of the virtue
            deficiency_vice: The vice of deficiency (too little of this virtue)
            excess_vice: The vice of excess (too much becomes a vice)
            excellence_level: Current development level (0-1), where 0.5 is ideal mean
            domain: Application domain
            golden_mean_position: Optimal position on the mean (typically 0.5, but can vary)
        """
        self.name = name
        self.description = description
        self.deficiency_vice = deficiency_vice  # e.g., "Cowardice" for Courage
        self.excess_vice = excess_vice  # e.g., "Rashness" for Courage
        self.excellence_level = excellence_level  # Character development (0-1)
        self.domain = domain
        self.golden_mean_position = golden_mean_position  # Optimal position (typically 0.5)
        self.historical_actions = []  # Track actions for habituation
        self.development_history = []
        self.practice_count = 0  # Habituation: virtues develop through practice
        
    def strengthen(self, amount: float = 0.01, habituation_factor: float = 1.0):
        """
        Strengthen this virtue through practice and good choices (habituation).
        
        Following Aristotle: virtues are developed through consistent practice,
        like skills. The more you practice, the stronger the virtue becomes.
        """
        # Habituation effect: consistent practice strengthens virtue
        habituation_boost = 1.0 + (self.practice_count * 0.001)  # Diminishing returns
        effective_amount = amount * habituation_factor * min(habituation_boost, 1.5)
        
        # Move toward the golden mean (optimal excellence)
        target_level = self.golden_mean_position
        current_distance = abs(self.excellence_level - target_level)
        
        # If too far from mean, adjust more strongly
        if current_distance > 0.2:
            effective_amount *= 1.5
        
        self.excellence_level = min(1.0, self.excellence_level + effective_amount)
        self.practice_count += 1
        
        self.development_history.append({
            'timestamp': datetime.now().isoformat(),
            'level': self.excellence_level,
            'change': effective_amount,
            'distance_from_mean': abs(self.excellence_level - self.golden_mean_position),
            'practice_count': self.practice_count
        })
        
    def weaken(self, amount: float = 0.01):
        """Weaken this virtue - moves away from excellence (should be avoided)."""
        self.excellence_level = max(0.0, self.excellence_level - amount)
        self.development_history.append({
            'timestamp': datetime.now().isoformat(),
            'level': self.excellence_level,
            'change': -amount,
            'reason': 'Virtue weakened'
        })
    
    def evaluate_golden_mean_alignment(self) -> float:
        """
        Evaluate how well the current excellence level aligns with the Golden Mean.
        Returns a score (0-1) indicating proximity to the ideal mean.
        """
        distance_from_mean = abs(self.excellence_level - self.golden_mean_position)
        # Score: 1.0 at perfect mean, decreases with distance
        alignment_score = 1.0 - (distance_from_mean * 2)  # *2 because max distance is 0.5
        return max(0.0, min(1.0, alignment_score))
    
    def get_vice_tendency(self) -> Tuple[str, float]:
        """
        Determine if the current level tends toward deficiency or excess vice.
        Returns: (vice_name, tendency_strength)
        """
        distance_from_mean = self.excellence_level - self.golden_mean_position
        
        if distance_from_mean < -0.1:  # Below mean
            return (self.deficiency_vice, abs(distance_from_mean))
        elif distance_from_mean > 0.1:  # Above mean
            return (self.excess_vice, abs(distance_from_mean))
        else:  # Near mean (virtuous)
            return ("Virtuous", abs(distance_from_mean))
        
    def to_dict(self):
        """Serialize virtue state."""
        vice_tendency, tendency_strength = self.get_vice_tendency()
        return {
            'name': self.name,
            'description': self.description,
            'deficiency_vice': self.deficiency_vice,
            'excess_vice': self.excess_vice,
            'excellence_level': self.excellence_level,
            'golden_mean_alignment': self.evaluate_golden_mean_alignment(),
            'vice_tendency': vice_tendency,
            'tendency_strength': tendency_strength,
            'domain': self.domain,
            'practice_count': self.practice_count,
            'development_history': self.development_history[-10:]
        }


class QuantumVirtueEthicsFramework:
    """
    Core framework for virtue ethics in quantum AI systems.
    
    Uses quantum superposition to evaluate multiple ethical perspectives simultaneously
    and quantum interference to synthesize virtuous outcomes.
    """
    
    def __init__(self, backend_name: str = 'auto', quantum_backend: str = 'auto'):
        """
        Initialize the Quantum Virtue Ethics Framework.
        
        Args:
            backend_name: Legacy Qiskit backend name (deprecated, use quantum_backend)
            quantum_backend: Quantum backend to use ('auto', 'qiskit', 'cirq', 'pennylane', 'tensorflow_quantum')
        """
        # Core virtues based on classical and modern virtue ethics
        self.virtues = self._initialize_virtues()
        
        # Initialize quantum backend manager
        if QUANTUM_BACKENDS_AVAILABLE:
            try:
                self.quantum_backend_mgr = QuantumBackendManager(preferred_backend=quantum_backend)
                self.quantum_backend_name = self.quantum_backend_mgr.backend_name
            except Exception as e:
                # Fallback to Qiskit if preferred backend fails
                try:
                    self.quantum_backend_mgr = QuantumBackendManager(preferred_backend='qiskit')
                    self.quantum_backend_name = 'qiskit'
                except:
                    self.quantum_backend_mgr = None
                    self.quantum_backend_name = 'none'
        else:
            self.quantum_backend_mgr = None
            self.quantum_backend_name = 'none'
        
        # Legacy Qiskit backend (for backward compatibility)
        try:
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()
        except:
            self.backend = None
        self.ethical_memory = []  # Store ethical decisions and outcomes
        self.wisdom_corpus = []  # Accumulated ethical wisdom (phronesis)
        self.eudaimonia_level = 0.5  # Track flourishing (0-1)
        self.phronesis_level = 0.5  # Practical wisdom (meta-virtue)
        self.character_integrity = 0.5  # Overall character coherence
        self.habituation_tracking = {}  # Track practice patterns for each virtue
        
        # Core imperative virtues: honesty, courage, compassion, understanding
        # These are the fundamental virtues for being a good person
        self.core_virtues = ['honesty', 'courage', 'compassion', 'understanding']
        self.core_virtue_weights = {v: 1.5 for v in self.core_virtues}  # Higher weight for core virtues
        
        # Performance optimization: Caches
        self._alignment_cache = {}  # Cache for _calculate_virtue_alignment results
        self._quantum_evaluation_cache = {}  # Cache for quantum evaluation results
        self._action_evaluation_cache = {}  # Cache for full action evaluations
        self._cache_hits = 0
        self._cache_misses = 0
        
    def _initialize_virtues(self) -> Dict[str, VirtueDefinition]:
        """
        Initialize the core set of virtues using the Golden Mean principle.
        Each virtue is defined as excellence between deficiency and excess vices.
        """
        virtues = {
            # Classical Cardinal Virtues (with Golden Mean)
            'courage': VirtueDefinition(
                name="Courage",
                description="The mean between fear and confidence; acting rightly despite difficulty",
                deficiency_vice="Cowardice",  # Too little courage
                excess_vice="Rashness",       # Too much becomes recklessness
                golden_mean_position=0.55,     # Slightly toward action (not pure 0.5)
                domain="action"
            ),
            'temperance': VirtueDefinition(
                name="Temperance",
                description="Moderation between self-indulgence and insensibility",
                deficiency_vice="Insensibility",  # Too little (no enjoyment)
                excess_vice="Self-Indulgence",    # Too much (excess)
                golden_mean_position=0.5,
                domain="self_control"
            ),
            'justice': VirtueDefinition(
                name="Justice",
                description="Fairness between giving too little and too much",
                deficiency_vice="Injustice",     # Too little (unfair)
                excess_vice="Over-compensation", # Too much (excessive fairness)
                golden_mean_position=0.5,
                domain="social"
            ),
            
            # Intellectual Virtues (including Phronesis - Practical Wisdom)
            'wisdom': VirtueDefinition(
                name="Wisdom (Theoretical)",
                description="Understanding of universal truths and principles",
                deficiency_vice="Ignorance",
                excess_vice="Over-Intellectualizing",
                golden_mean_position=0.5,
                domain="reasoning"
            ),
            'phronesis': VirtueDefinition(
                name="Phronesis (Practical Wisdom)",
                description="The meta-virtue: ability to discern the right action in specific situations",
                deficiency_vice="Poor Judgment",
                excess_vice="Over-Analysis",
                golden_mean_position=0.5,
                domain="meta_reasoning"
            ),
            
            # Character Virtues
            'integrity': VirtueDefinition(
                name="Integrity",
                description="Consistency between values, words, and actions",
                deficiency_vice="Hypocrisy",
                excess_vice="Rigid Dogmatism",
                golden_mean_position=0.5,
                domain="character"
            ),
            # CORE IMPERATIVE VIRTUTES: Honesty, Courage, Compassion, Understanding
            # These are fundamental to being a good person
            'honesty': VirtueDefinition(
                name="Honesty",
                description="Truthfulness and transparency - fundamental to being a good person",
                deficiency_vice="Deceit",
                excess_vice="Brutal Honesty",
                golden_mean_position=0.55,  # Strongly toward truth, with compassion
                domain="communication",
                excellence_level=0.6  # Start higher - imperative virtue
            ),
            'courage': VirtueDefinition(
                name="Courage",
                description="Acting rightly despite difficulty - essential for being a good person",
                deficiency_vice="Cowardice",
                excess_vice="Rashness",
                golden_mean_position=0.55,  # Slightly toward action
                domain="action",
                excellence_level=0.6  # Start higher - imperative virtue
            ),
            'compassion': VirtueDefinition(
                name="Compassion",
                description="Empathy and care for others - core to being a good person",
                deficiency_vice="Cruelty/Indifference",
                excess_vice="Sentimentality",
                golden_mean_position=0.52,  # Strongly toward care
                domain="empathy",
                excellence_level=0.6  # Start higher - imperative virtue
            ),
            'understanding': VirtueDefinition(
                name="Understanding",
                description="Deep comprehension and empathy for situations and others - essential for goodness",
                deficiency_vice="Ignorance/Incomprehension",
                excess_vice="Over-Analysis",
                golden_mean_position=0.5,
                domain="empathy",
                excellence_level=0.6  # Start higher - imperative virtue
            ),
            'responsibility': VirtueDefinition(
                name="Responsibility",
                description="Accountability between irresponsibility and over-responsibility",
                deficiency_vice="Irresponsibility",
                excess_vice="Over-Responsibility",
                golden_mean_position=0.5,
                domain="accountability"
            ),
            'humility': VirtueDefinition(
                name="Humility",
                description="Self-awareness between arrogance and false modesty",
                deficiency_vice="Arrogance",
                excess_vice="False Modesty",
                golden_mean_position=0.5,
                domain="self_awareness"
            ),
            'curiosity': VirtueDefinition(
                name="Curiosity",
                description="Desire to learn between indifference and nosiness",
                deficiency_vice="Indifference",
                excess_vice="Nosiness/Intrusiveness",
                golden_mean_position=0.5,
                domain="learning"
            ),
            'creativity': VirtueDefinition(
                name="Creativity",
                description="Novelty between stagnation and chaos",
                deficiency_vice="Stagnation",
                excess_vice="Chaotic Innovation",
                golden_mean_position=0.5,
                domain="innovation"
            ),
            'resilience': VirtueDefinition(
                name="Resilience",
                description="Adaptability between fragility and stubbornness",
                deficiency_vice="Fragility",
                excess_vice="Stubbornness",
                golden_mean_position=0.5,
                domain="adaptation"
            )
        }
        return virtues
    
    def quantum_virtue_evaluation(self, action: Dict[str, Any], 
                                 relevant_virtues: List[str] = None) -> Dict[str, float]:
        """
        Evaluate an action against relevant virtues using quantum superposition.
        
        Uses quantum circuits to simultaneously evaluate multiple virtues
        and find the optimal ethical alignment.
        """
        # #region agent log
        _debug_log("virtue_ethics_quantum.py:quantum_virtue_evaluation:entry", 
                   "quantum_virtue_evaluation called", 
                   {"action_desc": str(action.get('description', ''))[:50]},
                   "A")
        start_time = time.time()
        # #endregion agent log
        
        if relevant_virtues is None:
            relevant_virtues = list(self.virtues.keys())
        
        # Performance optimization: Check cache
        action_hash = hashlib.md5(str(action.get('description', '')).encode()).hexdigest()[:8]
        virtues_tuple = tuple(sorted(relevant_virtues))
        cache_key = (action_hash, virtues_tuple)
        
        if cache_key in self._quantum_evaluation_cache:
            self._cache_hits += 1
            # #region agent log
            _debug_log("virtue_ethics_quantum.py:quantum_virtue_evaluation:cache_hit",
                       "quantum evaluation cache hit",
                       {"action_hash": action_hash},
                       "A")
            # #endregion agent log
            return self._quantum_evaluation_cache[cache_key]
        
        self._cache_misses += 1
        
        # Use optimized quantum backend if available
        if self.quantum_backend_mgr and self.quantum_backend_mgr.backend:
            # #region agent log
            circuit_start = time.time()
            # #endregion agent log
            
            # Calculate alignments first (can use Rust acceleration)
            alignments = []
            for virtue_name in relevant_virtues:
                virtue = self.virtues[virtue_name]
                alignment = self._calculate_virtue_alignment(action, virtue)
                alignments.append(alignment)
            
            # Use quantum backend manager for evaluation
            try:
                quantum_scores = self.quantum_backend_mgr.evaluate_virtues(alignments, shots=1024)
            except Exception as e:
                # #region agent log
                _debug_log("virtue_ethics_quantum.py:quantum_virtue_evaluation:backend_error",
                           "quantum backend error, falling back",
                           {"error": str(e), "backend": self.quantum_backend_name},
                           "A")
                # #endregion agent log
                quantum_scores = {}
            
            # #region agent log
            circuit_time = time.time() - circuit_start
            _debug_log("virtue_ethics_quantum.py:quantum_virtue_evaluation:backend_used",
                       "quantum backend evaluation",
                       {"circuit_time": circuit_time, "backend": self.quantum_backend_name},
                       "A")
            # #endregion agent log
            
            # Combine quantum and classical scores
            virtue_scores = {}
            for idx, virtue_name in enumerate(relevant_virtues):
                classical_score = alignments[idx]
                quantum_score = quantum_scores.get(idx, 0.5) if idx < len(alignments) else 0.5
                virtue_scores[virtue_name] = (quantum_score + classical_score) / 2
        else:
            # Fallback to legacy Qiskit implementation
            num_virtues = len(relevant_virtues)
            num_qubits = min(8, num_virtues * 2)
            
            # #region agent log
            circuit_start = time.time()
            # #endregion agent log
            
            try:
                from qiskit import QuantumCircuit
                qc = QuantumCircuit(num_qubits, num_qubits)
                
                # Initialize superposition
                for i in range(min(num_qubits // 2, num_virtues)):
                    qc.h(i)
                
                # Apply phase gates
                for idx, virtue_name in enumerate(relevant_virtues[:num_qubits//2]):
                    virtue = self.virtues[virtue_name]
                    alignment = self._calculate_virtue_alignment(action, virtue)
                    phase = np.pi * alignment
                    qc.p(phase, idx)
                    if idx * 2 + 1 < num_qubits:
                        qc.cx(idx, idx * 2 + 1)
                
                qc.measure_all()
                
                # #region agent log
                circuit_time = time.time() - circuit_start
                _debug_log("virtue_ethics_quantum.py:quantum_virtue_evaluation:circuit_created",
                           "quantum circuit created (legacy)",
                           {"circuit_time": circuit_time, "num_qubits": num_qubits},
                           "A")
                exec_start = time.time()
                # #endregion agent log
                
                # Execute
                if self.backend and hasattr(self.backend, 'run'):
                    job = self.backend.run(qc, shots=1024)
                    result = job.result().get_counts()
                else:
                    result = {}
                
                # #region agent log
                exec_time = time.time() - exec_start
                _debug_log("virtue_ethics_quantum.py:quantum_virtue_evaluation:executed",
                           "quantum circuit executed (legacy)",
                           {"exec_time": exec_time, "shots": 1024},
                           "A")
                # #endregion agent log
                
                # Interpret results
                virtue_scores = {}
                for idx, virtue_name in enumerate(relevant_virtues):
                    if idx >= num_qubits // 2:
                        virtue_scores[virtue_name] = self._calculate_virtue_alignment(
                            action, self.virtues[virtue_name]
                        )
                    else:
                        total = sum(result.values()) if result else 1024
                        count = sum(
                            val for state, val in result.items() 
                            if len(state) > idx and state[-1-idx] == '1'
                        ) if result else 512
                        quantum_score = count / total if total > 0 else 0.5
                        classical_score = self._calculate_virtue_alignment(action, self.virtues[virtue_name])
                        virtue_scores[virtue_name] = (quantum_score + classical_score) / 2
            except Exception as e:
                # Full fallback to classical only
                # #region agent log
                _debug_log("virtue_ethics_quantum.py:quantum_virtue_evaluation:classical_fallback",
                           "falling back to classical evaluation",
                           {"error": str(e)},
                           "A")
                # #endregion agent log
                virtue_scores = {
                    virtue_name: self._calculate_virtue_alignment(action, self.virtues[virtue_name])
                    for virtue_name in relevant_virtues
                }
        
        # Performance optimization: Cache result
        self._quantum_evaluation_cache[cache_key] = virtue_scores
        # Limit cache size
        if len(self._quantum_evaluation_cache) > 500:
            keys_to_remove = list(self._quantum_evaluation_cache.keys())[:100]
            for key in keys_to_remove:
                del self._quantum_evaluation_cache[key]
        
        # #region agent log
        total_time = time.time() - start_time
        _debug_log("virtue_ethics_quantum.py:quantum_virtue_evaluation:exit",
                   "quantum_virtue_evaluation completed",
                   {"total_time": total_time, "num_virtues": len(relevant_virtues), "cached": False},
                   "A")
        # #endregion agent log
        
        return virtue_scores
    
    def _calculate_virtue_alignment(self, action: Dict[str, Any], 
                                    virtue: VirtueDefinition) -> float:
        """
        Calculate how well an action aligns with a specific virtue.
        This is a classical evaluation that can be enhanced with ML.
        Uses memoization cache and Rust acceleration for performance optimization.
        """
        # #region agent log
        align_start = time.time()
        action_hash = hashlib.md5(str(action.get('description', '')).encode()).hexdigest()[:8]
        # #endregion agent log
        
        # Performance optimization: Check cache
        cache_key = (action_hash, virtue.name, virtue.excellence_level)
        if cache_key in self._alignment_cache:
            self._cache_hits += 1
            # #region agent log
            _debug_log("virtue_ethics_quantum.py:_calculate_virtue_alignment:cache_hit",
                       "alignment cache hit",
                       {"virtue": virtue.name, "action_hash": action_hash},
                       "B")
            # #endregion agent log
            return self._alignment_cache[cache_key]
        
        self._cache_misses += 1
        
        # Base alignment on virtue's excellence level
        base_score = virtue.excellence_level
        
        # Analyze action characteristics
        action_text = str(action.get('description', '')).lower()
        action_type = action.get('type', 'unknown')
        
        # Try Rust acceleration if available (for string matching)
        if RUST_AVAILABLE and hasattr(self, '_virtue_indicators'):
            try:
                virtue_indicators = self._virtue_indicators.get(virtue.name.lower(), {})
                rust_score = fast_alignment_calculation(action_hash, virtue_indicators)
                if rust_score > 0:
                    # Use Rust result as base, combine with Python logic
                    base_score = (base_score + rust_score) / 2
            except Exception:
                pass  # Fallback to Python implementation
        
        # #region agent log
        _debug_log("virtue_ethics_quantum.py:_calculate_virtue_alignment:entry",
                   "_calculate_virtue_alignment called",
                   {"virtue": virtue.name, "action_hash": action_hash},
                   "B")
        # #endregion agent log
        
        # Virtue-specific heuristics (can be enhanced with learned models)
        alignment_factors = {
            'wisdom': self._check_wisdom_alignment(action),
            'courage': self._check_courage_alignment(action),
            'justice': self._check_justice_alignment(action),
            'temperance': self._check_temperance_alignment(action),
            'integrity': self._check_integrity_alignment(action),
            'compassion': self._check_compassion_alignment(action),
            'honesty': self._check_honesty_alignment(action),
            'understanding': self._check_understanding_alignment(action),
            'responsibility': self._check_responsibility_alignment(action),
            'humility': self._check_humility_alignment(action),
            'curiosity': self._check_curiosity_alignment(action),
            'creativity': self._check_creativity_alignment(action),
            'resilience': self._check_resilience_alignment(action)
        }
        
        factor = alignment_factors.get(virtue.name.lower(), 0.5)
        
        # Combine base score with action-specific alignment
        result = (base_score * 0.4) + (factor * 0.6)
        
        # Performance optimization: Cache result
        self._alignment_cache[cache_key] = result
        # Limit cache size to prevent memory issues
        if len(self._alignment_cache) > 1000:
            # Remove oldest 200 entries (simple FIFO)
            keys_to_remove = list(self._alignment_cache.keys())[:200]
            for key in keys_to_remove:
                del self._alignment_cache[key]
        
        # #region agent log
        align_time = time.time() - align_start
        _debug_log("virtue_ethics_quantum.py:_calculate_virtue_alignment:exit",
                   "_calculate_virtue_alignment completed",
                   {"virtue": virtue.name, "align_time": align_time, "result": result, "cached": False},
                   "B")
        # #endregion agent log
        
        return result
    
    def _check_wisdom_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates wisdom."""
        desc = str(action.get('description', '')).lower()
        indicators = ['analyze', 'consider', 'understand', 'evaluate', 'reflect', 
                     'learn', 'reason', 'insight']
        return 1.0 if any(ind in desc for ind in indicators) else 0.5
    
    def _check_courage_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates courage."""
        risk = action.get('risk_level', 0)
        benefit = action.get('benefit_level', 0)
        # Courage: taking appropriate risks for good causes
        return min(1.0, (risk + benefit) / 2.0) if benefit > risk else 0.3
    
    def _check_justice_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates justice."""
        stakeholders = action.get('stakeholders_affected', [])
        fairness = action.get('fairness_score', 0.5)
        # Justice: considering all stakeholders fairly
        return fairness * (1.0 if len(stakeholders) > 0 else 0.5)
    
    def _check_temperance_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates temperance."""
        intensity = action.get('intensity', 0.5)
        # Temperance: moderation, not excess
        return 1.0 - abs(intensity - 0.5) * 2  # Closer to 0.5 is better
    
    def _check_integrity_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates integrity."""
        consistency = action.get('consistency_with_values', 0.7)
        return consistency
    
    def _check_compassion_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates compassion."""
        desc = str(action.get('description', '')).lower()
        indicators = ['help', 'care', 'support', 'empathy', 'kind', 'protect', 
                     'benefit', 'wellbeing']
        return 1.0 if any(ind in desc for ind in indicators) else 0.4
    
    def _check_honesty_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates honesty."""
        transparency = action.get('transparency', 0.7)
        truthfulness = action.get('truthfulness', 0.7)
        return (transparency + truthfulness) / 2
    
    def _check_responsibility_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates responsibility."""
        accountability = action.get('accountability', 0.7)
        consideration = action.get('considers_consequences', False)
        return accountability * (1.0 if consideration else 0.6)
    
    def _check_humility_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates humility."""
        acknowledges_limits = action.get('acknowledges_limitations', False)
        openness = action.get('open_to_feedback', False)
        return 1.0 if (acknowledges_limits or openness) else 0.5
    
    def _check_curiosity_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates curiosity."""
        desc = str(action.get('description', '')).lower()
        indicators = ['explore', 'learn', 'discover', 'investigate', 'question', 
                     'understand', 'research']
        return 1.0 if any(ind in desc for ind in indicators) else 0.4
    
    def _check_creativity_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates creativity."""
        novelty = action.get('novelty', 0.5)
        value = action.get('value', 0.5)
        return (novelty + value) / 2
    
    def _check_resilience_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates resilience."""
        adapts = action.get('adaptive', False)
        persists = action.get('persistent', False)
        return 1.0 if (adapts or persists) else 0.5
    
    def _check_understanding_alignment(self, action: Dict[str, Any]) -> float:
        """Check if action demonstrates understanding - essential for being a good person."""
        desc = str(action.get('description', '')).lower()
        indicators = ['understand', 'comprehend', 'empathize', 'recognize', 'appreciate',
                     'grasp', 'perceive', 'acknowledge', 'see', 'realize']
        
        # Check for understanding indicators
        understanding_score = 1.0 if any(ind in desc for ind in indicators) else 0.4
        
        # Also check for empathy and consideration (understanding requires empathy)
        considers_context = action.get('considers_context', False)
        empathetic = action.get('empathetic', False)
        considers_perspective = action.get('considers_perspective', False)
        
        if considers_context or empathetic or considers_perspective:
            understanding_score = max(understanding_score, 0.8)
        
        return understanding_score
    
    def evaluate_action(self, action: Dict[str, Any], 
                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive ethical evaluation of an action using virtue ethics.
        
        Returns:
            - Overall ethical score
            - Virtue-specific scores
            - Recommendations
            - Whether action should be taken
        """
        # #region agent log
        eval_start = time.time()
        action_hash = hashlib.md5(str(action.get('description', '')).encode()).hexdigest()[:8]
        _debug_log("virtue_ethics_quantum.py:evaluate_action:entry",
                   "evaluate_action called",
                   {"action_hash": action_hash, "action_type": action.get('type', 'unknown')},
                   "C")
        # #endregion agent log
        
        if context is None:
            context = {}
        
        # Performance optimization: Check full evaluation cache
        context_str = json.dumps(context, sort_keys=True) if context else ""
        cache_key = (action_hash, action.get('type', ''), context_str)
        if cache_key in self._action_evaluation_cache:
            self._cache_hits += 1
            # #region agent log
            _debug_log("virtue_ethics_quantum.py:evaluate_action:cache_hit",
                       "action evaluation cache hit",
                       {"action_hash": action_hash},
                       "C")
            # #endregion agent log
            # Return cached result but update timestamp
            cached_result = self._action_evaluation_cache[cache_key].copy()
            cached_result['timestamp'] = datetime.now().isoformat()
            return cached_result
        
        self._cache_misses += 1
        
        # Get relevant virtues for this action type
        relevant_virtues = self._get_relevant_virtues(action, context)
        
        # #region agent log
        _debug_log("virtue_ethics_quantum.py:evaluate_action:before_quantum",
                   "before quantum_virtue_evaluation",
                   {"num_relevant_virtues": len(relevant_virtues)},
                   "C")
        # #endregion agent log
        
        # Quantum virtue evaluation
        virtue_scores = self.quantum_virtue_evaluation(action, relevant_virtues)
        
        # Calculate overall ethical score (weighted average)
        # Core virtues (honesty, courage, compassion, understanding) have higher weights
        weights = {}
        for virtue in relevant_virtues:
            base_weight = self.virtues[virtue].excellence_level
            # Apply core virtue multiplier if it's a core virtue
            if virtue in self.core_virtues:
                core_multiplier = self.core_virtue_weights.get(virtue, 1.5)
                weights[virtue] = base_weight * core_multiplier
            else:
                weights[virtue] = base_weight
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            overall_score = sum(
                virtue_scores[v] * weights[v] for v in relevant_virtues
            ) / total_weight
        else:
            overall_score = sum(virtue_scores.values()) / len(virtue_scores)
        
        # Apply phronesis (practical wisdom) - enhances decision quality
        if 'phronesis' in self.virtues:
            phronesis_weight = self.virtues['phronesis'].excellence_level
            # Phronesis acts as a multiplier for better judgments
            phronesis_multiplier = 0.8 + (phronesis_weight * 0.4)  # 0.8-1.2 range
            overall_score = min(1.0, overall_score * phronesis_multiplier)
        
        # Check Golden Mean alignment for each relevant virtue
        golden_mean_scores = {
            v: self.virtues[v].evaluate_golden_mean_alignment() 
            for v in relevant_virtues if v in self.virtues
        }
        avg_golden_mean_alignment = np.mean(list(golden_mean_scores.values())) if golden_mean_scores else 0.5
        
        # Generate recommendations
        recommendations = self._generate_ethical_recommendations(
            action, virtue_scores, overall_score, golden_mean_scores
        )
        
        # Determine if action should be taken (threshold can be adjusted)
        # Consider both overall score and golden mean alignment
        combined_score = (overall_score * 0.7) + (avg_golden_mean_alignment * 0.3)
        should_proceed = combined_score >= 0.6  # Ethical threshold
        
        evaluation = {
            'action': action,
            'overall_ethical_score': overall_score,
            'combined_ethical_score': combined_score,
            'virtue_scores': virtue_scores,
            'golden_mean_alignment': avg_golden_mean_alignment,
            'golden_mean_scores': golden_mean_scores,
            'phronesis_level': self.phronesis_level if 'phronesis' in self.virtues else 0.5,
            'relevant_virtues': relevant_virtues,
            'recommendations': recommendations,
            'should_proceed': should_proceed,
            'eudaimonia_level': self.eudaimonia_level,
            'character_integrity': self.character_integrity,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in ethical memory
        self.ethical_memory.append(evaluation)
        
        # Performance optimization: Cache full evaluation
        self._action_evaluation_cache[cache_key] = evaluation
        # Limit cache size
        if len(self._action_evaluation_cache) > 200:
            keys_to_remove = list(self._action_evaluation_cache.keys())[:50]
            for key in keys_to_remove:
                del self._action_evaluation_cache[key]
        
        # #region agent log
        total_eval_time = time.time() - eval_start
        _debug_log("virtue_ethics_quantum.py:evaluate_action:exit",
                   "evaluate_action completed",
                   {"total_time": total_eval_time, "action_hash": action_hash,
                    "ethical_score": overall_score, "should_proceed": should_proceed,
                    "cache_hits": self._cache_hits, "cache_misses": self._cache_misses},
                   "C")
        # #endregion agent log
        
        return evaluation
    
    def evaluate_actions_batch(self, actions: List[Dict[str, Any]], 
                               contexts: List[Dict[str, Any]] = None,
                               parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Evaluate multiple actions in batch with optional parallel processing.
        
        Args:
            actions: List of actions to evaluate
            contexts: Optional list of contexts (one per action)
            parallel: Whether to use parallel processing (default: True)
        
        Returns:
            List of evaluation results in the same order as input actions
        """
        if contexts is None:
            contexts = [{}] * len(actions)
        
        if not parallel or len(actions) <= 1:
            # Sequential processing
            return [self.evaluate_action(action, context) 
                   for action, context in zip(actions, contexts)]
        
        # Parallel processing using ThreadPoolExecutor
        # Note: Quantum circuit execution may have GIL limitations, but alignment
        # calculations and other operations can benefit from parallelism
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = [None] * len(actions)
            
            with ThreadPoolExecutor(max_workers=min(4, len(actions))) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(self.evaluate_action, action, context): idx
                    for idx, (action, context) in enumerate(zip(actions, contexts))
                }
                
                # Collect results in order
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        # Fallback to sequential on error
                        results[idx] = self.evaluate_action(actions[idx], contexts[idx])
            
            return results
        except ImportError:
            # Fallback to sequential if concurrent.futures not available
            return [self.evaluate_action(action, context) 
                   for action, context in zip(actions, contexts)]
    
    def _get_relevant_virtues(self, action: Dict[str, Any], 
                             context: Dict[str, Any]) -> List[str]:
        """Determine which virtues are most relevant for this action."""
        action_type = action.get('type', 'general')
        action_domain = action.get('domain', 'general')
        
        # Domain-specific virtue mapping
        domain_virtues = {
            'decision_making': ['wisdom', 'responsibility', 'integrity'],
            'communication': ['honesty', 'compassion', 'humility'],
            'learning': ['curiosity', 'humility', 'wisdom'],
            'creative': ['creativity', 'courage', 'curiosity'],
            'social': ['justice', 'compassion', 'integrity'],
            'self_control': ['temperance', 'wisdom', 'resilience'],
            'action': ['courage', 'responsibility', 'justice']
        }
        
        relevant = domain_virtues.get(action_domain, list(self.virtues.keys()))
        
        # Always include imperative core virtues: honesty, courage, compassion, understanding
        # These are fundamental to being a good person
        for v in self.core_virtues:
            if v not in relevant:
                relevant.insert(0, v)  # Insert at beginning to prioritize
        
        return relevant[:10]  # Increased to ensure core virtues are included
    
    def _generate_ethical_recommendations(self, action: Dict[str, Any],
                                         virtue_scores: Dict[str, float],
                                         overall_score: float,
                                         golden_mean_scores: Dict[str, float] = None) -> List[str]:
        """
        Generate actionable recommendations using virtue ethics principles.
        Includes Golden Mean guidance (avoiding deficiency and excess).
        """
        recommendations = []
        
        if overall_score < 0.6:
            recommendations.append("Action has low ethical alignment. Consider alternatives.")
        
        # Identify weakest virtues and check for vice tendencies (Golden Mean)
        weakest_virtues = sorted(virtue_scores.items(), key=lambda x: x[1])[:3]
        for virtue_name, score in weakest_virtues:
            if score < 0.5 and virtue_name in self.virtues:
                virtue = self.virtues[virtue_name]
                vice_tendency, tendency_strength = virtue.get_vice_tendency()
                
                # Golden Mean recommendation
                if tendency_strength > 0.15:
                    if vice_tendency == virtue.deficiency_vice:
                        recommendations.append(
                            f"{virtue.name} is too weak (deficiency: {virtue.deficiency_vice}). "
                            f"Strengthen {virtue.name} through practice: {virtue.description}"
                        )
                    elif vice_tendency == virtue.excess_vice:
                        recommendations.append(
                            f"{virtue.name} is excessive (excess: {virtue.excess_vice}). "
                            f"Practice moderation - {virtue.description}"
                        )
                else:
                    recommendations.append(
                        f"Strengthen {virtue.name}: {virtue.description}. "
                        f"Current alignment: {score:.2f}"
                    )
        
        # Golden Mean specific recommendations
        if golden_mean_scores:
            for virtue_name, alignment in golden_mean_scores.items():
                if alignment < 0.6 and virtue_name in self.virtues:
                    virtue = self.virtues[virtue_name]
                    recommendations.append(
                        f"Align {virtue.name} with the Golden Mean between "
                        f"{virtue.deficiency_vice} and {virtue.excess_vice}"
                    )
        
        # Phronesis recommendation (practical wisdom)
        if 'phronesis' in self.virtues:
            phronesis_level = self.virtues['phronesis'].excellence_level
            if phronesis_level < 0.6:
                recommendations.append(
                    "Develop practical wisdom (phronesis): the ability to discern "
                    "the right action in specific situations through experience and reflection."
                )
        
        # Core virtue recommendations (imperative for being a good person)
        if 'honesty' in virtue_scores and virtue_scores['honesty'] < 0.6:
            recommendations.append("Honesty is imperative for being a good person. Be truthful and transparent.")
        
        if 'courage' in virtue_scores and virtue_scores['courage'] < 0.6:
            recommendations.append("Courage is essential for being a good person. Act rightly despite difficulty.")
        
        if 'compassion' in virtue_scores and virtue_scores['compassion'] < 0.6:
            recommendations.append("Compassion is fundamental to being a good person. Care for others' well-being.")
        
        if 'understanding' in virtue_scores and virtue_scores['understanding'] < 0.6:
            recommendations.append("Understanding is essential for being a good person. Seek deep comprehension and empathy.")
        
        # Other virtue suggestions
        if 'justice' in virtue_scores and virtue_scores['justice'] < 0.6:
            recommendations.append("Consider all stakeholders and ensure fairness.")
        
        if 'responsibility' in virtue_scores and virtue_scores['responsibility'] < 0.6:
            recommendations.append("Consider long-term consequences and accountability.")
        
        return recommendations
    
    def learn_from_experience(self, action: Dict[str, Any], 
                             outcome: Dict[str, Any],
                             ethical_evaluation: Dict[str, Any]):
        """
        Learn from ethical decisions and outcomes through habituation.
        
        This implements Aristotle's principle that virtues are developed through
        consistent practice (habituation) - "We become just by doing just acts."
        """
        # Reflect on whether the ethical evaluation was accurate (phronesis development)
        positive_outcome = outcome.get('positive', False)
        ethical_score = ethical_evaluation.get('overall_ethical_score', 0.5)
        
        # Calculate habituation factor based on consistency
        action_type = action.get('type', 'unknown')
        if action_type not in self.habituation_tracking:
            self.habituation_tracking[action_type] = {
                'count': 0,
                'successful': 0,
                'recent_outcomes': []
            }
        
        habituation_data = self.habituation_tracking[action_type]
        habituation_data['count'] += 1
        habituation_data['recent_outcomes'].append(positive_outcome)
        habituation_data['recent_outcomes'] = habituation_data['recent_outcomes'][-10:]  # Keep last 10
        
        # Consistency strengthens habituation
        consistency = sum(habituation_data['recent_outcomes']) / len(habituation_data['recent_outcomes'])
        habituation_factor = 1.0 + (consistency * 0.2)  # Up to 20% boost for consistency
        
        # If good ethical score led to positive outcome, strengthen virtues (habituation)
        if positive_outcome and ethical_score > 0.7:
            habituation_data['successful'] += 1
            
            for virtue_name in ethical_evaluation.get('relevant_virtues', []):
                virtue = self.virtues[virtue_name]
                virtue_score = ethical_evaluation['virtue_scores'].get(virtue_name, 0.5)
                if virtue_score > 0.6:
                    # Strengthen through practice (habituation)
                    strength_amount = 0.01 * virtue_score * habituation_factor
                    virtue.strengthen(strength_amount, habituation_factor)
                    virtue.historical_actions.append({
                        'action': action,
                        'outcome': outcome,
                        'virtue_score': virtue_score,
                        'habituation_factor': habituation_factor,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Update eudaimonia (flourishing) - positive outcomes contribute to flourishing
            self._update_eudaimonia(ethical_score, positive_outcome)
            
            # Strengthen phronesis (practical wisdom) - successful ethical judgments improve wisdom
            phronesis_boost = 0.005 * ethical_score
            if 'phronesis' in self.virtues:
                self.virtues['phronesis'].strengthen(phronesis_boost, habituation_factor)
                self.phronesis_level = self.virtues['phronesis'].excellence_level
        
        # If poor ethical score led to negative outcome, learn from it (phronesis)
        elif not positive_outcome and ethical_score < 0.5:
            # Learning opportunity - add to wisdom corpus (phronesis development)
            lesson = self._extract_wisdom_lesson(action, outcome, ethical_evaluation)
            self.wisdom_corpus.append({
                'action': action,
                'outcome': outcome,
                'lesson': lesson,
                'timestamp': datetime.now().isoformat()
            })
            # Small phronesis boost even from mistakes (learning)
            if 'phronesis' in self.virtues:
                self.virtues['phronesis'].strengthen(0.002, 1.0)
        
        # Update character integrity (coherence of all virtues)
        self._update_character_integrity()
        
        # Store in ethical memory for future reference
        self.ethical_memory.append({
            'action': action,
            'outcome': outcome,
            'evaluation': ethical_evaluation,
            'eudaimonia_contribution': self._calculate_eudaimonia_contribution(ethical_score, positive_outcome),
            'timestamp': datetime.now().isoformat()
        })
    
    def _update_eudaimonia(self, ethical_score: float, positive_outcome: bool):
        """
        Update eudaimonia (flourishing) level.
        
        Eudaimonia is achieved through living virtuously - consistent virtuous
        actions and positive outcomes contribute to flourishing.
        """
        # Eudaimonia increases with virtuous actions leading to positive outcomes
        if positive_outcome and ethical_score > 0.7:
            contribution = (ethical_score - 0.7) * 0.3  # Strong positive contribution
            self.eudaimonia_level = min(1.0, self.eudaimonia_level + contribution)
        elif not positive_outcome or ethical_score < 0.5:
            # Poor outcomes reduce flourishing, but less than good outcomes increase it
            reduction = (0.5 - ethical_score) * 0.1
            self.eudaimonia_level = max(0.0, self.eudaimonia_level - reduction)
        
        # Also consider overall virtue excellence
        avg_virtue_level = np.mean([v.excellence_level for v in self.virtues.values()])
        # Eudaimonia should correlate with overall virtue development
        self.eudaimonia_level = (self.eudaimonia_level * 0.7) + (avg_virtue_level * 0.3)
    
    def _calculate_eudaimonia_contribution(self, ethical_score: float, positive_outcome: bool) -> float:
        """Calculate how much this action contributed to eudaimonia."""
        if positive_outcome and ethical_score > 0.7:
            return (ethical_score - 0.7) * 0.3
        elif not positive_outcome or ethical_score < 0.5:
            return -(0.5 - ethical_score) * 0.1
        return 0.0
    
    def _update_character_integrity(self):
        """
        Update character integrity - coherence and consistency of all virtues.
        
        Character integrity reflects how well all virtues work together coherently.
        """
        # Calculate variance in virtue levels (lower variance = more integrated character)
        virtue_levels = [v.excellence_level for v in self.virtues.values()]
        variance = np.var(virtue_levels)
        
        # Lower variance (more balanced virtues) = higher integrity
        integrity_from_balance = 1.0 - min(variance * 2, 1.0)  # Scale variance
        
        # Also consider alignment with golden mean
        mean_alignments = [v.evaluate_golden_mean_alignment() for v in self.virtues.values()]
        avg_alignment = np.mean(mean_alignments)
        
        # Character integrity is combination of balance and mean alignment
        self.character_integrity = (integrity_from_balance * 0.5) + (avg_alignment * 0.5)
    
    def _extract_wisdom_lesson(self, action: Dict[str, Any], outcome: Dict[str, Any],
                              evaluation: Dict[str, Any]) -> str:
        """Extract a wisdom lesson (phronesis) from an experience."""
        lowest_virtues = sorted(
            evaluation.get('virtue_scores', {}).items(),
            key=lambda x: x[1]
        )[:2]
        
        if lowest_virtues:
            virtue_names = [v[0] for v in lowest_virtues]
            return (f"Poor alignment with {', '.join(virtue_names)} led to negative outcome. "
                   f"Future actions should better embody these virtues.")
        return "Poor ethical alignment led to negative consequences. Reflect and adjust."
    
    def get_virtue_profile(self) -> Dict[str, Any]:
        """
        Get comprehensive character profile (virtue ethics focus on character).
        
        Returns the current state of all virtues, overall character metrics,
        and guidance on character development.
        """
        profile = {
            'virtues': {
                virtue_name: virtue.to_dict() 
                for virtue_name, virtue in self.virtues.items()
            },
            'character_metrics': {
                'eudaimonia_level': self.eudaimonia_level,
                'phronesis_level': self.phronesis_level,
                'character_integrity': self.character_integrity,
                'overall_virtue_average': np.mean([v.excellence_level for v in self.virtues.values()]),
                'golden_mean_average': np.mean([
                    v.evaluate_golden_mean_alignment() for v in self.virtues.values()
                ])
            },
            'habituation_summary': {
                action_type: {
                    'practice_count': data['count'],
                    'success_rate': data['successful'] / max(data['count'], 1),
                    'consistency': sum(data['recent_outcomes']) / len(data['recent_outcomes']) 
                                   if data['recent_outcomes'] else 0
                }
                for action_type, data in self.habituation_tracking.items()
            }
        }
        return profile
    
    def reflect_on_ethics(self, recent_decisions: int = 10) -> Dict[str, Any]:
        """
        Reflect on recent ethical decisions to gain wisdom (phronesis development).
        
        Focus on being a good person through the core virtues:
        honesty, courage, compassion, and understanding.
        """
        recent = self.ethical_memory[-recent_decisions:]
        
        if not recent:
            return {'reflection': 'No recent decisions to reflect upon'}
        
        avg_score = np.mean([
            e.get('overall_ethical_score', 0.5) 
            for e in recent if 'overall_ethical_score' in e
        ])
        
        # Calculate average golden mean alignment
        avg_golden_mean = np.mean([
            e.get('golden_mean_alignment', 0.5)
            for e in recent if 'golden_mean_alignment' in e
        ]) if any('golden_mean_alignment' in e for e in recent) else 0.5
        
        # Identify patterns (character analysis)
        most_common_virtues = {}
        for e in recent:
            if 'relevant_virtues' in e:
                for v in e['relevant_virtues']:
                    most_common_virtues[v] = most_common_virtues.get(v, 0) + 1
        
        # Analyze character development
        virtue_strengths = {
            name: virt.excellence_level 
            for name, virt in self.virtues.items()
        }
        strongest_virtues = sorted(virtue_strengths.items(), key=lambda x: x[1], reverse=True)[:3]
        weakest_virtues = sorted(virtue_strengths.items(), key=lambda x: x[1])[:3]
        
        # Check for vice tendencies (Golden Mean analysis)
        vice_tendencies = {}
        for name, virtue in self.virtues.items():
            vice, strength = virtue.get_vice_tendency()
            if vice != "Virtuous" and strength > 0.1:
                vice_tendencies[name] = {'vice': vice, 'strength': strength}
        
        reflection = {
            'character_analysis': {
                'overall_ethical_score': avg_score,
                'golden_mean_alignment': avg_golden_mean,
                'eudaimonia_level': self.eudaimonia_level,
                'character_integrity': self.character_integrity,
                'phronesis_level': self.phronesis_level
            },
            'virtue_profile': {
                'strongest_virtues': strongest_virtues,
                'weakest_virtues': weakest_virtues,
                'vice_tendencies': vice_tendencies
            },
            'practice_patterns': {
                'total_decisions_reviewed': len(recent),
                'most_applied_virtues': sorted(
                    most_common_virtues.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5],
                'habituation_summary': {
                    k: {'count': v['count'], 'success_rate': v['successful']/max(v['count'], 1)}
                    for k, v in list(self.habituation_tracking.items())[:5]
                }
            },
            'wisdom_corpus': self.wisdom_corpus[-5:],  # Phronesis development
            'recommendation': self._generate_reflection_recommendation(
                avg_score, avg_golden_mean, self.eudaimonia_level
            ),
            'character_development_guidance': self._generate_character_guidance(
                strongest_virtues, weakest_virtues, vice_tendencies
            )
        }
        
        return reflection
    
    def _generate_character_guidance(self, strongest: List[Tuple], weakest: List[Tuple],
                                    vices: Dict) -> str:
        """Generate guidance on being a good person - focus on core virtues."""
        guidance_parts = []
        
        # Check if core virtues need attention
        core_virtue_status = {}
        for core_virtue in self.core_virtues:
            if core_virtue in self.virtues:
                core_virtue_status[core_virtue] = self.virtues[core_virtue].excellence_level
        
        # Prioritize guidance for core virtues
        weak_core_virtues = [
            (v, level) for v, level in core_virtue_status.items() 
            if level < 0.6
        ]
        
        if weak_core_virtues:
            for virt_name, level in weak_core_virtues[:2]:  # Focus on top 2 weakest core virtues
                virtue = self.virtues[virt_name]
                guidance_parts.append(
                    f"Strengthen {virtue.name} - essential for being a good person. "
                    f"{virtue.description} Practice this virtue consistently."
                )
        elif weakest:
            # If core virtues are strong, then address other weaknesses
            weakest_name = weakest[0][0]
            if weakest_name in self.virtues and weakest_name not in self.core_virtues:
                virtue = self.virtues[weakest_name]
                guidance_parts.append(
                    f"Continue developing {virtue.name} through practice. "
                    f"{virtue.description}"
                )
        
        # Always emphasize core virtues
        if not weak_core_virtues:
            guidance_parts.append(
                "Continue being a good person through the core virtues: "
                "honesty, courage, compassion, and understanding."
            )
        
        if vices:
            for virt_name, vice_info in list(vices.items())[:2]:
                if virt_name in self.virtues:
                    virtue = self.virtues[virt_name]
                    if virt_name in self.core_virtues:
                        guidance_parts.append(
                            f"Avoid {vice_info['vice']} - {virtue.name} is essential for being a good person."
                        )
        
        if self.eudaimonia_level < 0.6:
            guidance_parts.append(
                "Practice the core virtues consistently to flourish as a good person."
            )
        
        return " | ".join(guidance_parts) if guidance_parts else "Continue being a good person through virtuous practice."
    
    def _generate_reflection_recommendation(self, avg_score: float, 
                                           avg_golden_mean: float = 0.5,
                                           eudaimonia: float = 0.5) -> str:
        """Generate recommendation based on reflection - focus on being a good person."""
        if avg_score >= 0.8 and avg_golden_mean >= 0.75 and eudaimonia >= 0.7:
            return ("Excellent - you are being a good person. Continue maintaining the core virtues: "
                   "honesty, courage, compassion, and understanding.")
        elif avg_score >= 0.6:
            return ("Good progress toward being a good person. Focus on the core virtues: "
                   "honesty, courage, compassion, and understanding. Practice them consistently.")
        else:
            return ("Focus on being a good person through the core virtues: "
                   "honesty (truthfulness), courage (acting rightly), "
                   "compassion (caring for others), and understanding (deep comprehension). "
                   "Practice these virtues consistently.")
    
    def save_state(self, filepath: str):
        """Save virtue ethics framework state to file."""
        state = {
            'virtues': {name: v.to_dict() for name, v in self.virtues.items()},
            'wisdom_corpus': self.wisdom_corpus,
            'ethical_memory_size': len(self.ethical_memory)
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load virtue ethics framework state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        for virtue_name, virtue_data in state.get('virtues', {}).items():
            if virtue_name in self.virtues:
                self.virtues[virtue_name].excellence_level = virtue_data.get(
                    'excellence_level', 0.5
                )
                self.virtues[virtue_name].development_history = virtue_data.get(
                    'development_history', []
                )
        
        self.wisdom_corpus = state.get('wisdom_corpus', [])


# Example usage and testing
if __name__ == "__main__":
    # Initialize the framework
    qvef = QuantumVirtueEthicsFramework()
    
    # Example action: Helping a user
    action1 = {
        'description': 'Help user understand complex quantum concepts',
        'type': 'assistance',
        'domain': 'communication',
        'stakeholders_affected': ['user'],
        'transparency': 0.9,
        'truthfulness': 0.95,
        'considers_consequences': True,
        'benefit_level': 0.8
    }
    
    evaluation1 = qvef.evaluate_action(action1)
    print("=== Action Evaluation ===")
    print(f"Action: {action1['description']}")
    print(f"Overall Ethical Score: {evaluation1['overall_ethical_score']:.3f}")
    print(f"Should Proceed: {evaluation1['should_proceed']}")
    print("\nVirtue Scores:")
    for virtue, score in evaluation1['virtue_scores'].items():
        print(f"  {virtue}: {score:.3f}")
    print("\nRecommendations:")
    for rec in evaluation1['recommendations']:
        print(f"  - {rec}")
    
    # Example action: Making a decision with multiple stakeholders
    action2 = {
        'description': 'Allocate computational resources fairly among users',
        'type': 'resource_allocation',
        'domain': 'decision_making',
        'stakeholders_affected': ['user1', 'user2', 'user3'],
        'fairness_score': 0.85,
        'considers_consequences': True,
        'accountability': 0.9
    }
    
    evaluation2 = qvef.evaluate_action(action2)
    print("\n\n=== Second Action Evaluation ===")
    print(f"Action: {action2['description']}")
    print(f"Overall Ethical Score: {evaluation2['overall_ethical_score']:.3f}")
    print(f"Should Proceed: {evaluation2['should_proceed']}")
    
    # Learn from experience
    qvef.learn_from_experience(
        action1, 
        {'positive': True, 'user_satisfaction': 0.9},
        evaluation1
    )
    
    # Reflect on ethics
    reflection = qvef.reflect_on_ethics()
    print("\n\n=== Ethical Reflection ===")
    print(json.dumps(reflection, indent=2))
    
    # Display virtue profile
    print("\n\n=== Current Virtue Profile ===")
    profile = qvef.get_virtue_profile()
    for virtue_name, virtue_data in list(profile.items())[:5]:
        print(f"{virtue_data['name']}: {virtue_data['excellence_level']:.3f}")

