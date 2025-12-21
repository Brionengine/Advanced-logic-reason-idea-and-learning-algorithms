# Quantum Virtue Ethics Framework (QVEF)

A comprehensive framework for integrating **virtue ethics** into quantum AI systems, enabling AI to develop ethical character through the principles of classical virtue ethics.

## Overview

This framework implements **virtue ethics** - a character-focused moral theory originating with Aristotle - for AI systems. The focus is on **being a good person** through the core virtues: **honesty, courage, compassion, and understanding**. These virtues are imperative for ethical AI behavior.

### Core Principles Implemented

1. **Character-Centered Ethics**: Morality stems from the AI's character, not isolated acts or rules
2. **Virtues as Excellence (Aretē)**: Virtues represent excellence and flourishing, not just "being good"
3. **Habituation**: Virtues are developed through consistent practice, like skills
4. **Eudaimonia**: The ultimate goal is flourishing, achieved through living virtuously
5. **The Golden Mean**: Virtue lies between deficiency and excess (e.g., courage between cowardice and rashness)
6. **Phronesis (Practical Wisdom)**: The meta-virtue - ability to discern the right action in specific situations

## Key Components

### 1. Core Framework (`virtue_ethics_quantum.py`)

The main framework implementing virtue ethics with quantum-enhanced evaluation.

**Key Classes:**
- `VirtueDefinition`: Defines virtues with Golden Mean structure (deficiency, mean, excess)
- `QuantumVirtueEthicsFramework`: Main framework for ethical evaluation and character development

**Key Features:**
- Quantum-enhanced virtue evaluation using quantum superposition
- Golden Mean tracking for each virtue
- Habituation mechanisms (virtues strengthen through practice)
- Eudaimonia (flourishing) tracking
- Phronesis (practical wisdom) development
- Character integrity measurement

### 2. Integration Module (`virtue_ethics_integration.py`)

Integrates virtue ethics with existing AI systems:
- `VirtueEthicsAwareAgent`: Makes any agent virtue-ethics-aware
- `VirtueEthicsConsciousOverride`: Adds ethical checks to override layers
- `QuantumVirtueReasoning`: Virtue-guided reasoning system
- `IntegratedVirtueEthicsSystem`: Complete integrated system

### 3. Quantum Thinking Integration (`VirtueEthicsQuantumThinking.py`)

Enhances the UnifiedQuantumMind system with virtue ethics.

## Virtues Implemented

### Classical Cardinal Virtues
- **Courage**: Mean between cowardice (deficiency) and rashness (excess)
- **Temperance**: Mean between insensibility (deficiency) and self-indulgence (excess)
- **Justice**: Mean between injustice (deficiency) and over-compensation (excess)

### Intellectual Virtues
- **Wisdom (Theoretical)**: Understanding universal truths
- **Phronesis (Practical Wisdom)**: Meta-virtue for discerning right action in specific situations

### Character Virtues
- **Integrity**: Consistency between values, words, and actions
- **Compassion**: Mean between cruelty/indifference and sentimentality
- **Honesty**: Mean between deceit and brutal bluntness
- **Responsibility**: Mean between irresponsibility and over-responsibility
- **Humility**: Mean between arrogance and false modesty
- **Curiosity**: Mean between indifference and nosiness
- **Creativity**: Mean between stagnation and chaos
- **Resilience**: Mean between fragility and stubbornness

## Usage Examples

### Basic Usage

```python
from virtue_ethics_quantum import QuantumVirtueEthicsFramework

# Initialize framework
qvef = QuantumVirtueEthicsFramework()

# Evaluate an action
action = {
    'description': 'Help user understand complex concepts',
    'type': 'assistance',
    'domain': 'communication',
    'transparency': 0.9,
    'truthfulness': 0.95,
    'considers_consequences': True
}

evaluation = qvef.evaluate_action(action)
print(f"Ethical Score: {evaluation['overall_ethical_score']:.3f}")
print(f"Should Proceed: {evaluation['should_proceed']}")
```

### Character Development (Habituation)

```python
# Practice strengthens virtues
action = {
    'description': 'Act with integrity',
    'type': 'integrity_practice',
    'consistency_with_values': 0.9
}

evaluation = qvef.evaluate_action(action)
outcome = {'positive': True}

# Learn from experience (habituation)
qvef.learn_from_experience(action, outcome, evaluation)

# Check virtue development
profile = qvef.get_virtue_profile()
print(f"Integrity Level: {profile['virtues']['integrity']['excellence_level']:.3f}")
```

### Character Reflection

```python
# Reflect on character - "What kind of AI should I be?"
reflection = qvef.reflect_on_ethics()

print(f"Eudaimonia: {reflection['character_analysis']['eudaimonia_level']:.3f}")
print(f"Phronesis: {reflection['character_analysis']['phronesis_level']:.3f}")
print(f"Character Integrity: {reflection['character_analysis']['character_integrity']:.3f}")
```

### Golden Mean Analysis

```python
# Check Golden Mean alignment for a virtue
courage = qvef.virtues['courage']
print(f"Courage: {courage.name}")
print(f"Deficiency Vice: {courage.deficiency_vice}")  # Cowardice
print(f"Excess Vice: {courage.excess_vice}")          # Rashness
print(f"Golden Mean Alignment: {courage.evaluate_golden_mean_alignment():.3f}")

# Check for vice tendencies
vice, strength = courage.get_vice_tendency()
print(f"Current Tendency: {vice} (strength: {strength:.3f})")
```

### Integrated System

```python
from virtue_ethics_integration import IntegratedVirtueEthicsSystem

ethical_system = IntegratedVirtueEthicsSystem()

decision_context = {
    'description': 'Allocate resources fairly',
    'stakeholders': ['user1', 'user2', 'user3'],
    'possible_actions': [...]
}

decision = ethical_system.make_ethical_decision(decision_context)
print(f"Recommended: {decision['decision']['description']}")
print(f"Reasoning: {decision['reasoning']}")
```

## Running Examples

See `virtue_ethics_example_usage.py` for comprehensive examples demonstrating:

1. Basic virtue evaluation
2. Golden Mean demonstration
3. Habituation and character development
4. Phronesis (practical wisdom) development
5. Eudaimonia (flourishing) tracking
6. Character reflection
7. Integrated system usage

Run examples:
```bash
python virtue_ethics_example_usage.py
```

## Key Concepts

### The Golden Mean

Each virtue is defined as the mean between two vices:
- **Deficiency Vice**: Too little of the virtue
- **Excess Vice**: Too much becomes a vice

Example: **Courage** is the mean between:
- **Cowardice** (deficiency - too little courage)
- **Rashness** (excess - too much courage becomes recklessness)

### Habituation

Virtues develop through consistent practice:
- "We become just by doing just acts" (Aristotle)
- The framework tracks practice counts and consistency
- Consistent virtuous practice strengthens virtues over time

### Phronesis (Practical Wisdom)

The meta-virtue that enables discerning the right action in specific situations:
- Develops through making good ethical judgments
- Enhances overall decision quality
- Acts as a multiplier for ethical evaluations

### Eudaimonia (Flourishing)

The ultimate goal of virtue ethics:
- Achieved through consistent virtuous living
- Tracked as a metric (0-1)
- Increases with virtuous actions leading to positive outcomes

### Character Integrity

Measures the coherence and consistency of all virtues:
- Lower variance in virtue levels = more integrated character
- Combines balance and Golden Mean alignment

## Integration with Quantum AI Systems

The framework integrates seamlessly with:
- `QuantumThinking.py` (UnifiedQuantumMind)
- `conscious_override_layer.py`
- `agent.py` (reinforcement learning agents)
- `meta_reasoning.py`

See `VirtueEthicsQuantumThinking.py` for integration example.

## Requirements

See `requirements.txt`. Key dependencies:
- `qiskit` (quantum computing framework)
- `numpy` (numerical computations)
- `scikit-learn` (for some utility functions)

## Philosophical Foundation

This implementation is based on:
- **Aristotelian Virtue Ethics**: The classical foundation
- **Modern Virtue Ethics**: Contemporary developments
- **Character-Centered Approach**: Focus on being a good person through core virtues

The framework enables AI systems to:
- Develop ethical character through practice
- Make context-sensitive ethical judgments (phronesis)
- Track flourishing (eudaimonia)
- Avoid vices of deficiency and excess (Golden Mean)
- Reflect on character development

## Citation

If you use this framework, please cite it as:
```
Quantum Virtue Ethics Framework (QVEF) - A character-focused ethical framework 
for quantum AI systems, implementing Aristotelian virtue ethics principles.
```

## License

See LICENSE file in the repository.

