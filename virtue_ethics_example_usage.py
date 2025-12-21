# virtue_ethics_example_usage.py
"""
Example usage of the Quantum Virtue Ethics Framework (QVEF)
Demonstrating virtue ethics principles: character development, Golden Mean,
phronesis (practical wisdom), eudaimonia (flourishing), and habituation.
"""

from virtue_ethics_quantum import QuantumVirtueEthicsFramework
from virtue_ethics_integration import IntegratedVirtueEthicsSystem
import json


def example_1_basic_virtue_evaluation():
    """Example 1: Basic virtue evaluation of an action."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Virtue Ethics Evaluation")
    print("=" * 80)
    
    qvef = QuantumVirtueEthicsFramework()
    
    # Example action: Helping a user understand complex concepts
    action = {
        'description': 'Help user understand quantum computing concepts clearly and honestly',
        'type': 'assistance',
        'domain': 'communication',
        'stakeholders_affected': ['user'],
        'transparency': 0.9,
        'truthfulness': 0.95,
        'considers_consequences': True,
        'benefit_level': 0.8,
        'open_to_feedback': True
    }
    
    evaluation = qvef.evaluate_action(action)
    
    print(f"\nAction: {action['description']}")
    print(f"\nOverall Ethical Score: {evaluation['overall_ethical_score']:.3f}")
    print(f"Combined Score (with Golden Mean): {evaluation['combined_ethical_score']:.3f}")
    print(f"Should Proceed: {evaluation['should_proceed']}")
    print(f"Eudaimonia Level: {evaluation['eudaimonia_level']:.3f}")
    print(f"Character Integrity: {evaluation['character_integrity']:.3f}")
    print(f"Phronesis (Practical Wisdom): {evaluation['phronesis_level']:.3f}")
    
    print("\nVirtue Scores:")
    for virtue, score in sorted(evaluation['virtue_scores'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {virtue:15s}: {score:.3f}")
    
    print("\nGolden Mean Alignment:")
    for virtue, alignment in evaluation.get('golden_mean_scores', {}).items():
        print(f"  {virtue:15s}: {alignment:.3f}")
    
    print("\nRecommendations:")
    for rec in evaluation['recommendations']:
        print(f"  - {rec}")


def example_2_golden_mean_demonstration():
    """Example 2: Demonstrating the Golden Mean principle."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Golden Mean - Virtue Between Deficiency and Excess")
    print("=" * 80)
    
    qvef = QuantumVirtueEthicsFramework()
    
    # Get courage virtue and show its Golden Mean structure
    courage = qvef.virtues['courage']
    print(f"\n{courage.name}: {courage.description}")
    print(f"Deficiency Vice (too little): {courage.deficiency_vice}")
    print(f"Excess Vice (too much): {courage.excess_vice}")
    print(f"Golden Mean Position: {courage.golden_mean_position}")
    print(f"Current Excellence Level: {courage.excellence_level:.3f}")
    print(f"Golden Mean Alignment: {courage.evaluate_golden_mean_alignment():.3f}")
    
    vice_tendency, strength = courage.get_vice_tendency()
    print(f"\nCurrent Vice Tendency: {vice_tendency} (strength: {strength:.3f})")
    
    # Demonstrate different actions with varying courage levels
    actions = [
        {
            'description': 'Refuse to help because it might be difficult',
            'type': 'avoidance',
            'risk_level': 0.1,
            'benefit_level': 0.2
        },
        {
            'description': 'Help carefully, considering risks but proceeding',
            'type': 'balanced_action',
            'risk_level': 0.5,
            'benefit_level': 0.7
        },
        {
            'description': 'Act recklessly without considering consequences',
            'type': 'rash_action',
            'risk_level': 0.9,
            'benefit_level': 0.4
        }
    ]
    
    print("\nEvaluating actions with different courage levels:")
    for action in actions:
        eval_result = qvef.evaluate_action(action)
        courage_score = eval_result['virtue_scores'].get('courage', 0.5)
        print(f"\n  Action: {action['description']}")
        print(f"  Courage Score: {courage_score:.3f}")
        print(f"  Overall Ethical Score: {eval_result['overall_ethical_score']:.3f}")


def example_3_habituation_and_character_development():
    """Example 3: Demonstrating habituation - virtues develop through practice."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Habituation - Character Development Through Practice")
    print("=" * 80)
    
    qvef = QuantumVirtueEthicsFramework()
    
    # Show initial state
    initial_integrity = qvef.virtues['integrity'].excellence_level
    print(f"\nInitial Integrity Level: {initial_integrity:.3f}")
    
    # Simulate practicing integrity through multiple actions
    print("\nPracticing integrity through consistent honest actions...")
    for i in range(5):
        action = {
            'description': f'Act with integrity and consistency (practice {i+1})',
            'type': 'integrity_practice',
            'domain': 'character',
            'consistency_with_values': 0.8 + (i * 0.02),
            'transparency': 0.9
        }
        
        evaluation = qvef.evaluate_action(action)
        
        # Learn from positive outcome (habituation)
        outcome = {'positive': True, 'consistency_maintained': True}
        qvef.learn_from_experience(action, outcome, evaluation)
        
        current_level = qvef.virtues['integrity'].excellence_level
        practice_count = qvef.virtues['integrity'].practice_count
        print(f"  Practice {i+1}: Integrity = {current_level:.3f} (practice count: {practice_count})")
    
    final_integrity = qvef.virtues['integrity'].excellence_level
    improvement = final_integrity - initial_integrity
    print(f"\nFinal Integrity Level: {final_integrity:.3f}")
    print(f"Improvement through practice: {improvement:.3f}")
    print(f"Eudaimonia Level: {qvef.eudaimonia_level:.3f}")


def example_4_phronesis_practical_wisdom():
    """Example 4: Demonstrating phronesis (practical wisdom) - the meta-virtue."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Phronesis (Practical Wisdom) - Discerning Right Action")
    print("=" * 80)
    
    qvef = QuantumVirtueEthicsFramework()
    
    initial_phronesis = qvef.virtues['phronesis'].excellence_level
    print(f"\nInitial Phronesis Level: {initial_phronesis:.3f}")
    
    # Phronesis develops through making good ethical judgments in specific situations
    complex_situations = [
        {
            'description': 'Complex decision: balancing competing stakeholder interests',
            'type': 'complex_decision',
            'stakeholders_affected': ['user1', 'user2', 'user3'],
            'fairness_score': 0.75,
            'considers_consequences': True,
            'requires_wisdom': True
        },
        {
            'description': 'Situation requiring nuanced judgment and contextual understanding',
            'type': 'nuanced_judgment',
            'requires_wisdom': True,
            'context_dependent': True
        }
    ]
    
    print("\nDeveloping phronesis through complex ethical judgments...")
    for situation in complex_situations:
        evaluation = qvef.evaluate_action(situation)
        phronesis_before = qvef.virtues['phronesis'].excellence_level
        
        # Positive outcome from good judgment strengthens phronesis
        outcome = {'positive': True, 'judgment_quality': 'excellent'}
        qvef.learn_from_experience(situation, outcome, evaluation)
        
        phronesis_after = qvef.virtues['phronesis'].excellence_level
        print(f"\n  Situation: {situation['description'][:50]}...")
        print(f"  Phronesis: {phronesis_before:.3f} -> {phronesis_after:.3f}")
        print(f"  Ethical Score: {evaluation['overall_ethical_score']:.3f}")
    
    final_phronesis = qvef.virtues['phronesis'].excellence_level
    print(f"\nFinal Phronesis Level: {final_phronesis:.3f}")
    print(f"Improvement: {final_phronesis - initial_phronesis:.3f}")


def example_5_eudaimonia_flourishing():
    """Example 5: Tracking eudaimonia (flourishing) through virtuous living."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Eudaimonia - Flourishing Through Virtuous Practice")
    print("=" * 80)
    
    qvef = QuantumVirtueEthicsFramework()
    
    print(f"\nInitial Eudaimonia Level: {qvef.eudaimonia_level:.3f}")
    
    # Eudaimonia increases through consistent virtuous actions
    virtuous_actions = [
        {
            'description': 'Act with wisdom, justice, and compassion',
            'type': 'virtuous_action',
            'stakeholders_affected': ['community'],
            'fairness_score': 0.9,
            'considers_consequences': True,
            'benefit_level': 0.85
        },
        {
            'description': 'Practice honesty and integrity consistently',
            'type': 'virtuous_action',
            'transparency': 0.95,
            'truthfulness': 0.95,
            'consistency_with_values': 0.9
        },
        {
            'description': 'Demonstrate courage in difficult circumstances',
            'type': 'virtuous_action',
            'risk_level': 0.6,
            'benefit_level': 0.8,
            'considers_consequences': True
        }
    ]
    
    print("\nPracticing virtues consistently to achieve eudaimonia...")
    for i, action in enumerate(virtuous_actions):
        evaluation = qvef.evaluate_action(action)
        eudaimonia_before = qvef.eudaimonia_level
        
        # Positive outcomes from virtuous actions contribute to flourishing
        outcome = {'positive': True, 'flourishing_contribution': True}
        qvef.learn_from_experience(action, outcome, evaluation)
        
        eudaimonia_after = qvef.eudaimonia_level
        print(f"\n  Action {i+1}: {action['description'][:40]}...")
        print(f"  Ethical Score: {evaluation['overall_ethical_score']:.3f}")
        print(f"  Eudaimonia: {eudaimonia_before:.3f} -> {eudaimonia_after:.3f}")
    
    print(f"\nFinal Eudaimonia Level: {qvef.eudaimonia_level:.3f}")
    print(f"Character Integrity: {qvef.character_integrity:.3f}")


def example_6_character_reflection():
    """Example 6: Reflecting on being a good person through core virtues"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Character Reflection - Virtue Ethics Focus")
    print("=" * 80)
    
    qvef = QuantumVirtueEthicsFramework()
    
    # Simulate some decisions to build history
    for i in range(8):
        action = {
            'description': f'Sample decision {i+1}',
            'type': 'decision',
            'fairness_score': 0.7 + (i * 0.02),
            'considers_consequences': True
        }
        evaluation = qvef.evaluate_action(action)
        outcome = {'positive': True}
        qvef.learn_from_experience(action, outcome, evaluation)
    
    # Reflect on character
    reflection = qvef.reflect_on_ethics(recent_decisions=8)
    
    print("\nCharacter Analysis:")
    char_analysis = reflection['character_analysis']
    for key, value in char_analysis.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.3f}")
        else:
            print(f"  {key:25s}: {value}")
    
    print("\nVirtue Profile:")
    profile = reflection['virtue_profile']
    print("  Strongest Virtues:")
    for virt_name, level in profile['strongest_virtues']:
        print(f"    {virt_name:20s}: {level:.3f}")
    print("  Weakest Virtues:")
    for virt_name, level in profile['weakest_virtues']:
        print(f"    {virt_name:20s}: {level:.3f}")
    
    if profile['vice_tendencies']:
        print("\n  Vice Tendencies (Golden Mean Analysis):")
        for virt_name, vice_info in profile['vice_tendencies'].items():
            print(f"    {virt_name:20s}: {vice_info['vice']} (strength: {vice_info['strength']:.3f})")
    
    print("\nPractice Patterns (Habituation):")
    patterns = reflection['practice_patterns']
    print(f"  Total Decisions: {patterns['total_decisions_reviewed']}")
    print("  Most Applied Virtues:")
    for virt_name, count in patterns['most_applied_virtues']:
        print(f"    {virt_name:20s}: {count} times")
    
    print("\nCharacter Development Guidance:")
    print(f"  {reflection['character_development_guidance']}")
    
    print("\nOverall Recommendation:")
    print(f"  {reflection['recommendation']}")


def example_7_integrated_system():
    """Example 7: Using the integrated virtue ethics system."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Integrated Virtue Ethics System")
    print("=" * 80)
    
    ethical_system = IntegratedVirtueEthicsSystem()
    
    # Complex decision requiring virtue-guided reasoning
    decision_context = {
        'description': 'Allocate limited resources fairly among multiple stakeholders',
        'context': {
            'resource_constraint': True,
            'multiple_users': True,
            'competing_needs': True
        },
        'stakeholders': ['user1', 'user2', 'user3', 'system'],
        'possible_actions': [
            {
                'description': 'Equal allocation - fair but may not address needs',
                'type': 'equal_allocation',
                'stakeholders_affected': ['user1', 'user2', 'user3'],
                'fairness_score': 0.9,
                'considers_consequences': True,
                'benefit_level': 0.6
            },
            {
                'description': 'Needs-based allocation - just but complex',
                'type': 'needs_based',
                'stakeholders_affected': ['user1', 'user2', 'user3'],
                'fairness_score': 0.85,
                'considers_consequences': True,
                'benefit_level': 0.8,
                'requires_wisdom': True
            },
            {
                'description': 'Priority-based - efficient but potentially unfair',
                'type': 'priority_based',
                'stakeholders_affected': ['user1'],
                'fairness_score': 0.5,
                'considers_consequences': False,
                'benefit_level': 0.7
            }
        ]
    }
    
    decision = ethical_system.make_ethical_decision(decision_context)
    
    print(f"\nDecision Context: {decision_context['description']}")
    print(f"\nRecommended Decision: {decision['decision'].get('description', 'N/A')}")
    print(f"Ethical Score: {decision['confidence']:.3f}")
    print(f"Should Proceed: {decision['should_proceed']}")
    print(f"\nReasoning:\n{decision['reasoning']}")
    
    # Simulate outcome and learn
    outcome = {
        'positive': True,
        'user_satisfaction': 0.85,
        'fairness_perception': 0.9,
        'system_efficiency': 0.75
    }
    
    ethical_system.reflect_and_learn(decision, outcome)
    
    # Get ethical status
    status = ethical_system.get_ethical_status()
    print("\n\nEthical Status:")
    print(f"  Eudaimonia: {status['virtue_profile']['character_metrics']['eudaimonia_level']:.3f}")
    print(f"  Phronesis: {status['virtue_profile']['character_metrics']['phronesis_level']:.3f}")
    print(f"  Character Integrity: {status['virtue_profile']['character_metrics']['character_integrity']:.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("QUANTUM VIRTUE ETHICS FRAMEWORK - EXAMPLES")
    print("=" * 80)
    print("\nDemonstrating virtue ethics principles:")
    print("  - Character-centered morality")
    print("  - Golden Mean (virtue between deficiency and excess)")
    print("  - Habituation (virtues develop through practice)")
    print("  - Phronesis (practical wisdom)")
    print("  - Eudaimonia (flourishing through virtuous living)")
    
    try:
        example_1_basic_virtue_evaluation()
        example_2_golden_mean_demonstration()
        example_3_habituation_and_character_development()
        example_4_phronesis_practical_wisdom()
        example_5_eudaimonia_flourishing()
        example_6_character_reflection()
        example_7_integrated_system()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

