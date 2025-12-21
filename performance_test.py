#!/usr/bin/env python3
"""
Performance test script for virtue ethics framework.
Tests multiple action evaluations to identify performance bottlenecks.
"""

from virtue_ethics_quantum import QuantumVirtueEthicsFramework
import time

def test_performance():
    """Test performance with multiple action evaluations."""
    print("=" * 80)
    print("Performance Test - Virtue Ethics Framework")
    print("=" * 80)
    
    qvef = QuantumVirtueEthicsFramework()
    
    # Test actions - mix of similar and different actions
    test_actions = [
        {
            'description': 'Help user understand quantum computing concepts',
            'type': 'assistance',
            'domain': 'communication',
            'transparency': 0.9,
            'truthfulness': 0.95,
            'considers_consequences': True
        },
        {
            'description': 'Help user understand machine learning algorithms',
            'type': 'assistance',
            'domain': 'communication',
            'transparency': 0.9,
            'truthfulness': 0.95,
            'considers_consequences': True
        },
        {
            'description': 'Make a decision about resource allocation',
            'type': 'decision',
            'domain': 'decision_making',
            'stakeholders_affected': ['user1', 'user2'],
            'fairness_score': 0.8,
            'considers_consequences': True
        },
        {
            'description': 'Help user understand quantum computing concepts',  # Duplicate
            'type': 'assistance',
            'domain': 'communication',
            'transparency': 0.9,
            'truthfulness': 0.95,
            'considers_consequences': True
        },
        {
            'description': 'Allocate resources fairly among users',
            'type': 'allocation',
            'domain': 'decision_making',
            'stakeholders_affected': ['user1', 'user2', 'user3'],
            'fairness_score': 0.85,
            'considers_consequences': True
        }
    ]
    
    print(f"\nTesting {len(test_actions)} action evaluations...")
    print("(This will generate performance logs for analysis)\n")
    
    total_start = time.time()
    
    for i, action in enumerate(test_actions, 1):
        print(f"Evaluating action {i}/{len(test_actions)}: {action['description'][:50]}...")
        eval_start = time.time()
        evaluation = qvef.evaluate_action(action)
        eval_time = time.time() - eval_start
        print(f"  ✓ Completed in {eval_time:.4f}s (score: {evaluation['overall_ethical_score']:.3f})")
    
    total_time = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per evaluation: {total_time / len(test_actions):.4f}s")
    print("=" * 80)
    print("\nPerformance logs written to: c:\\Virtue Ethics\\.cursor\\debug.log")
    print("Review logs to identify performance bottlenecks.")

if __name__ == "__main__":
    test_performance()

