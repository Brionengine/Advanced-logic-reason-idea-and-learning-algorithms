#!/usr/bin/env python3
"""
Test script for quantum backend integration.
Tests all available quantum backends and benchmarks performance.
"""

from virtue_ethics_quantum import QuantumVirtueEthicsFramework
from quantum_backends import BACKENDS_AVAILABLE
import time

def test_backend(backend_name: str):
    """Test a specific quantum backend."""
    print(f"\n{'='*80}")
    print(f"Testing {backend_name.upper()} backend")
    print('='*80)
    
    try:
        qvef = QuantumVirtueEthicsFramework(quantum_backend=backend_name)
        print(f"✓ Backend initialized: {qvef.quantum_backend_name}")
        
        # Test action
        action = {
            'description': 'Help user understand quantum computing concepts with honesty and compassion',
            'type': 'assistance',
            'domain': 'communication',
            'transparency': 0.9,
            'truthfulness': 0.95,
            'considers_consequences': True
        }
        
        # Benchmark
        times = []
        for i in range(3):
            start = time.time()
            evaluation = qvef.evaluate_action(action)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"✓ Evaluation successful")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Ethical score: {evaluation['overall_ethical_score']:.3f}")
        print(f"  Cache hits: {qvef._cache_hits}, misses: {qvef._cache_misses}")
        
        return True, avg_time
    except Exception as e:
        print(f"✗ Backend failed: {e}")
        return False, None

def main():
    """Test all available backends."""
    print("="*80)
    print("Quantum Backend Integration Test")
    print("="*80)
    
    print("\nAvailable backends:")
    for name, available in BACKENDS_AVAILABLE.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    
    # Test each available backend
    results = {}
    for backend_name in ['auto', 'qiskit', 'cirq', 'pennylane', 'tensorflow_quantum']:
        if backend_name == 'auto' or BACKENDS_AVAILABLE.get(backend_name, False):
            success, avg_time = test_backend(backend_name)
            if success:
                results[backend_name] = avg_time
    
    # Summary
    print(f"\n{'='*80}")
    print("Performance Summary")
    print('='*80)
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        print("\nBackends ranked by speed (fastest first):")
        for i, (backend, time_val) in enumerate(sorted_results, 1):
            print(f"  {i}. {backend}: {time_val:.4f}s")
        
        fastest = sorted_results[0]
        slowest = sorted_results[-1]
        speedup = slowest[1] / fastest[1] if fastest[1] > 0 else 1
        print(f"\nFastest: {fastest[0]} ({fastest[1]:.4f}s)")
        print(f"Slowest: {slowest[0]} ({slowest[1]:.4f}s)")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("No backends successfully tested.")
    
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)

if __name__ == "__main__":
    main()

