#!/usr/bin/env python3
"""
Script to copy Brion Quantum AI modules to Virtue Ethics workspace.

Run this script from the Virtue Ethics workspace directory to copy
the important Brion Quantum AI modules.

Usage:
    python copy_brion_modules.py
"""

import shutil
import os
from pathlib import Path

# Source and destination paths
BRION_SOURCE = Path(r"C:\Brion Quantum AI\Brion-Quantum-A.I.-Large-Language-Model-Agent-L.L.M.A--main")
DEST_DIR = Path(__file__).parent

# Modules to copy (core/imperative modules)
MODULES_TO_COPY = [
    "UnifiedQuantumMind.py",
    "BrionQuantumAI_Main.py",
    "QuantumOSIntegration.py",
    "GoogleWillowIntegration.py",
    "epistemic_confidence.py",
    "idea_generator.py",
    "subconscious_framework.py",
    # These should already exist, but list for completeness:
    # "meta_reasoning.py",
    # "logic_engine.py",
    # "probabilistic_reasoning.py",
    # "qt_inspired_memory.py",
    # "experience_replay.py",
]

# Optional: Fault-tolerant quantum modules
OPTIONAL_MODULES = [
    "fault_tolerant_quantum_system.py",
    "fault_tolerance_analysis.py",
    "fault_tolerant_gates.py",
    "error_decoder.py",
    "surface_code_implementation.py",
    "magic_state_distillation.py",
    "noise_models.py",
]


def copy_file(source_file: Path, dest_file: Path, prefix: str = "Brion_"):
    """Copy a file with optional prefix to destination."""
    if not source_file.exists():
        print(f"  ⚠️  Source file not found: {source_file}")
        return False
    
    try:
        # Add prefix to avoid conflicts
        dest_name = prefix + dest_file.name if prefix else dest_file.name
        dest_path = dest_file.parent / dest_name
        
        shutil.copy2(source_file, dest_path)
        print(f"  ✅ Copied: {source_file.name} -> {dest_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ Error copying {source_file.name}: {e}")
        return False


def main():
    """Main function to copy modules."""
    print("=" * 80)
    print("Brion Quantum AI Module Copy Script")
    print("=" * 80)
    print(f"\nSource: {BRION_SOURCE}")
    print(f"Destination: {DEST_DIR}")
    
    if not BRION_SOURCE.exists():
        print(f"\n❌ Error: Source directory not found: {BRION_SOURCE}")
        print("\nPlease ensure the Brion Quantum AI workspace is accessible.")
        return
    
    print(f"\n✅ Source directory found")
    print(f"\n📋 Copying core modules...")
    
    copied_count = 0
    skipped_count = 0
    
    # Copy core modules
    for module_name in MODULES_TO_COPY:
        source_file = BRION_SOURCE / module_name
        dest_file = DEST_DIR / module_name
        
        # Skip if already exists (user may have already copied)
        if dest_file.exists():
            print(f"  ⏭️  Already exists: {module_name}")
            skipped_count += 1
            continue
        
        if copy_file(source_file, dest_file, prefix=""):
            copied_count += 1
        else:
            skipped_count += 1
    
    # Ask about optional modules
    print(f"\n📋 Optional fault-tolerant quantum modules available:")
    print("   (These are large and may not be needed for basic integration)")
    response = input("\nCopy optional modules? (y/N): ").strip().lower()
    
    if response == 'y':
        for module_name in OPTIONAL_MODULES:
            source_file = BRION_SOURCE / module_name
            dest_file = DEST_DIR / module_name
            
            if dest_file.exists():
                print(f"  ⏭️  Already exists: {module_name}")
                continue
            
            copy_file(source_file, dest_file, prefix="")
    
    print("\n" + "=" * 80)
    print(f"✅ Copy complete!")
    print(f"   Copied: {copied_count} files")
    print(f"   Skipped: {skipped_count} files")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the copied modules")
    print("2. Run: python VirtueEthics_QuantumAI_Integration.py")
    print("3. Check integration_summary.md for integration details")


if __name__ == "__main__":
    main()

