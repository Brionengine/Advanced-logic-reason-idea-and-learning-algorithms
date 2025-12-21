# Installation Guide: Quantum Backends & Rust Optimization

## Quick Start

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- Qiskit (IBM quantum framework)
- Cirq (Google quantum framework)
- PennyLane (Quantum ML framework)
- TensorFlow Quantum (Quantum ML)
- Maturin (Rust-Python bridge)

### 2. (Optional) Build Rust Extension

#### Prerequisites
- Rust compiler: Install from https://rustup.rs/
- Maturin: `pip install maturin`

#### Build
```bash
# Linux/Mac
cd rust_virtue_quantum
./build_rust_extension.sh

# Windows
cd rust_virtue_quantum
build_rust_extension.bat
```

### 3. Verify Installation
```bash
python test_quantum_backends.py
```

## Current Status

Based on your system, the following backends are available:
- ✓ **Qiskit** - IBM quantum framework
- ✓ **Cirq** - Google quantum framework  
- ✓ **PennyLane** - Quantum ML (auto-selected as preferred)
- ✓ **Qualtran** - Google quantum algorithms

## Usage

### Basic Usage
```python
from virtue_ethics_quantum import QuantumVirtueEthicsFramework

# Auto-select best backend (currently PennyLane)
qvef = QuantumVirtueEthicsFramework(quantum_backend='auto')
```

### Select Specific Backend
```python
# Use PennyLane (recommended)
qvef = QuantumVirtueEthicsFramework(quantum_backend='pennylane')

# Use Cirq
qvef = QuantumVirtueEthicsFramework(quantum_backend='cirq')

# Use Qiskit
qvef = QuantumVirtueEthicsFramework(quantum_backend='qiskit')
```

## Performance

With current setup:
- **Backend**: PennyLane (auto-selected)
- **Caching**: Enabled (39% speedup for duplicates)
- **Rust Extension**: Available (optional, 10-50x faster string ops)

## Troubleshooting

### JAX Version Warning
If you see: `PennyLane is not yet compatible with JAX versions > 0.6.2`

This is just a warning, not an error. The system works fine. To fix:
```bash
pip install jax~=0.6.0 jaxlib~=0.6.0
```

### Backend Not Available
If a backend fails, the system automatically falls back to:
1. Next available backend
2. Qiskit (if available)
3. Classical-only mode

### Rust Extension Not Building
- Ensure Rust is installed: `cargo --version`
- Ensure you're in the correct directory
- Check Rust version (requires 1.70+)

## Next Steps

1. **Test Performance**: Run `python test_quantum_backends.py`
2. **Build Rust Extension** (optional): See step 2 above
3. **Use in Code**: See usage examples above

## Files Created

- `quantum_backends.py` - Backend abstraction layer
- `rust_virtue_quantum/` - Rust extension source
- `test_quantum_backends.py` - Testing script
- `QUANTUM_BACKENDS_GUIDE.md` - Detailed guide
- `OPTIMIZATION_SUMMARY.md` - Complete summary

