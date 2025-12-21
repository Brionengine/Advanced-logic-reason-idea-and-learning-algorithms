#!/bin/bash
# Build script for Rust extension
# Usage: ./build_rust_extension.sh

set -e

echo "Building Rust extension for Virtue Ethics Framework..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed. Please install from https://rustup.rs/"
    exit 1
fi

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Navigate to Rust extension directory
cd rust_virtue_quantum

# Build the extension
echo "Building Rust extension..."
maturin develop --release

echo "Rust extension built successfully!"
echo "You can now use rust_virtue_quantum in Python."

