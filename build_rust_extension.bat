@echo off
REM Build script for Rust extension on Windows
REM Usage: build_rust_extension.bat

echo Building Rust extension for Virtue Ethics Framework...

REM Check if Rust is installed
where cargo >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Rust is not installed. Please install from https://rustup.rs/
    exit /b 1
)

REM Check if maturin is installed
where maturin >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing maturin...
    pip install maturin
)

REM Navigate to Rust extension directory
cd rust_virtue_quantum

REM Build the extension
echo Building Rust extension...
maturin develop --release

echo Rust extension built successfully!
echo You can now use rust_virtue_quantum in Python.

