#!/bin/bash
# Setup script for Model-Aware-OT environment
# This script creates a virtual environment and installs dependencies

set -e  # Exit on error

echo "========================================"
echo "Setting up Model-Aware-OT environment"
echo "========================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found!"
fi

# Optionally install runner-requirements.txt if needed
# Note: train.py doesn't require these, but they may be needed for other scripts
if [ -f "runner-requirements.txt" ]; then
    echo ""
    echo "Installing additional dependencies from runner-requirements.txt..."
    echo "Note: These are optional for basic training. If you encounter issues, you can skip this step."
    pip install -r runner-requirements.txt || {
        echo ""
        echo "WARNING: Failed to install some packages from runner-requirements.txt"
        echo "This is usually fine if you only need to run train.py"
        echo "You can continue with just requirements.txt dependencies"
    }
fi

echo ""
echo "========================================"
echo "Environment setup completed!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run training, use:"
echo "  bash train_all_methods.sh"
echo ""
