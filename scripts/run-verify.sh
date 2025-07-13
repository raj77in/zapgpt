#!/bin/bash
# Helper script for running verification in pre-commit hooks
# Handles different environments (uv, venv, system python)

set -e

# Function to check if dependencies are available
check_dependencies() {
    python -c "import subprocess, sys" 2>/dev/null
}

# Try different Python environments in order of preference
if command -v uv >/dev/null 2>&1; then
    echo "Using uv to run verification..."
    uv run python verify_install.py
elif [ -f .venv/bin/python ] && .venv/bin/python -c "import subprocess, sys" 2>/dev/null; then
    echo "Using virtual environment to run verification..."
    .venv/bin/python verify_install.py
elif check_dependencies; then
    echo "Using system Python to run verification..."
    python verify_install.py
else
    echo "⚠️  Skipping verification - basic Python modules not available"
    echo "💡 This is unusual - please check your Python installation"
    exit 0  # Don't fail the commit, just skip
fi
