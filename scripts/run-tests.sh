#!/bin/bash
# Helper script for running tests in pre-commit hooks
# Handles different environments (uv, venv, system python)

set -e

# Function to check if dependencies are available
check_dependencies() {
    python -c "import openai, pytest" 2>/dev/null
}

# Try different Python environments in order of preference
if command -v uv >/dev/null 2>&1; then
    echo "Using uv to run tests..."
    uv run pytest tests/ --tb=short -q
elif [ -f .venv/bin/python ] && .venv/bin/python -c "import openai, pytest" 2>/dev/null; then
    echo "Using virtual environment to run tests..."
    .venv/bin/python -m pytest tests/ --tb=short -q
elif check_dependencies; then
    echo "Using system Python to run tests..."
    python -m pytest tests/ --tb=short -q
else
    echo "⚠️  Skipping tests - dependencies (openai, pytest) not available in any Python environment"
    echo "💡 To fix this, run: uv sync  or  pip install -e ."
    exit 0  # Don't fail the commit, just skip
fi
