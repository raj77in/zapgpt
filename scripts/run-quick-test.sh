#!/bin/bash
# Helper script for running a quick test in pre-commit hooks
# Handles different environments (uv, venv, system python)

set -e

# Function to check if dependencies are available
check_dependencies() {
    python -c "import openai, pytest" 2>/dev/null
}

# Try different Python environments in order of preference
if command -v uv >/dev/null 2>&1; then
    echo "Using uv to run quick test..."
    uv run pytest tests/test_cli.py::TestCLIBasics::test_cli_help -q
elif [ -f .venv/bin/python ] && .venv/bin/python -c "import openai, pytest" 2>/dev/null; then
    echo "Using virtual environment to run quick test..."
    .venv/bin/python -m pytest tests/test_cli.py::TestCLIBasics::test_cli_help -q
elif check_dependencies; then
    echo "Using system Python to run quick test..."
    python -m pytest tests/test_cli.py::TestCLIBasics::test_cli_help -q
else
    echo "⚠️  Skipping quick test - dependencies (openai, pytest) not available"
    echo "💡 To fix this, run: uv sync  or  pip install -e ."
    exit 0  # Don't fail the commit, just skip
fi
