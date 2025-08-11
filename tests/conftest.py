#!/usr/bin/env python3
"""
Pytest configuration and fixtures for ZapGPT tests
"""

import os
import tempfile
from pathlib import Path

import pytest

# Set up test environment
os.environ["OPENAI_API_KEY"] = "dummy_key_for_testing"
os.environ["OPENROUTER_KEY"] = "dummy_key_for_testing"
os.environ["TOGETHER_API_KEY"] = "dummy_key_for_testing"
os.environ["REPLICATE_API_TOKEN"] = "dummy_key_for_testing"
os.environ["DEEPINFRA_API_TOKEN"] = "dummy_key_for_testing"
os.environ["GITHUB_KEY"] = "dummy_key_for_testing"


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Set up test database path and ensure cleanup."""
    # Create a temporary directory for the test database
    test_db_dir = tmp_path / "test_db"
    test_db_dir.mkdir(exist_ok=True)
    test_db_path = test_db_dir / "test_db.sqlite"

    # Set the environment variable
    os.environ["ZAPGPT_DB_PATH"] = str(test_db_path)

    # Ensure the directory exists and is writable
    test_db_path.parent.mkdir(parents=True, exist_ok=True)
    test_db_path.touch()

    # Make sure the database is writable
    test_db_path.chmod(0o644)

    yield str(test_db_path)

    # Cleanup
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except OSError:
            pass


@pytest.fixture
def temp_file():
    """Create a temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Test content\n2025-01-13 INFO: Sample log entry\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        f.write("2025-01-13 01:30:00 INFO: User login successful\n")
        f.write("2025-01-13 01:30:05 WARNING: Failed login attempt\n")
        f.write("2025-01-13 01:30:10 ERROR: Multiple failed attempts\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write(
            """
# Sample Python code for testing
import os

def example_function():
    password = "hardcoded_password"  # Security issue
    return password

if __name__ == "__main__":
    example_function()
        """
        )
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def project_dir():
    """Get the project directory path"""
    return Path(__file__).parent.parent
