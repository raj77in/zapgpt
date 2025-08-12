#!/usr/bin/env python3
"""
Test suite for ZapGPT programmatic API
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from openai import AuthenticationError

# Test data directory
TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent

# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "dummy_key_for_testing"


def run_zapgpt_command(args, **kwargs):
    """Helper function to run zapgpt commands with proper encoding"""
    cmd = [sys.executable, "-m", "zapgpt"] + args
    defaults = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "cwd": PROJECT_DIR,
    }
    defaults.update(kwargs)
    return subprocess.run(cmd, **defaults)


class TestQueryLLMAPI:
    """Test the query_llm programmatic API"""

    def test_import_query_llm(self):
        """Test that query_llm can be imported successfully"""
        from zapgpt import query_llm

        assert callable(query_llm)

    def test_query_llm_with_invalid_provider(self):
        """Test error handling for invalid provider"""
        from zapgpt import query_llm

        with pytest.raises(ValueError, match="Unsupported provider"):
            query_llm("Hello", provider="invalid_provider")

    def test_query_llm_missing_api_key(self):
        """Test error handling for missing API key"""
        from zapgpt import query_llm

        # Temporarily remove API key
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            with pytest.raises(
                EnvironmentError, match="Missing required environment variable"
            ):
                query_llm("Hello", provider="openai")
        finally:
            # Restore API key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_query_llm_structure_with_dummy_key(self):
        """Test that query_llm structure works (will fail at API call)"""
        from zapgpt import query_llm

        # This should fail with authentication error, not structure error
        with pytest.raises(Exception) as exc_info:
            query_llm("Hello", provider="openai", quiet=True)
        print("Executed")
        # Should be an API authentication error, not a code structure error
        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in ["authentication", "api key", "unauthorized", "invalid"]
        )

    def test_query_llm_with_custom_parameters(self):
        """Test query_llm with various parameters"""
        from zapgpt import query_llm

        # Test with custom parameters (will fail at API call but structure should work)
        with pytest.raises(AuthenticationError):
            query_llm(
                "Test prompt",
                provider="openai",
                model="gpt-4o",
                temperature=0.8,
                max_tokens=100,
                quiet=True,
            )


class TestFileProcessing:
    """Test file processing functionality"""

    def test_file_processing_structure(self):
        """Test file processing with temporary file"""
        from zapgpt import query_llm

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test log entry\n2025-01-13 INFO: Sample log message\n")
            temp_file = f.name

        try:
            # Read file content
            with open(temp_file) as f:
                content = f.read()

            # Test structure (will fail at API call)
            with pytest.raises(AuthenticationError):
                query_llm(f"Analyze this log: {content}", provider="openai", quiet=True)
        finally:
            # Cleanup
            os.unlink(temp_file)


class TestSubprocessIntegration:
    """Test subprocess integration examples"""

    def test_subprocess_integration_structure(self):
        """Test subprocess integration structure"""
        from zapgpt import query_llm

        # Run simple command
        result = subprocess.run(["echo", "test output"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "test output" in result.stdout

        # Test integration structure (will fail at API call)
        with pytest.raises(AuthenticationError):
            query_llm(
                f"Analyze this output: {result.stdout}", provider="openai", quiet=True
            )


class TestProviderMapping:
    """Test provider mapping and environment variables"""

    def test_all_providers_have_env_vars(self):
        """Test that all providers have corresponding environment variables"""
        from zapgpt.main import provider_env_vars, provider_map

        for provider in provider_map.keys():
            assert provider in provider_env_vars, (
                f"Provider {provider} missing environment variable mapping"
            )

    def test_provider_classes_exist(self):
        """Test that all provider classes are defined"""
        from zapgpt.main import provider_map

        for provider, client_class in provider_map.items():
            assert callable(client_class), (
                f"Provider {provider} class {client_class} is not callable"
            )


class TestCLIIntegration:
    """Test CLI integration and flags"""

    def test_quiet_flag_exists(self):
        """Test that --quiet flag is available in CLI"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--quiet" in output or "-q" in output

    def test_file_flag_exists(self):
        """Test that --file flag is available in CLI"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--file" in output or "-f" in output

    def test_provider_flag_exists(self):
        """Test that --provider flag is available in CLI"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--provider" in output or "-p" in output


if __name__ == "__main__":
    pytest.main([__file__])
