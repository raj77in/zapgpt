#!/usr/bin/env python3
"""
Test suite for ZapGPT CLI functionality
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Test data directory
TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent


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


class TestCLIBasics:
    """Test basic CLI functionality"""

    def test_cli_help(self):
        """Test that CLI help works"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        output = output.lower()
        assert "zapgpt" in output or "usage" in output

    def test_cli_version_info(self):
        """Test that CLI shows version information"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        # Should contain version or description
        output = result.stdout or result.stderr or ""
        output = output.lower()
        assert any(word in output for word in ["version", "gpt", "llm"])


class TestCLIFlags:
    """Test CLI flags and options"""

    def test_quiet_flag(self):
        """Test --quiet flag is available"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--quiet" in output or "-q" in output

    def test_file_flag(self):
        """Test --file flag is available"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--file" in output or "-f" in output

    def test_provider_flag(self):
        """Test --provider flag is available"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--provider" in output or "-p" in output

    def test_model_flag(self):
        """Test --model flag is available"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--model" in output or "-m" in output


class TestCLIFileInput:
    """Test CLI file input functionality"""

    def test_file_input_structure(self):
        """Test that file input works structurally"""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test log entry\n")
            f.write("2025-01-13 01:30:00 INFO: Sample log message\n")
            f.write("2025-01-13 01:30:05 WARNING: Test warning\n")
            temp_file = f.name

        try:
            # Set dummy API key
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = "dummy_key_for_testing"

            # Test file input (will fail at API call but structure should work)
            result = run_zapgpt_command(
                ["--file", temp_file, "--quiet", "Analyze this file"],
                env=env,
            )

            # Should either succeed or fail with authentication error (not file error)
            if result.returncode != 0:
                error_output = (result.stderr or "").lower()
                # Should be API-related error, not file-related
                assert not any(
                    word in error_output
                    for word in ["file not found", "no such file", "permission denied"]
                )

        finally:
            # Cleanup
            os.unlink(temp_file)


class TestCLIPrompts:
    """Test CLI prompt functionality"""

    def test_list_prompts(self):
        """Test --list-prompt functionality"""
        result = run_zapgpt_command(["--list-prompt"])

        assert result.returncode == 0
        # Should list some prompts
        output = result.stdout or result.stderr or ""
        assert len(output.strip()) > 0

    def test_show_prompt_help(self):
        """Test --show-prompt flag exists"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--show-prompt" in output

    def test_use_prompt_help(self):
        """Test --use-prompt flag exists"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        output = result.stdout or result.stderr or ""
        assert "--use-prompt" in output


class TestCLIConfiguration:
    """Test CLI configuration functionality"""

    def test_config_command(self):
        """Test --config command works"""
        result = run_zapgpt_command(["--config", "--provider", "openai"])

        assert result.returncode == 0
        # Should show configuration information
        output = result.stdout or result.stderr or ""
        output = output.lower()
        assert any(word in output for word in ["config", "directory", "path"])


class TestCLIProviders:
    """Test CLI provider functionality"""

    def test_provider_options_in_help(self):
        """Test that provider options are shown in help"""
        result = run_zapgpt_command(["--help"])

        assert result.returncode == 0
        # Should mention some providers
        output = result.stdout or result.stderr or ""
        output = output.lower()
        assert any(
            provider in output for provider in ["openai", "openrouter", "together"]
        )


if __name__ == "__main__":
    pytest.main([__file__])
