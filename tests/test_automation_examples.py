#!/usr/bin/env python3
"""
Test suite for ZapGPT automation examples from README
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


class TestPenetrationTestingExamples:
    """Test penetration testing automation examples"""

    def test_nmap_analysis_structure(self):
        """Test nmap analysis example structure"""
        from zapgpt import query_llm

        # Simulate nmap output
        fake_nmap_output = """
        Starting Nmap 7.80 ( https://nmap.org )
        Nmap scan report for example.com (93.184.216.34)
        Host is up (0.032s latency).
        PORT     STATE SERVICE
        22/tcp   open  ssh
        80/tcp   open  http
        443/tcp  open  https
        """

        # Test the structure (will fail at API call)
        with pytest.raises(AuthenticationError):
            query_llm(
                f"Analyze this nmap scan: {fake_nmap_output}",
                provider="openai",
                use_prompt="vuln_assessment",
                quiet=True,
            )

    def test_subprocess_nmap_simulation(self):
        """Test subprocess integration with echo (simulating nmap)"""
        # Use echo to simulate nmap command
        result = subprocess.run(
            ["echo", "PORT 22/tcp open ssh"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "ssh" in result.stdout

        # Test integration structure
        from zapgpt import query_llm

        with pytest.raises(AuthenticationError):
            query_llm(f"Analyze scan: {result.stdout}", provider="openai", quiet=True)


class TestLogAnalysisExamples:
    """Test log analysis automation examples"""

    def test_log_analysis_structure(self):
        """Test log analysis example structure"""
        from zapgpt import query_llm

        # Create fake log content
        fake_logs = """
        2025-01-13 01:30:00 INFO: User login successful for admin
        2025-01-13 01:30:05 WARNING: Failed login attempt from IP 192.168.1.100
        2025-01-13 01:30:10 ERROR: Multiple failed login attempts detected
        """

        # Test structure (will fail at API call)
        with pytest.raises(AuthenticationError):
            query_llm(
                f"Detect suspicious activity: {fake_logs}",
                provider="openai",
                quiet=True,
            )

    def test_file_based_log_analysis(self):
        """Test file-based log analysis"""
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write("2025-01-13 01:30:00 INFO: Normal operation\n")
            f.write("2025-01-13 01:30:05 WARNING: Suspicious activity detected\n")
            f.write("2025-01-13 01:30:10 ERROR: Security breach attempt\n")
            temp_log = f.name

        try:
            # Read log file
            with open(temp_log) as f:
                logs = f.read()

            assert "suspicious" in logs.lower()

            # Test analysis structure
            from zapgpt import query_llm

            with pytest.raises(AuthenticationError):
                query_llm(f"Analyze these logs: {logs}", provider="openai", quiet=True)
        finally:
            os.unlink(temp_log)


class TestCodeReviewExamples:
    """Test code review automation examples"""

    def test_code_review_structure(self):
        """Test code review example structure"""
        from zapgpt import query_llm

        # Sample Python code
        sample_code = """
        import os
        password = "hardcoded_password"

        def login(user_input):
            if user_input == password:
                return True
            return False
        """

        # Test structure (will fail at API call)
        with pytest.raises(AuthenticationError):
            query_llm(
                f"Review this code for security issues: {sample_code}",
                provider="openai",
                use_prompt="coding",
                quiet=True,
            )

    def test_multiple_file_processing(self):
        """Test processing multiple files structure"""
        # Create temporary Python files
        temp_files = []

        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".py"
                ) as f:
                    f.write(f"# Test file {i}\nprint('Hello world {i}')\n")
                    temp_files.append(f.name)

            # Test processing structure
            from zapgpt import query_llm

            for file_path in temp_files:
                with open(file_path) as f:
                    code = f.read()

                # Test structure (will fail at API call)
                with pytest.raises(AuthenticationError):
                    query_llm(
                        f"Review this code: {code}",
                        provider="openai",
                        use_prompt="coding",
                        quiet=True,
                    )

        finally:
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestCLIAutomationExamples:
    """Test CLI automation examples from README"""

    def test_quiet_mode_cli(self):
        """Test quiet mode CLI functionality"""
        # Create test file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test content for analysis\n")
            temp_file = f.name

        try:
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = "dummy_key"

            # Test quiet mode structure
            result = run_zapgpt_command(
                ["--quiet", "--file", temp_file, "Analyze this"],
                env=env,
            )

            # Should either succeed or fail with API error (not CLI error)
            if result.returncode != 0:
                # Should not be a CLI argument error
                stderr = result.stderr or ""
                assert "unrecognized arguments" not in stderr
                assert "invalid choice" not in stderr

        finally:
            os.unlink(temp_file)

    def test_batch_processing_structure(self):
        """Test batch processing structure"""
        # Create multiple test files
        temp_files = []

        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".log"
                ) as f:
                    f.write(f"Log entry {i}\nTest content {i}\n")
                    temp_files.append(f.name)

            env = os.environ.copy()
            env["OPENAI_API_KEY"] = "dummy_key"

            # Test batch processing structure
            for temp_file in temp_files:
                result = run_zapgpt_command(
                    ["-q", "-f", temp_file, "Summarize"],
                    env=env,
                )

                # Should not be a CLI structure error
                if result.returncode != 0:
                    stderr = result.stderr or ""
                    assert "unrecognized arguments" not in stderr

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestREADMEExampleValidity:
    """Test that README examples are syntactically correct"""

    def test_python_examples_syntax(self):
        """Test that Python code examples in README are valid Python"""
        # Sample Python examples from README
        examples = [
            """
import subprocess
from zapgpt import query_llm

def analyze_nmap_scan(target):
    result = subprocess.run(['nmap', '-sV', target], capture_output=True, text=True)
    analysis = query_llm(
        f"Analyze this nmap scan: {result.stdout}",
        provider="openai",
        use_prompt="vuln_assessment"
    )
    return analysis
            """,
            """
from zapgpt import query_llm

def monitor_logs(log_file):
    with open(log_file, 'r') as f:
        logs = f.read()

    alert = query_llm(
        f"Detect suspicious activity: {logs}",
        provider="openai",
        quiet=True
    )

    if "suspicious" in alert.lower():
        print(f"ALERT: {alert}")
        return True
    return False
            """,
        ]

        # Test that examples compile
        for i, example in enumerate(examples):
            try:
                compile(example, f"<example_{i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"README example {i} has syntax error: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
