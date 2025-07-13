#!/usr/bin/env python3
"""
ZapGPT Installation Verification Script

This script verifies that ZapGPT is properly installed and configured.
It checks dependencies, imports, CLI functionality, and basic operations.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path


# Colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")


def print_header(message):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.END}")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Python Version Check")

    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version >= (3, 9):
        print_success("Python version is compatible (>=3.9)")
        return True
    else:
        print_error(
            f"Python version {version.major}.{version.minor} is not supported. Requires Python >=3.9"
        )
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("Dependency Check")

    required_packages = [
        "openai",
        "requests",
        "tabulate",
        "tiktoken",
        "rich",
        "pygments",
        "httpx",
    ]

    all_good = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is NOT installed")
            all_good = False

    return all_good


def check_zapgpt_import():
    """Check if zapgpt can be imported"""
    print_header("ZapGPT Import Check")

    try:
        import zapgpt

        print_success("zapgpt module imported successfully")

        # Check main functions
        from zapgpt import main, query_llm

        # Verify the imports are callable
        assert callable(main)
        assert callable(query_llm)
        print_success("Main functions (query_llm, main) imported successfully")

        # Check version if available
        if hasattr(zapgpt, "__version__"):
            print_info(f"ZapGPT version: {zapgpt.__version__}")

        return True
    except ImportError as e:
        print_error(f"Failed to import zapgpt: {e}")
        return False


def check_cli_functionality():
    """Check if CLI commands work"""
    print_header("CLI Functionality Check")

    try:
        # Test help command
        result = subprocess.run(
            [sys.executable, "-m", "zapgpt", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print_success("CLI help command works")
        else:
            print_error(f"CLI help command failed with return code {result.returncode}")
            return False

        # Test version command
        result = subprocess.run(
            [sys.executable, "-m", "zapgpt", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print_success("CLI version command works")
            if result.stdout.strip():
                print_info(f"Version output: {result.stdout.strip()}")
        else:
            print_warning(
                "CLI version command returned non-zero exit code (may be expected)"
            )

        return True

    except subprocess.TimeoutExpired:
        print_error("CLI commands timed out")
        return False
    except Exception as e:
        print_error(f"CLI test failed: {e}")
        return False


def check_configuration():
    """Check configuration and environment"""
    print_header("Configuration Check")

    # Check if config command works
    try:
        result = subprocess.run(
            [sys.executable, "-m", "zapgpt", "--config"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print_success("Configuration command works")
        else:
            print_warning(
                "Configuration command returned non-zero exit code (may indicate missing API keys)"
            )

    except Exception as e:
        print_warning(f"Configuration check failed: {e}")

    # Check for API keys
    api_keys = [
        "OPENAI_API_KEY",
        "OPENROUTER_KEY",
        "TOGETHER_API_KEY",
        "REPLICATE_API_TOKEN",
        "DEEPINFRA_API_TOKEN",
        "GITHUB_KEY",
    ]

    found_keys = []
    for key in api_keys:
        if os.getenv(key):
            found_keys.append(key)

    if found_keys:
        print_success(f"Found API keys: {', '.join(found_keys)}")
    else:
        print_warning("No API keys found in environment variables")
        print_info("Set at least one API key to use ZapGPT with real providers")


def check_project_structure():
    """Check if project structure is correct"""
    print_header("Project Structure Check")

    current_dir = Path.cwd()
    expected_files = [
        "pyproject.toml",
        "README.md",
        "zapgpt/__init__.py",
        "zapgpt/main.py",
        "zapgpt/cli.py",
        "zapgpt/config.py",
    ]

    all_good = True
    for file_path in expected_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print_success(f"{file_path} exists")
        else:
            print_error(f"{file_path} is missing")
            all_good = False

    return all_good


def run_basic_functionality_test():
    """Test basic functionality without making API calls"""
    print_header("Basic Functionality Test")

    try:
        from zapgpt import query_llm

        # This should fail with missing API key or authentication error
        # but it tests that the function structure works
        try:
            query_llm("test", provider="openai", quiet=True)
            print_warning("API call succeeded (unexpected - may indicate dummy keys)")
        except Exception as e:
            if "API key" in str(e) or "Authentication" in str(e) or "401" in str(e):
                print_success(
                    "Function structure works (failed as expected due to API key)"
                )
            else:
                print_warning(f"Function failed with unexpected error: {e}")

        return True

    except Exception as e:
        print_error(f"Basic functionality test failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("🚀 ZapGPT Installation Verification")
    print("====================================")
    print(f"{Colors.END}")

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("ZapGPT Import", check_zapgpt_import),
        ("CLI Functionality", check_cli_functionality),
        ("Configuration", check_configuration),
        ("Project Structure", check_project_structure),
        ("Basic Functionality", run_basic_functionality_test),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_error(f"{check_name} check failed with exception: {e}")
            results.append((check_name, False))

    # Summary
    print_header("Verification Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        if result:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")

    print(f"\n{Colors.BOLD}Overall Result: {passed}/{total} checks passed{Colors.END}")

    if passed == total:
        print_success("🎉 ZapGPT installation is fully verified!")
        return 0
    elif passed >= total * 0.8:  # 80% pass rate
        print_warning("⚠️  ZapGPT installation is mostly working with some issues")
        return 1
    else:
        print_error("❌ ZapGPT installation has significant issues")
        return 2


if __name__ == "__main__":
    sys.exit(main())
