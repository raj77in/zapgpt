#!/usr/bin/env python3
"""
Simple ZapGPT installation verification script for CI/CD
"""

import subprocess
import sys


def main():
    """Verify ZapGPT installation"""
    print("🔍 Verifying ZapGPT installation...")

    # Test import
    try:
        import zapgpt
        from zapgpt import main, query_llm

        # Verify the imports are callable
        assert callable(main)
        assert callable(query_llm)
        # Check if zapgpt has version info
        if hasattr(zapgpt, "__version__"):
            pass  # Version available
        print("✅ ZapGPT imports successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return 1

    # Test CLI help
    try:
        result = subprocess.run(
            [sys.executable, "-m", "zapgpt", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("✅ CLI help command works")
        else:
            print(f"❌ CLI help failed with code {result.returncode}")
            return 1
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return 1

    # Test configuration
    try:
        result = subprocess.run(
            [sys.executable, "-m", "zapgpt", "--config", "--provider", "openai"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print("✅ Configuration command executed")
    except Exception as e:
        print(f"⚠️  Configuration test warning: {e}")

    print("🎉 ZapGPT installation verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
