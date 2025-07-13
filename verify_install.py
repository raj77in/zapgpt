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

    # Test all CLI commands that should work without API keys
    commands_to_test = [
        ([sys.executable, "-m", "zapgpt", "--help"], "CLI help command"),
        ([sys.executable, "-m", "zapgpt", "--config"], "CLI config command"),
        ([sys.executable, "-m", "zapgpt", "--list-prompt"], "CLI list-prompt command"),
        (
            [sys.executable, "-m", "zapgpt", "--show-prompt", "coding"],
            "CLI show-prompt command",
        ),
    ]

    for cmd, description in commands_to_test:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print(f"✅ {description} works")
            else:
                print(f"❌ {description} failed with code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()[:100]}...")
                return 1
        except Exception as e:
            print(f"❌ {description} failed: {e}")
            return 1

    print("🎉 ZapGPT installation verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
