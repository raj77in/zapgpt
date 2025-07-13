#!/usr/bin/env python3
"""
Simple ZapGPT installation verification script for CI/CD
"""

import subprocess
import sys


def safe_print(text):
    """Print text with fallback for encoding issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        fallback = (
            text.replace("🔍", "[INFO]")
            .replace("✅", "[OK]")
            .replace("❌", "[FAIL]")
            .replace("🎉", "[SUCCESS]")
        )
        print(fallback)


def main():
    """Verify ZapGPT installation"""
    safe_print("🔍 Verifying ZapGPT installation...")

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
        safe_print("✅ ZapGPT imports successfully")
    except ImportError as e:
        safe_print(f"❌ Import failed: {e}")
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
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            if result.returncode == 0:
                safe_print(f"✅ {description} works")
            else:
                safe_print(f"❌ {description} failed with code {result.returncode}")
                if result.stderr:
                    # Safely handle Unicode in error messages
                    error_msg = result.stderr.strip()[:100]
                    try:
                        safe_print(f"   Error: {error_msg}...")
                    except UnicodeEncodeError:
                        safe_print("   Error: [Unicode encoding error in stderr]...")
                return 1
        except Exception as e:
            safe_print(f"❌ {description} failed: {e}")
            return 1

    safe_print("🎉 ZapGPT installation verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
