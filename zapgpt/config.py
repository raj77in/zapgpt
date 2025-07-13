#!/usr/bin/env python3
"""
Configuration management for zapgpt.
"""

import os
from pathlib import Path


def get_config_dir():
    """Get the zapgpt configuration directory path."""
    return Path.home() / ".config" / "zapgpt"


def ensure_config_directory():
    """Ensure the configuration directory exists."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_pricing_file_path():
    """Get the pricing file path."""
    return get_config_dir() / "pricing.json"


def get_config_dir_old():
    """Get the zapgpt configuration directory path."""
    return os.path.expanduser("~/.config/zapgpt")


def get_prompts_dir():
    """Get the prompts directory path."""
    return os.path.join(get_config_dir(), "prompts")


def get_db_file():
    """Get the database file path."""
    return os.path.join(get_config_dir(), "gpt_usage.db")


def get_pricing_file():
    """Get the pricing file path."""
    return os.path.join(get_config_dir(), "pricing.json")


def show_config_info():
    """Display configuration directory information."""
    config_dir = get_config_dir()
    prompts_dir = get_prompts_dir()
    db_file = get_db_file()
    pricing_file = get_pricing_file()

    print("📁 ZapGPT Configuration")
    print("=" * 30)
    print(f"Config directory: {config_dir}")
    print(f"Prompts directory: {prompts_dir}")
    print(f"Database file: {db_file}")
    print(f"Pricing file: {pricing_file}")
    print()

    if os.path.exists(config_dir):
        print("✅ Configuration directory exists")
        if os.path.exists(prompts_dir):
            prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith(".json")]
            print(f"✅ Prompts directory exists ({len(prompt_files)} prompts)")
            for prompt_file in sorted(prompt_files):
                print(f"  - {prompt_file}")
        else:
            print("❌ Prompts directory does not exist")

        if os.path.exists(db_file):
            print("✅ Database file exists")
        else:
            print("ℹ️  Database file will be created on first use")

        if os.path.exists(pricing_file):
            print("✅ Pricing file exists")
        else:
            print("❌ Pricing file missing")
    else:
        print("❌ Configuration directory does not exist")
        print("Run zapgpt once to initialize configuration")


if __name__ == "__main__":
    show_config_info()
