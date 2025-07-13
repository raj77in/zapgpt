#!/usr/bin/env python3
"""
Test suite for ZapGPT configuration functionality
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Test data directory
TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent


class TestConfigurationPaths:
    """Test configuration directory and file paths"""

    def test_config_module_import(self):
        """Test that config module can be imported"""
        from zapgpt import config

        assert hasattr(config, "get_config_dir")

    def test_config_directory_creation(self):
        """Test configuration directory creation"""
        from zapgpt.config import ensure_config_directory, get_config_dir

        # Should not raise an exception
        config_dir = get_config_dir()
        assert isinstance(config_dir, Path)

        # Should be able to ensure directory exists
        ensure_config_directory()
        assert config_dir.exists()

    def test_pricing_file_path(self):
        """Test pricing file path functionality"""
        from zapgpt.config import get_pricing_file_path

        pricing_path = get_pricing_file_path()
        assert isinstance(pricing_path, Path)
        assert pricing_path.name == "pricing.json"


class TestPromptLoading:
    """Test prompt loading functionality"""

    def test_prompt_directory_structure(self):
        """Test that prompt directory exists and has prompts"""
        import glob

        from zapgpt.main import current_script_path

        prompts_dir = os.path.join(current_script_path, "prompts")
        assert os.path.exists(prompts_dir)

        # Should have some JSON prompt files
        prompt_files = glob.glob(os.path.join(prompts_dir, "*.json"))
        assert len(prompt_files) > 0

    def test_default_prompts_exist(self):
        """Test that expected default prompts exist"""
        import json

        from zapgpt.main import current_script_path

        prompts_dir = os.path.join(current_script_path, "prompts")
        expected_prompts = ["coding", "default", "common_base"]

        for prompt_name in expected_prompts:
            prompt_file = os.path.join(prompts_dir, f"{prompt_name}.json")
            if os.path.exists(prompt_file):
                # Should be valid JSON
                with open(prompt_file) as f:
                    data = json.load(f)
                    assert isinstance(data, dict)
                    # Should have expected structure
                    assert "system_prompt" in data or "model" in data


class TestPricingConfiguration:
    """Test pricing configuration functionality"""

    def test_default_pricing_file_exists(self):
        """Test that default pricing file exists"""
        from zapgpt.main import current_script_path

        default_pricing_file = os.path.join(current_script_path, "default_pricing.json")
        assert os.path.exists(default_pricing_file)

        # Should be valid JSON
        import json

        with open(default_pricing_file) as f:
            data = json.load(f)
            assert isinstance(data, dict)

    def test_pricing_data_loading(self):
        """Test pricing data loading functionality"""
        from zapgpt.main import load_pricing_data

        # Should not raise an exception
        pricing_data = load_pricing_data()
        assert isinstance(pricing_data, dict)


class TestEnvironmentVariables:
    """Test environment variable handling"""

    def test_provider_env_var_mapping(self):
        """Test that provider environment variable mapping is complete"""
        from zapgpt.main import provider_env_vars, provider_map

        # Every provider should have an environment variable
        for provider in provider_map.keys():
            assert provider in provider_env_vars
            env_var = provider_env_vars[provider]
            assert isinstance(env_var, str)
            assert len(env_var) > 0

    def test_env_var_names_are_valid(self):
        """Test that environment variable names follow conventions"""
        from zapgpt.main import provider_env_vars

        for _provider, env_var in provider_env_vars.items():
            # Should be uppercase
            assert env_var.isupper()
            # Should contain API or KEY or TOKEN
            assert any(word in env_var for word in ["API", "KEY", "TOKEN"])


class TestConfigurationDisplay:
    """Test configuration display functionality"""

    def test_show_config_info_import(self):
        """Test that show_config_info can be imported"""
        from zapgpt.config import show_config_info

        assert callable(show_config_info)

    @patch("builtins.print")
    def test_show_config_info_execution(self, mock_print):
        """Test that show_config_info executes without error"""
        from zapgpt.config import show_config_info

        # Should not raise an exception
        show_config_info()

        # Should have printed something
        assert mock_print.called


if __name__ == "__main__":
    pytest.main([__file__])
