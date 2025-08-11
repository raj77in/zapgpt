"""
Safe tests for configuration functionality in main.py that don't delete source files.
"""

import json
import shutil
from unittest.mock import MagicMock, patch

import pytest

# Import the module without patching first

# Test data that matches the actual pricing data format
TEST_PRICING_DATA = {
    "openai": {
        "gpt-3.5-turbo": {"prompt_tokens": 0.0015, "completion_tokens": 0.002},
        "gpt-4": {"prompt_tokens": 0.03, "completion_tokens": 0.06},
    }
}


@pytest.fixture
def safe_test_env(tmp_path):
    """Create a safe test environment with temporary files."""
    # Create a test directory structure
    test_dir = tmp_path / "test_zapgpt"
    test_dir.mkdir()

    # Create a test prompts directory
    test_prompts_dir = test_dir / "prompts"
    test_prompts_dir.mkdir()

    # Create a test prompt file
    test_prompt_file = test_prompts_dir / "test_prompt.json"
    with open(test_prompt_file, "w", encoding="utf-8") as f:
        json.dump({"test": "prompt"}, f)

    # Create a test default_pricing.json
    test_pricing_file = test_dir / "default_pricing.json"
    with open(test_pricing_file, "w", encoding="utf-8") as f:
        json.dump(TEST_PRICING_DATA, f)

    # Create a config directory that doesn't exist yet
    config_dir = tmp_path / ".config" / "zapgpt"
    prompts_dir = config_dir / "prompts"

    # Ensure the config directory doesn't exist before the test
    if config_dir.exists():
        shutil.rmtree(config_dir)

    return {
        "test_dir": test_dir,
        "test_prompts_dir": test_prompts_dir,
        "test_prompt_file": test_prompt_file,
        "test_pricing_file": test_pricing_file,
        "config_dir": config_dir,
        "prompts_dir": prompts_dir,
    }


class TestConfigSafe:
    """Safe tests for configuration functionality."""

    def test_ensure_config_directory_safe(self, safe_test_env, tmp_path):
        """Test that ensure_config_directory works without deleting source files."""

        from zapgpt.main import ensure_config_directory

        # Setup test environment
        config_dir = safe_test_env["config_dir"]
        prompts_dir = safe_test_env["prompts_dir"]

        # Create a mock logger
        mock_logger = MagicMock()

        # Import the module to get the actual CONFIG_DIR

        # Patch the necessary paths and functions
        with (
            patch("zapgpt.main.CONFIG_DIR", str(config_dir)),
            patch("zapgpt.main.USER_PROMPTS_DIR", str(prompts_dir)),
            patch("zapgpt.main.copy_default_pricing_to_config") as mock_copy_pricing,
            patch("os.path.exists", return_value=True),
            patch("os.makedirs"),
        ):
            # Configure mock for copy_default_pricing_to_config
            mock_copy_pricing.return_value = True

            # Run the function
            ensure_config_directory(logger_instance=mock_logger)

            # Verify directories were created
            mock_logger.debug.assert_any_call(
                f"[ensure_config_directory] Ensuring config directory exists: {config_dir}"
            )
            mock_logger.debug.assert_any_call(
                f"[ensure_config_directory] Ensuring prompts directory exists: {prompts_dir}"
            )

            # Verify pricing file was handled - it might be called multiple times
            assert mock_copy_pricing.call_count >= 1, (
                "Expected copy_default_pricing_to_config to be called at least once"
            )
            mock_copy_pricing.assert_any_call(
                force_update=True, logger_instance=mock_logger
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
