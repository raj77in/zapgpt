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

        # Import the module to get the actual CONFIG_DIR
        from zapgpt import main

        # Save original values
        original_config_dir = getattr(main, 'CONFIG_DIR', None)
        original_prompts_dir = getattr(main, 'USER_PROMPTS_DIR', None)

        try:
            # Set the attributes directly on the module
            main.CONFIG_DIR = str(config_dir)
            main.USER_PROMPTS_DIR = str(prompts_dir)

            # Patch the remaining functions
            with (
                patch("zapgpt.main.copy_default_pricing_to_config") as mock_copy_pricing,
                patch("os.path.exists", return_value=True),
                patch("os.makedirs"),
            ):
                # Configure mock for copy_default_pricing_to_config
                mock_copy_pricing.return_value = True

                # Run the function
                ensure_config_directory(logger_instance=mock_logger)
        finally:
            # Restore original values
            if original_config_dir is not None:
                main.CONFIG_DIR = original_config_dir
            if original_prompts_dir is not None:
                main.USER_PROMPTS_DIR = original_prompts_dir

            # Verify the function performed the expected operations
            # Check if the config directory was created
            mock_logger.debug.assert_any_call(
                f"[ensure_config_directory] Ensuring pricing file is up to date"
            )

            # Verify the prompts directory check was performed
            from unittest.mock import call
            mock_calls = [str(c) for c in mock_logger.debug.mock_calls]
            assert any("[ensure_config_directory] Found prompts directory at:" in str(c) for c in mock_logger.debug.mock_calls), \
                f"Expected log message about prompts directory not found in: {mock_calls}"

            # Verify pricing file was handled - check it was called with the expected arguments
            # The function might be called multiple times, so we check for the expected calls
            mock_copy_pricing.assert_any_call(force_update=True, logger_instance=mock_logger)

            # Verify the number of calls is as expected (2 in this case)
            assert mock_copy_pricing.call_count == 2, \
                f"Expected 2 calls to copy_default_pricing_to_config, got {mock_copy_pricing.call_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
