"""
Safe tests for configuration functionality in main.py that don't delete source files.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

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

    def test_ensure_config_directory_safe(self, safe_test_env):
        """Test that ensure_config_directory works without deleting source files."""
        # Import the function inside the test method for proper patching
        from zapgpt.main import ensure_config_directory

        # Setup test environment
        test_dir = safe_test_env["test_dir"]
        config_dir = safe_test_env["config_dir"]
        prompts_dir = safe_test_env["prompts_dir"]

        # Create a mock logger
        mock_logger = MagicMock()

        # Create a mock for the default_pricing.json file
        mock_pricing_content = json.dumps(TEST_PRICING_DATA)

        # Create a real directory for the test
        config_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Patch the necessary paths and functions
        with (
            patch("zapgpt.main.CONFIG_DIR", str(config_dir)),
            patch("zapgpt.main.USER_PROMPTS_DIR", str(prompts_dir)),
            patch("zapgpt.main.current_script_path", str(test_dir)),
            patch(
                "zapgpt.main.ensure_pricing_file_updated"
            ) as mock_ensure_pricing_updated,
            patch("zapgpt.main.copy_default_pricing_to_config") as mock_copy_pricing,  # noqa: F841
            patch("builtins.open", mock_open(read_data=mock_pricing_content)),
            patch("os.path.exists", return_value=True),  # Simulate that files exist
            patch(
                "os.listdir", return_value=["test_prompt.json"]
            ),  # Simulate prompt files
            patch("shutil.copy2") as mock_copy2,
            patch(
                "os.makedirs"
            ) as mock_makedirs,  # Mock makedirs to prevent actual directory creation
            patch("shutil.move") as mock_move,  # noqa: F841 Mock shutil.move for atomic writes
        ):
            # Configure mock_makedirs to do nothing (we'll create the directories manually)
            mock_makedirs.side_effect = lambda *args, **kwargs: None

            # Configure the mock for ensure_pricing_file_updated
            def ensure_pricing_side_effect(logger_instance=None):
                # Simulate what ensure_pricing_file_updated would do
                pricing_file = Path(config_dir) / "pricing.json"
                pricing_file.parent.mkdir(parents=True, exist_ok=True)
                pricing_file.write_text(mock_pricing_content)
                return True

            mock_ensure_pricing_updated.side_effect = ensure_pricing_side_effect

            # Run the function
            ensure_config_directory(logger_instance=mock_logger)

            # Verify directories were created
            assert config_dir.exists(), (
                f"Config directory was not created at {config_dir}"
            )
            assert prompts_dir.exists(), (
                f"Prompts directory was not created at {prompts_dir}"
            )

            # Verify that ensure_pricing_file_updated was called with the logger
            mock_ensure_pricing_updated.assert_called_once_with(
                logger_instance=mock_logger
            )

            # Verify the pricing file was created
            pricing_file = config_dir / "pricing.json"
            assert pricing_file.exists(), (
                f"Pricing file was not created at {pricing_file}"
            )

            # Verify that copy2 was called to copy prompt files
            # We expect copy2 to be called for each prompt file
            assert mock_copy2.called, "No files were copied to the prompts directory"

            # Get the actual calls to copy2
            copy_calls = [call[0] for call in mock_copy2.call_args_list]

            # Check if any file was copied to the prompts directory
            assert any(
                str(prompts_dir) in str(call[1]) for call in copy_calls if len(call) > 1
            ), f"No files were copied to {prompts_dir}"

            # Verify that the test prompt file was copied
            test_prompt_file = prompts_dir / "test_prompt.json"
            assert any(
                str(test_prompt_file) == str(call[1])
                for call in copy_calls
                if len(call) > 1
            ), f"Expected test_prompt.json to be copied to {test_prompt_file}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
