"""
Unit tests for the cli.py module.
"""

import sys
from unittest.mock import patch

import pytest

# Import the module under test
from zapgpt.cli import cli


def test_cli_function_calls_main():
    """Test that the cli() function calls main() with sys.argv[1:]."""
    with patch("zapgpt.cli.main") as mock_main:
        # Save original argv
        old_argv = sys.argv

        # Set up test arguments
        test_args = ["zapgpt", "--help"]
        sys.argv = test_args

        try:
            # Call the cli function
            cli()

            # Verify main was called with the correct arguments
            mock_main.assert_called_once()
        finally:
            # Restore original argv
            sys.argv = old_argv


def test_cli_function_without_args():
    """Test that the cli() function works with no arguments."""
    with patch("zapgpt.cli.main") as mock_main:
        # Save original argv
        old_argv = sys.argv

        # Set up test with no arguments
        test_args = ["zapgpt"]
        sys.argv = test_args

        try:
            # Call the cli function
            cli()

            # Verify main was called with no arguments
            mock_main.assert_called_once()
        finally:
            # Restore original argv
            sys.argv = old_argv


@patch("zapgpt.cli.main")
def test_cli_module_execution(mock_main):
    """Test that the cli module can be executed as __main__."""
    with patch("zapgpt.cli.__name__", "__main__"):
        # Save original argv
        old_argv = sys.argv

        # Set up test arguments
        test_args = ["zapgpt", "--version"]
        sys.argv = test_args

        try:
            # Execute the module
            with patch("sys.exit") as mock_exit:
                cli()

                # Verify main was called
                mock_main.assert_called_once()

                # Verify sys.exit was not called (no error)
                mock_exit.assert_not_called()
        finally:
            # Restore original argv
            sys.argv = old_argv


@patch("zapgpt.cli.main")
def test_cli_module_execution_with_error(mock_main):
    """Test that the cli module handles errors properly."""
    # Set up the mock to raise SystemExit
    mock_main.side_effect = SystemExit(1)

    with patch("zapgpt.cli.__name__", "__main__"):
        # Save original argv
        old_argv = sys.argv

        # Set up test arguments
        test_args = ["zapgpt", "--invalid-flag"]
        sys.argv = test_args

        try:
            # Execute the module and expect SystemExit
            with pytest.raises(SystemExit) as excinfo:
                cli()

            # Verify the exit code is 1
            assert excinfo.value.code == 1

            # Verify main was called
            mock_main.assert_called_once()
        finally:
            # Restore original argv
            sys.argv = old_argv


if __name__ == "__main__":
    pytest.main([__file__])
