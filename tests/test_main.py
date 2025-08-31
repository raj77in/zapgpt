"""
Test cases for main.py
"""

import json
import os
import sys
from argparse import ArgumentTypeError
from unittest.mock import patch

import pytest

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zapgpt.main import (
    BaseLLMClient,
    DeepInfraClient,
    OpenAIClient,
    OpenRouterClient,
    OutputHandler,
    ReplicateClient,
    TogetherClient,
    color_cost,
    fmt_colored,
    get_filenames_without_extension,
    load_pricing_data,
    match_abbreviation,
    pretty,
)


# Fixtures
@pytest.fixture
def output_handler():
    return OutputHandler()


@pytest.fixture
def mock_config_dir(tmp_path, monkeypatch):
    """Create a mock config directory for testing."""
    config_dir = tmp_path / ".config" / "zapgpt"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create a test pricing file
    pricing_data = {
        "openai": {
            "gpt-4": {"prompt_tokens": 0.03, "completion_tokens": 0.06},
            "gpt-3.5-turbo": {"prompt_tokens": 0.0015, "completion_tokens": 0.002},
        },
        "together": {
            "togethercomputer/llama-2-70b-chat": {
                "prompt_tokens": 0.0009,
                "completion_tokens": 0.0009,
            }
        },
        "deepinfra": {
            "meta-llama/Llama-2-70b-chat-hf": {
                "prompt_tokens": 0.0007,
                "completion_tokens": 0.0009,
            }
        },
        "replicate": {
            "meta/llama-2-70b-chat": {
                "prompt_tokens": 0.0007,
                "completion_tokens": 0.0009,
            }
        },
    }

    with open(config_dir / "pricing.json", "w") as f:
        json.dump(pricing_data, f)

    # Create a test prompt file
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    test_prompt = """
    You are a helpful AI assistant.
    Model: {model}
    """

    with open(prompts_dir / "test_prompt.txt", "w") as f:
        f.write(test_prompt)

    # Patch the config directory paths
    monkeypatch.setattr("zapgpt.main.CONFIG_DIR", config_dir)
    monkeypatch.setattr("zapgpt.main.USER_PROMPTS_DIR", str(prompts_dir))

    return config_dir


# Test OutputHandler
class TestOutputHandler:
    def test_print(self, capsys):
        handler = OutputHandler()
        handler.print("Test message")
        captured = capsys.readouterr()
        assert captured.out == "Test message\n"

    def test_console_print_quiet(self, capsys):
        handler = OutputHandler(quiet_mode=True)
        handler.console_print("Test message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_console_print_verbose(self, capsys):
        handler = OutputHandler(quiet_mode=False)
        handler.console_print("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_error_quiet(self, capsys):
        handler = OutputHandler(quiet_mode=True)
        handler.error("Error message")
        captured = capsys.readouterr()
        assert "Error message" in captured.err

    def test_success_quiet(self, capsys):
        handler = OutputHandler(quiet_mode=True)
        handler.success("Success message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_info_quiet(self, capsys):
        handler = OutputHandler(quiet_mode=True)
        handler.info("Info message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_warning_quiet(self, capsys):
        handler = OutputHandler(quiet_mode=True)
        handler.warning("Warning message")
        captured = capsys.readouterr()
        # Warning messages go to stderr in quiet mode
        assert "Warning message" in captured.err


# Test utility functions
def test_pretty():
    # Test with a long float
    result = pretty(1.23456789123)
    assert result.startswith("1.234567891")  # Check first 10 decimal places

    # Test with integer-like float
    assert pretty(1.0) in ("1.0", "1")

    # Test with very small decimal part
    assert pretty(1.00000000001) in ("1.0", "1")


def test_color_cost():
    assert "0.0" in color_cost(0.0)
    assert "0.0" in color_cost("0.0")
    assert "0.0" in color_cost("0.0000")


def test_fmt_colored():
    # Test with integer
    result = fmt_colored(0)
    assert "0" in result or "0.0" in result

    # Test with float
    result = fmt_colored(0.0)
    assert "0.0" in result

    # Test with string (should be returned as-is)
    assert fmt_colored("test") == "test"


def test_get_filenames_without_extension(tmp_path):
    # Create test files with different extensions
    test_files = ["test1.txt", "test2.md", "test3.py"]
    for fname in test_files:
        (tmp_path / fname).touch()

    # Create a subdirectory to ensure it's not included
    (tmp_path / "subdir").mkdir()

    # Call the function and get results
    files = get_filenames_without_extension(str(tmp_path))

    # Check that all expected base names are present
    assert len(files) == len(test_files)
    for fname in test_files:
        base = os.path.splitext(fname)[0]
        assert base in files, f"Expected {base} in results"


def test_match_abbreviation():
    # Test with list
    options = ["apple", "banana", "cherry"]
    matcher = match_abbreviation(options)
    assert matcher("app") == "apple"
    assert matcher("b") == "banana"
    assert matcher("c") == "cherry"  # Single match for 'c' with these options

    # Test with dict
    options_dict = {"apple": 1, "banana": 2, "cherry": 3}
    matcher = match_abbreviation(options_dict)
    assert matcher("app") == "apple"
    assert matcher("b") == "banana"
    assert matcher("c") == "cherry"  # Single match for 'c' with these options

    # Test exact match
    assert matcher("banana") == "banana"

    # Test no match - should raise ArgumentTypeError with a specific message
    with pytest.raises(ArgumentTypeError) as excinfo:
        matcher("xyz")
    assert "Invalid input 'xyz' → expected one of: apple, banana, cherry" in str(
        excinfo.value
    )

    # Test with ambiguous options
    options_ambiguous = ["cat", "car", "dog"]
    matcher = match_abbreviation(options_ambiguous)
    with pytest.raises(ArgumentTypeError) as excinfo:
        matcher("c")  # 'c' is ambiguous between 'cat' and 'car'
    assert "Ambiguous input 'c' → matches: " in str(excinfo.value)


def test_ensure_config_directory(tmp_path, monkeypatch):
    # Mock the home directory and required attributes
    monkeypatch.setattr("os.path.expanduser", lambda _: str(tmp_path))

    # Import the module after patching to ensure the patch takes effect
    import sys

    if "zapgpt.main" in sys.modules:
        del sys.modules["zapgpt.main"]

    # Set environment variables for config paths
    import os

    config_dir = tmp_path / ".config" / "zapgpt"
    os.environ["ZAPGPT_CONFIG_DIR"] = str(config_dir)

    # Import after setting environment variables
    from unittest.mock import MagicMock

    from zapgpt.main import CONFIG_DIR, USER_PROMPTS_DIR, ensure_config_directory

    # Create a mock logger
    mock_logger = MagicMock()

    # Ensure the directory doesn't exist yet
    if os.path.exists(CONFIG_DIR):
        import shutil

        shutil.rmtree(CONFIG_DIR)

    # Call the function with the mock logger
    ensure_config_directory(logger_instance=mock_logger)

    # Verify the directories were created
    assert os.path.exists(CONFIG_DIR), (
        f"Config directory was not created at {CONFIG_DIR}"
    )
    assert os.path.exists(USER_PROMPTS_DIR), (
        f"Prompts directory was not created at {USER_PROMPTS_DIR}"
    )

    # Call again to test idempotency (should not raise exceptions)
    ensure_config_directory(logger_instance=mock_logger)


def test_load_pricing_data(mock_config_dir):
    """Test loading pricing data from the config directory."""
    import json
    import sys

    if "zapgpt.main" in sys.modules:
        del sys.modules["zapgpt.main"]

    # Create test pricing data matching the actual default pricing structure
    test_pricing = {
        "openai": {
            "gpt-5": {"prompt_tokens": 0.00125, "completion_tokens": 0.01},
            "gpt-5-mini": {"prompt_tokens": 0.00025, "completion_tokens": 0.002},
        }
    }

    # Write test pricing file
    pricing_file = mock_config_dir / "pricing.json"
    pricing_file.write_text(json.dumps(test_pricing))

    # Test loading pricing data
    pricing_data = load_pricing_data()

    # Verify the structure of the returned data
    assert isinstance(pricing_data, dict), "Pricing data should be a dictionary"

    # Check for expected provider (only 'openai' is in the default pricing)
    assert "openai" in pricing_data, "Expected 'openai' in pricing data"

    # Verify the structure of the openai pricing
    assert isinstance(pricing_data["openai"], dict), (
        "OpenAI pricing should be a dictionary"
    )
    assert len(pricing_data["openai"]) > 0, "OpenAI pricing should contain models"

    # Check model pricing structure
    for provider, models in test_pricing.items():
        assert provider in pricing_data, (
            f"Provider {provider} not found in pricing data"
        )
        for model, _prices in models.items():
            assert model in pricing_data[provider], (
                f"Model {model} not found in {provider} pricing"
            )
            assert "prompt_tokens" in pricing_data[provider][model], (
                f"prompt_tokens not found for {provider}/{model}"
            )
            assert "completion_tokens" in pricing_data[provider][model], (
                f"completion_tokens not found for {provider}/{model}"
            )

            # Verify price values are numbers
            assert isinstance(
                pricing_data[provider][model]["prompt_tokens"], (int, float)
            ), (
                f"prompt_tokens should be a number, got {type(pricing_data[provider][model]['prompt_tokens'])}"
            )
            assert isinstance(
                pricing_data[provider][model]["completion_tokens"], (int, float)
            ), (
                f"completion_tokens should be a number, got {type(pricing_data[provider][model]['completion_tokens'])}"
            )


def test_show_complete_prompt(mock_config_dir, capsys, monkeypatch):
    # Import the module after setting up the mock config
    import sys

    if "zapgpt.main" in sys.modules:
        del sys.modules["zapgpt.main"]

    # Mock the console.print function to capture output
    captured_output = []

    def mock_console_print(*args, **kwargs):
        captured_output.append(" ".join(str(a) for a in args))

    # Import the module and patch the console
    import zapgpt.main

    monkeypatch.setattr("zapgpt.main.console.print", mock_console_print)

    # Create a test prompt file
    test_prompt = {"system_prompt": "This is a test prompt", "model": "test-model"}

    # Add the test prompt to the prompt_jsons
    zapgpt.main.prompt_jsons["test_prompt"] = test_prompt

    # Test showing the prompt
    zapgpt.main.show_complete_prompt("test_prompt", "gpt-4")

    # Verify the output contains expected content
    output = "\n".join(captured_output)
    assert "Complete Prompt Preview: 'test_prompt'" in output
    assert "gpt-4" in output


# Test BaseLLMClient
class TestBaseLLMClient:
    def test_init(self):
        client = BaseLLMClient("test-model")
        assert client.model == "test-model"
        assert client.temperature == 0.7

        # Check that the database was created
        assert os.path.exists(os.path.expanduser("~/.config/zapgpt/gpt_usage.db"))

    def test_count_tokens(self):
        client = BaseLLMClient("test-model")
        messages = [{"role": "user", "content": "Hello, world!"}]
        token_count = client.count_tokens(messages)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_price(self):
        # Create a test client with mock pricing data
        client = BaseLLMClient("test-model")
        client.price = {
            "test-model": {
                "prompt_tokens": "0.01",  # per 1K tokens
                "completion_tokens": "0.02",  # per 1K tokens
            }
        }

        # Test with known model
        # (1000/1000 * 0.01) + (500/1000 * 0.02) = 0.01 + 0.01 = 0.02
        assert abs(client.get_price("test-model", 1000, 500) - 0.02) < 0.001

        # Test with unknown model (should use default pricing of 0.03)
        # (1000/1000 * 0.03) + (500/1000 * 0.03) = 0.03 + 0.015 = 0.045
        assert abs(client.get_price("unknown-model", 1000, 500) - 0.045) < 0.001


# Test OpenAIClient
class TestOpenAIClient:
    def test_init(self, mock_config_dir):
        client = OpenAIClient(api_key="test-key")
        assert client.model == "openai/gpt-4o-mini"
        assert client.temperature == 0.7
        assert client.api_key == "test-key"

    @patch("zapgpt.main.OpenAI")
    def test_build_payload(self, mock_openai):
        client = OpenAIClient(api_key="test-key")
        payload = client.build_payload("Test prompt")

        assert "messages" in payload
        assert len(payload["messages"]) > 0
        assert payload["messages"][0]["content"] == "Test prompt"
        assert "temperature" in payload

    def test_get_endpoint(self):
        client = OpenAIClient(api_key="test-key")
        assert "api.openai.com" in client.get_endpoint()


# Test OpenRouterClient
class TestOpenRouterClient:
    def test_init(self, mock_config_dir):
        client = OpenRouterClient(api_key="test-key")
        assert client.model == "openai/gpt-4o-mini"
        assert client.temperature == 0.7
        assert client.api_key == "test-key"

    def test_get_endpoint(self):
        client = OpenRouterClient(api_key="test-key")
        assert "openrouter.ai" in client.get_endpoint()


# Test ReplicateClient
class TestReplicateClient:
    def test_init(self):
        client = ReplicateClient(model="test-model", api_key="test-key")
        assert client.model == "test-model"
        assert client.api_key == "test-key"

    def test_get_headers(self):
        client = ReplicateClient(model="test-model", api_key="test-key")
        headers = client.get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Token test-key"

    def test_get_endpoint(self):
        client = ReplicateClient(model="test-model", api_key="test-key")
        assert "api.replicate.com" in client.get_endpoint()


# Test TogetherClient
class TestTogetherClient:
    def test_get_endpoint(self):
        client = TogetherClient(model="test-model")
        assert "api.together.xyz" in client.get_endpoint()


# Test DeepInfraClient
class TestDeepInfraClient:
    def test_get_endpoint(self):
        client = DeepInfraClient(model="test-model")
        assert "api.deepinfra.com" in client.get_endpoint()
