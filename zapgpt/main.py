#!/usr/bin/env python3
"""
ZapGPT - A command-line tool for interacting with various LLM providers
======================================================================
Author: Amit Agarwal (aka)
Created: 2025-05-06
Updated: 2025-08-10

A minimalist CLI tool to chat with LLMs from your terminal. Features include:
- Support for multiple LLM providers (OpenAI, Anthropic, Mistral, etc.)
- Usage and cost tracking with local SQLite database
- User-customizable prompt management system
- Rich CLI output and logging
- Configuration stored in ~/.config/zapgpt/

Installation:
    pip install zapgpt

Usage:
    zapgpt "Your question here"
    zapgpt --use-prompt coding "Refactor this function"
    zapgpt --list-prompt
    zapgpt --config
    zapgpt --show-prompt coding

Configuration:
    Configuration and prompts are stored in ~/.config/zapgpt/
    - Prompts: ~/.config/zapgpt/prompts/
    - Database: ~/.config/zapgpt/gpt_usage.db

Dependencies:
    See pyproject.toml or requirements.txt
"""

# ===============================
# Standard Library Imports
# ===============================
import glob
import json  # For config and API responses
import logging  # For logging actions and errors
import os  # For environment variables and file paths
import re  # For regex operations
import sqlite3  # For usage tracking
import sys  # For system exit and arguments
import time  # For time calculations
from datetime import datetime  # For timestamps

# Get version from pyproject.toml
try:
    import tomli

    with open("pyproject.toml", "rb") as f:
        VERSION = tomli.load(f)["project"]["version"]
except (ImportError, FileNotFoundError, KeyError):
    # Fallback version if pyproject.toml is not available
    VERSION = "3.4.0"
from argparse import ArgumentParser, ArgumentTypeError  # For CLI parsing
from pathlib import Path  # For path manipulations
from textwrap import dedent  # For help/epilog formatting
from typing import Union  # For type hints

# ===============================
# Third-Party Imports
# ===============================
import requests  # For HTTP requests to APIs
import tiktoken  # For token counting
from openai import OpenAI  # OpenAI API client
from pygments import highlight  # For syntax highlighting
from pygments.formatters import TerminalFormatter  # For terminal color formatting
from pygments.lexers import get_lexer_by_name  # For syntax highlighting
from rich.console import Console  # For rich terminal output
from rich.logging import RichHandler  # For rich logging output
from rich.markdown import Markdown  # For markdown rendering
from rich.table import Table  # For pretty tables
from rich_argparse import RichHelpFormatter  # For rich CLI help
from tabulate import tabulate  # For table formatting

# ===============================
# Configuration and Setup
# ===============================


class OutputHandler:
    """Centralized output handler that respects quiet mode"""

    def __init__(self, quiet_mode=False):
        self.quiet_mode = quiet_mode
        self.console = Console(file=sys.stderr if quiet_mode else None)

    def print(self, *args, **kwargs):
        """Print to stdout (always shown, for data output)"""
        print(*args, **kwargs)

    def console_print(self, *args, **kwargs):
        """Styled console output (hidden in quiet mode)"""
        if not self.quiet_mode:
            self.console.print(*args, **kwargs)

    def error(self, message, styled_message=None):
        """Error output (always shown, but styled only if not quiet)"""
        if self.quiet_mode:
            print(f"❌ {message}", file=sys.stderr)
        else:
            self.console.print(styled_message or f"[red]❌ {message}[/red]")

    def success(self, message, styled_message=None):
        """Success output (hidden in quiet mode)"""
        if not self.quiet_mode:
            self.console.print(styled_message or f"[green]✅ {message}[/green]")

    def info(self, message, styled_message=None):
        """Info output (hidden in quiet mode)"""
        if not self.quiet_mode:
            self.console.print(styled_message or f"[blue]ℹ️ {message}[/blue]")

    def warning(self, message, styled_message=None):
        """Warning output (shown in quiet mode as plain text)"""
        if self.quiet_mode:
            print(f"⚠️ {message}", file=sys.stderr)
        else:
            self.console.print(styled_message or f"[yellow]⚠️ {message}[/yellow]")


# Global output handler (will be initialized in main)
output = None

# Use RichHandler for pretty, colorized terminal logs.
console = Console()
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    handlers=[RichHandler(rich_tracebacks=True)],
    format="%(message)s",
)
logger = logging.getLogger("llm")

# ===============================
# Script Configuration
# ===============================
current_script_path = str(Path(__file__).resolve().parent)
# Store database in user config directory or use environment variable
CONFIG_DIR = os.path.expanduser("~/.config/zapgpt")
DB_FILE = os.environ.get("ZAPGPT_DB_PATH") or os.path.join(CONFIG_DIR, "gpt_usage.db")

# Ensure the directory exists
os.makedirs(os.path.dirname(os.path.abspath(DB_FILE)), exist_ok=True)


# Configuration directory setup
USER_PROMPTS_DIR = os.path.join(CONFIG_DIR, "prompts")


def ensure_pricing_file_updated(logger_instance=None):
    """Ensure the pricing file is updated to the latest version.

    Args:
        logger_instance: Optional logger instance to use for logging. If not provided,
                       the global logger will be used.

    Returns:
        bool: True if the pricing file was updated, False otherwise.
    """
    logger.debug("Ensuring pricing file is up to date")
    return copy_default_pricing_to_config(
        force_update=True, logger_instance=logger_instance
    )


def copy_default_pricing_to_config(force_update=False, logger_instance=None):
    """Copy default pricing file from package to user config directory.

    Args:
        force_update (bool): If True, update the pricing file even if it exists.
        logger_instance: Optional logger instance to use for logging. If not provided,
                       the global logger will be used.

    Returns:
        bool: True if the file was created/updated, False if no action was taken.
    """
    # Use provided logger or fall back to global logger
    log = logger_instance or logger

    pricing_file = os.path.join(CONFIG_DIR, "pricing.json")
    needs_update = force_update

    # Check if the file exists and compare versions if not forcing update
    if not needs_update and os.path.exists(pricing_file):
        try:
            # Try to get version from installed package
            from importlib.metadata import version

            current_version = version("zapgpt")

            # Try to get version from existing pricing file
            with open(pricing_file, encoding="utf-8") as f:
                existing_data = json.load(f)
                if existing_data.get("_version") != current_version:
                    log.info(
                        f"Pricing file version mismatch, updating to version {current_version}"
                    )
                    needs_update = True
        except Exception as e:
            log.debug(f"Could not check versions, forcing update: {e}")
            needs_update = True

    # If we don't need to update, return early
    if not needs_update and os.path.exists(pricing_file):
        log.debug("Pricing file is up to date")
        return False

    # Ensure the config directory exists
    os.makedirs(os.path.dirname(pricing_file), exist_ok=True)

    # First, try to load from package resources (installed package)
    try:
        # Try Python 3.9+ importlib.resources.files
        from importlib import resources

        if hasattr(resources, "files"):  # Python 3.9+
            pricing_pkg = resources.files("zapgpt") / "default_pricing.json"
            content = pricing_pkg.read_text(encoding="utf-8")
        else:  # Python < 3.9 fallback
            import pkg_resources

            content = pkg_resources.resource_string(
                "zapgpt", "default_pricing.json"
            ).decode("utf-8")

        # Add version information
        try:
            from importlib.metadata import version

            content_dict = json.loads(content)
            content_dict["_version"] = version("zapgpt")
            content = json.dumps(content_dict, indent=2)
        except Exception as e:
            log.debug(f"Could not add version info: {e}")

        # Write the content to a temporary file first
        import shutil
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Move the temporary file to the target location atomically
        shutil.move(tmp_path, pricing_file)

        log.info(f"Updated pricing file at {pricing_file}")
        return True

    except (ImportError, FileNotFoundError, Exception) as e:
        log.debug(f"Failed to load pricing from package resources: {e}")

        # Fallback to development directory
        dev_pricing_file = os.path.join(current_script_path, "default_pricing.json")
        if os.path.exists(dev_pricing_file):
            try:
                # Use a temporary file to ensure atomic write
                import tempfile

                with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
                    with open(dev_pricing_file, "rb") as src_file:
                        shutil.copyfileobj(src_file, tmp_file)
                    tmp_path = tmp_file.name

                # Move the temporary file to the target location atomically
                shutil.move(tmp_path, pricing_file)
                log.info(f"Copied pricing from development directory to {pricing_file}")
                return True
            except Exception as e:
                log.error(f"Failed to copy pricing from development directory: {e}")
                # Clean up the temporary file if it exists
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        # Ignore errors when trying to delete temp file
                        pass
        else:
            log.error(f"Development pricing file not found at {dev_pricing_file}")

        # If we got here, all attempts failed
        log.error(f"Failed to create/update pricing file at {pricing_file}")
        return False


def load_pricing_data(logger_instance=None):
    """Load pricing data from the config directory.

    Args:
        logger_instance: Optional logger instance to use for logging. If not provided,
                       the global logger will be used.

    Returns:
        dict: The loaded pricing data, or an empty dict if the file is missing or invalid.
    """
    # Use provided logger or fall back to global logger
    log = logger_instance or logger

    pricing_file = os.path.join(CONFIG_DIR, "pricing.json")

    # Always return an empty dict if the file doesn't exist
    if not os.path.exists(pricing_file):
        log.warning(f"Pricing file not found at {pricing_file}")
        return {}

    try:
        # Read and parse the file
        with open(pricing_file, encoding="utf-8") as f:
            pricing_data = json.load(f)

        # If we got here, the file exists and is valid JSON
        log.debug(f"Loaded pricing data from {pricing_file}")
        return pricing_data

    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Failed to load pricing data: {e}")
        return {}
    except Exception as e:
        log.error(f"Unexpected error loading pricing data: {e}")
        return {}

    # Fallback to empty dict
    return {}


# Initialize configuration directory
def ensure_config_directory(logger_instance=None):
    """Ensure the configuration directory and required files exist.

    Args:
        logger_instance: Optional logger instance to use for logging. If not provided,
                       the global logger will be used.
    """
    # Use provided logger or fall back to global logger
    log = logger_instance or logger

    log.debug(
        f"[ensure_config_directory] Ensuring config directory exists: {CONFIG_DIR}"
    )
    os.makedirs(CONFIG_DIR, exist_ok=True)
    log.debug(
        f"[ensure_config_directory] Ensuring prompts directory exists: {USER_PROMPTS_DIR}"
    )
    os.makedirs(USER_PROMPTS_DIR, exist_ok=True)

    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log.debug(f"[ensure_config_directory] Current script directory: {current_dir}")

    # Look for prompts in the package directory
    default_prompts_dir = os.path.join(current_dir, "prompts")
    log.debug(
        f"[ensure_config_directory] Looking for default prompts in: {default_prompts_dir}"
    )

    # Also check in the parent directory (for development)
    parent_prompts_dir = os.path.join(os.path.dirname(current_dir), "prompts")
    log.debug(
        f"[ensure_config_directory] Also checking parent directory: {parent_prompts_dir}"
    )

    # Try both possible locations for the prompts directory
    prompts_dirs = [default_prompts_dir, parent_prompts_dir]
    found_prompts = False

    for prompts_dir in prompts_dirs:
        if os.path.exists(prompts_dir):
            log.debug(
                f"[ensure_config_directory] Found prompts directory at: {prompts_dir}"
            )
            found_prompts = True

            # Copy all JSON files from the prompts directory
            for prompt_file in os.listdir(prompts_dir):
                if prompt_file.endswith(".json"):
                    src_file = os.path.join(prompts_dir, prompt_file)
                    dest_file = os.path.join(USER_PROMPTS_DIR, prompt_file)

                    log.debug(
                        f"[ensure_config_directory] Processing prompt file: {src_file} -> {dest_file}"
                    )

                    try:
                        import filecmp
                        import shutil

                        # Check if files are different before copying
                        if os.path.exists(dest_file) and filecmp.cmp(
                            src_file, dest_file, shallow=False
                        ):
                            log.debug(
                                f"Prompt file {prompt_file} is up to date, skipping"
                            )
                            continue

                        # Copy the file if it's new or different
                        shutil.copy2(src_file, dest_file)
                        log.debug(
                            f"[ensure_config_directory] Successfully copied {src_file} to {dest_file}"
                        )
                    except Exception as e:
                        log.error(
                            f"[ensure_config_directory] Failed to copy {src_file} to {dest_file}: {e}"
                        )

            # If we found and processed a prompts directory, we can stop looking
            break

    if not found_prompts:
        log.warning(
            f"[ensure_config_directory] No prompts directory found in any of these locations: {prompts_dirs}"
        )

    # Ensure pricing file is up to date
    log.debug("[ensure_config_directory] Ensuring pricing file is up to date")
    copy_default_pricing_to_config(force_update=True, logger_instance=log)
    ensure_pricing_file_updated(logger_instance=log)


# Load prompts from user configuration directory
prompt_jsons = {}
if os.path.exists(USER_PROMPTS_DIR):
    for prompt_file in glob.glob(os.path.join(USER_PROMPTS_DIR, "*.json")):
        name = os.path.splitext(os.path.basename(prompt_file))[0]
        with open(prompt_file, encoding="utf-8") as f:
            try:
                prompt_jsons[name] = json.load(f)
                logger.debug(f"Loaded prompt: {name}")
            except Exception as e:
                logger.error(f"Failed to load prompt file {prompt_file}: {e}")

if not prompt_jsons:
    logger.warning("No prompt files loaded. Some features may not work correctly.")
    logger.info(f"Add custom prompts to: {USER_PROMPTS_DIR}")
else:
    logger.debug(f"Loaded {len(prompt_jsons)} prompts from {USER_PROMPTS_DIR}")


def show_complete_prompt(prompt_name, override_model=None):
    """
    Display the complete prompt that would be sent to the LLM.
    Args:
        prompt_name (str): Name of the prompt to show
        override_model (str, optional): Model to override prompt's default model
    """
    if prompt_name not in prompt_jsons:
        console.print(f"[red]❌ Prompt '{prompt_name}' not found.[/red]")
        console.print(
            f"[yellow]Available prompts:[/yellow] {', '.join(sorted(prompt_jsons.keys()))}"
        )
        return

    prompt_data = prompt_jsons[prompt_name]
    system_prompt = prompt_data.get("system_prompt", "")
    model = override_model or prompt_data.get("model", "openai/gpt-4o-mini")
    assistant_input = prompt_data.get("assistant_input", None)

    # Add common_base if it exists and this isn't the common_base prompt itself
    if prompt_name != "common_base" and "common_base" in prompt_jsons:
        common_base_prompt = prompt_jsons["common_base"].get("system_prompt", "")
        if common_base_prompt:
            system_prompt = f"{common_base_prompt}\n\n{system_prompt}"

    console.print(
        f"\n[bold blue]📋 Complete Prompt Preview: '{prompt_name}'[/bold blue]"
    )
    console.print("=" * 60)

    console.print(f"[bold green]🤖 Model:[/bold green] {model}")
    if override_model:
        console.print("[yellow]   (Overridden from command line)[/yellow]")

    console.print("\n[bold green]💬 System Prompt:[/bold green]")
    console.print(f"[dim]{'-' * 40}[/dim]")
    if system_prompt:
        console.print(system_prompt)
    else:
        console.print("[dim](No system prompt)[/dim]")

    if assistant_input:
        console.print("\n[bold green]🤖 Assistant Input:[/bold green]")
        console.print(f"[dim]{'-' * 40}[/dim]")
        console.print(assistant_input)

    console.print(f"\n[dim]{'=' * 60}[/dim]")
    console.print(
        f'[yellow]💡 Usage:[/yellow] zapgpt --use-prompt {prompt_name} "Your question here"'
    )
    if not override_model and prompt_data.get("model"):
        console.print(
            f'[yellow]💡 Override model:[/yellow] zapgpt --use-prompt {prompt_name} -m your-model "Your question"'
        )


def pretty(x):
    """
    Format a float to 10 decimal places, removing trailing zeros.
    Args:
        x (float): The number to format.
    Returns:
        str: The formatted number as a string.
    """
    return f"{x:.10f}".rstrip("0").rstrip(".")


def color_cost(value):
    """
    Return a color-coded string for cost values.
    Args:
        value (float or str): The cost value to color.
    Returns:
        str: The color-coded string representation of the cost.
    """
    try:
        num = float(value)
        if num == 0:
            return f"[green]{num:.10f}[/green]"
        else:
            return f"[cyan]{num:.10f}[/cyan]"
    except ValueError:
        logger.error(f"Failed to color cost for invalid value: {value}")
        return "[red]Invalid[/red]"


def fmt_colored(value):
    """
    Color the value green if zero, cyan otherwise.
    If the value is a string that can't be converted to float, return it as-is.
    Args:
        value (float or str): The value to color.
    Returns:
        str: The color-coded string representation of the value.
    """
    try:
        num = float(value)
        color = "green" if num == 0 else "cyan"
        return f"[{color}]{num:.10f}[/{color}]"
    except (ValueError, TypeError):
        # If we can't convert to float, return the original value as a string
        return str(value)


def get_filenames_without_extension(folder_path):
    """
    List all filenames in a folder without their extensions.
    Args:
        folder_path (str): The path to the folder to scan.
    Returns:
        list: Filenames without their extensions.
    """
    logger.debug(f"Scanning folder for files: {folder_path}")
    filenames = []

    # Check if directory exists
    if not os.path.isdir(folder_path):
        logger.warning(f"Directory does not exist: {folder_path}")
        return filenames

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Get the base name without extension
        base_name = os.path.splitext(filename)[0]

        # Add the base name to the list
        logger.debug(f"Found file: {filename} -> {base_name}")
        filenames.append(base_name)

    logger.debug(f"Found {len(filenames)} files in {folder_path}")
    return sorted(set(filenames))  # Remove duplicates and sort


# def match_abbreviation(arg, choices):
#     """Match partial input to full option with ambiguity handling."""
#     arg = arg.casefold()
#     matches = [choice for choice in choices if choice.casefold().startswith(arg)]
#     if not matches:
#         raise argparse.ArgumentTypeError(f"Invalid choice: '{arg}' (expected one of {choices})")
#     elif len(matches) > 1:
#         raise argparse.ArgumentTypeError(f"Ambiguous choice: '{arg}' (matches: {matches})")
#     return matches[0]


def match_abbreviation(options: Union[dict, list[str]]):
    """
    Returns a function for argparse `type=` that matches partial input to full option key.
    Accepts either a list of strings or dict keys.
    Args:
        options (dict or list): The valid options to match against.
    Returns:
        function: A function that matches user input to a valid option.
    """

    # Convert to list if a dict is passed
    logger.debug(f"Preparing abbreviation matcher for options: {options}")
    valid_keys = list(options.keys()) if isinstance(options, dict) else list(options)

    def _match(value: str) -> str:
        """
        Inner matcher function for match_abbreviation.
        Belongs to: match_abbreviation (top-level function).
        Args:
            value (str): The user input to match.
        Returns:
            str: The matched option.
        Raises:
            ArgumentTypeError: If input is ambiguous or invalid.
        """
        logger.debug(f"Trying to match user input '{value}'")
        value = value.strip().lower()
        matches = [key for key in valid_keys if key.lower().startswith(value)]

        if len(matches) == 1:
            logger.info(f"Matched input '{value}' to option '{matches[0]}'")
            return matches[0]
        elif len(matches) > 1:
            raise ArgumentTypeError(
                f"Ambiguous input '{value}' → matches: {', '.join(matches)}"
            )
        else:
            logger.error(f"Invalid input '{value}' for options {valid_keys}")
            raise ArgumentTypeError(
                f"Invalid input '{value}' → expected one of: {', '.join(valid_keys)}"
            )

    return _match


class BaseLLMClient:
    """
    Base client for LLM providers.
    Handles prompt creation, logging, cost tracking, and general request flow.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        output: str = "",
        file: str = None,
    ):
        """
        Initialize a BaseLLMClient instance.
        Belongs to: BaseLLMClient class.
        Args:
            model (str): The model name.
            system_prompt (str, optional): The system prompt to use.
            max_tokens (int, optional): Maximum tokens for responses.
            temperature (float, optional): Sampling temperature.
            output (str, optional): Output file path.
            file (str, optional): Input file path.
        """
        logger.debug(
            f"Initializing BaseLLMClient with model={model}, output={output}, file={file}"
        )
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chat_history = []
        self.current_script_path = str(current_script_path)
        self.prompts_path = self.current_script_path + "/prompts"
        self.output = output
        logger.debug("File is set to {file=}")
        if file:
            self.file = file
        else:
            self.file = None
        logger.debug(f"Prompts path is {self.prompts_path=}")
        self.init_db()
        # logger.debug(f"{self.system_prompt=}")

    def init_db(self):
        """
        Create the usage database if it does not exist.
        Belongs to: BaseLLMClient class.
        """
        logger.info(f"Initializing database at {DB_FILE}")
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            """
              CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cost REAL,
                query TEXT
              )
              """
        )
        conn.commit()
        conn.close()
        logger.debug("Database initialized (if not already present)")

    def record_usage(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost: int,
        query: int,
    ):
        """
        Record a model usage event to the database.
        """
        logger.info(
            f"Recording usage for model={model}, provider={provider}, tokens={prompt_tokens + completion_tokens}, cost={cost}"
        )
        conn = sqlite3.connect(DB_FILE)
        logger.debug(f"Opened {DB_FILE=}")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO usage (timestamp, model, prompt_tokens, completion_tokens, total_tokens, cost, query)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                provider + ":" + model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
                query,
            ),
        )
        conn.commit()
        conn.close()
        logger.debug("Usage record inserted and connection closed.")

    def create_prompt(self, query: str):
        """
        Build the prompt messages for the LLM request, optionally including a system prompt and file content.
        Belongs to: BaseLLMClient class.
        Args:
            query (str): The user query or prompt.
        Returns:
            list: List of messages formatted for the LLM API.
        """
        logger.debug(f"Creating prompt for query: {query}")
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})
        # if model_prompts[self.model]:
        # logger.debug(f"Adding assistant prompt: {model_prompts[self.model]['assistant_input']}")
        # messages.append({"role": "assistant", "content": model_prompts[self.model]['assistant_input']})
        if self.file:
            logger.debug(f"File is set to {self.file=}")
            try:
                with open(self.file, encoding="utf-8") as f:
                    file_content = f.read()
            except Exception as e:
                logger.critical(f"❌ Failed to read file: {e}")
                return
            messages.append(
                {"role": "user", "content": f"File content:\n{file_content}"}
            )
        logger.debug(f"Created prompt is : {messages=}")
        return messages

    def build_payload(self, prompt: str) -> dict:
        """
        Build the payload for the API request. Should be implemented by subclasses.
        Belongs to: BaseLLMClient class.
        Args:
            prompt (str): The prompt or query for the API.
        Returns:
            dict: The payload for the API request.
        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        logger.debug("build_payload called on BaseLLMClient (should be overridden)")
        raise NotImplementedError

    def get_headers(self) -> dict:
        """
        Return HTTP headers for API requests. Should be implemented by subclasses.
        Belongs to: BaseLLMClient class.
        Returns:
            dict: HTTP headers for the API request.
        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        logger.debug("get_headers called on BaseLLMClient (should be overridden)")
        raise NotImplementedError

    def get_endpoint(self) -> str:
        """
        Return the API endpoint URL. Should be implemented by subclasses.
        Belongs to: BaseLLMClient class.
        Returns:
            str: The API endpoint URL.
        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        logger.debug("get_endpoint called on BaseLLMClient (should be overridden)")
        raise NotImplementedError

    def send_request(self, prompt: str) -> str:
        """
        Send a chat completion request to the API and return the response.
        Belongs to: BaseLLMClient class.
        Args:
            prompt (str): The user prompt or query.
        Returns:
            str: The response from the LLM API.
        """
        logger.info(f"Sending request to LLM for model={self.model}")
        self.query = prompt
        logger.debug(f"User Prompt is set to {prompt}")
        prompt = self.create_prompt(prompt)
        logger.debug(f"Created prompt is : {prompt=}")

        # prompt_tokens = count_tokens(messages, model)
        prompt_tokens = self.count_tokens(prompt, self.model)
        self.prompt_tokens = prompt_tokens
        # max_total = 128000
        # max_tokens = min(4096, max_total - prompt_tokens)  # absolute safe cap
        params = {
            "model": self.model,
            "messages": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 1.0,
        }
        logger.debug(f"Making request with {params=}")
        response = self.client.chat.completions.create(**params)
        logger.debug(f"{response=}")
        return response

    def handle_response(self, response: str) -> str:
        """
        Print and log the LLM response, cost, and usage info.
        Belongs to: BaseLLMClient class.
        Args:
            response (str): The response object from the LLM API.
        Returns:
            str: The reply from the LLM API.
        """
        logger.debug("Handling LLM response")
        logger.debug(f"{response.model_dump()}")
        logger.debug(f"Chat ID: {response.id}")
        reply = response.choices[0].message.content
        completion_tokens = float(response.usage.completion_tokens)
        total_tokens = float(response.usage.total_tokens)
        cost = self.get_price(self.model, self.prompt_tokens, completion_tokens)

        if self.output:
            logger.info(f"Writing response to output file: {self.output}")
            with open(self.output, "w") as f:
                f.writelines(reply)
        else:
            # In quiet mode, show only the raw response
            if output and output.quiet_mode:
                print(self.highlight_code(reply, lang="markdown"))
            else:
                # Normal mode: show headers and usage info
                print("\n--- RESPONSE ---\n")
                print(self.highlight_code(reply, lang="markdown"))
                print("\n--- USAGE ---")
                print(f"Chat ID: {response.id}")
                print(f"Prompt tokens: {self.prompt_tokens}")
                print(f"Completion tokens: {completion_tokens}")
                print(f"Total tokens: {total_tokens}")
                print(f"Estimated cost: ${cost:.5f}")

        self.record_usage(
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            query=self.query,
            provider="openai",
        )

    def add_to_history(self, role: str, content: str):
        """
        Add a chat message to the in-memory history.
        Belongs to: BaseLLMClient class.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
        """
        logger.debug(f"Adding to history: {role=}, content length={len(content)}")
        self.chat_history.append({"role": role, "content": content})

    def highlight_code(self, code: str, lang: str = "python") -> str:
        """
        Color syntax highlight a code block for terminal output.
        Belongs to: BaseLLMClient class.
        Args:
            code (str): The code block to highlight.
            lang (str, optional): The programming language. Defaults to 'python'.
        Returns:
            str: The highlighted code as a string for terminal output.
        """
        logger.debug(f"Highlighting code output for lang={lang}")
        lexer = get_lexer_by_name(lang, stripall=True)
        formatter = TerminalFormatter()
        return highlight(code, lexer, formatter)

    @staticmethod
    def show_history():
        """
        Print past model usage history from the database.
        Belongs to: BaseLLMClient class (staticmethod).
        """
        logger.info("Displaying model usage history from the database")
        conn = sqlite3.connect(DB_FILE)
        logger.debug(f"Opened {DB_FILE=}")
        cursor = conn.cursor()
        for row in cursor.execute(
            "SELECT timestamp, model, total_tokens, cost, query FROM usage ORDER BY timestamp DESC"
        ):
            print(
                f"[{row[0]}] model={row[1]} tokens={row[2]} cost=${row[3]:.5f}\nquery: {row[4]}...\n"
            )
        conn.close()
        logger.debug("Closed DB after history display.")

    @staticmethod
    def show_total_cost():
        """
        Print the total cost from all model usage.
        Belongs to: BaseLLMClient class (staticmethod).
        """
        logger.info("Fetching and displaying total cost from DB")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(cost) FROM usage")
        total = cursor.fetchone()[0] or 0.0
        print(f"\n💸 Total estimated cost so far: ${total:.5f}")
        conn.close()
        logger.debug("Closed DB after total cost calculation.")

    def list_available_models(self, batch=False, filter=None):
        """
        List available models from the provider, optionally filtering.
        Belongs to: BaseLLMClient class.
        Args:
            batch (bool, optional): If True, return raw model data. Defaults to False.
            filter (str, optional): Filter string for model names. Defaults to None.
        Returns:
            list or None: List of models if batch is True, otherwise prints a table.
        """
        logger.info("Fetching available model list")
        rich_table = Table(title="Model List")

        # headers = ["ID", "Created", "Description", "Context Len", "Modality", "Supported Parameters" ]
        models = self.client.models.list()
        if batch:
            logger.debug("Batch mode enabled, returning raw models data")
            return models
        rich_table.add_column("ID")
        rich_table.add_column("Created")
        if not hasattr(models, "data"):
            print("No models available")
            return
        if hasattr(models.data[0], "context_length"):
            rich_table.add_column("Ctx Len")
            rich_table.add_column("Modality")
        print("\n📦 Available OpenAI Models:")
        for m in models.data:
            # Convert created to humban-readable formatting
            m.created = datetime.fromtimestamp(m.created).strftime("%Y-%m-%d %H:%M:%S")
            # print(f"* {m.id}, Owner: {m.owned_by}, Created: {m.created}")
            # table.append([ m.id, m.created, m.description, m.context_length, m.architecture["modality"], m.supported_parameters ])
            if filter:
                logger.debug(f"Filter is set to {filter=}")
                logger.debug(f"m is {m=}")
                if filter not in m.id and filter not in getattr(m, "name", ""):
                    logger.debug(f"Fitlering out {m.id}")
                    continue
            if hasattr(m, "context_length"):
                cl = (
                    f"{int(m.context_length / 1000)} K"
                    if m.context_length > 1000
                    else str(m.context_length)
                )
                logger.debug(f"{m.context_length=} and {cl=}")
                rich_table.add_row(m.id, m.created, cl, m.architecture["modality"])
            else:
                rich_table.add_row(m.id, m.created)
        # print(tabulate(clean_table, headers=headers, tablefmt="fancy_grid", maxcolwidths=[20, 20, 35, 10, 10, 35, 10] ))
        # Table format options: plain, simple, grid, fancy_grid, github, pipe, orgtbl, mediawiki, rst, html, latex, jira, pretty
        console.print(rich_table)
        logger.debug("Model list printed.")

    def print_model_pricing_table(self, pricing_data, filter=None):
        """
        Print a table of model pricing, optionally filtered.
        Belongs to: BaseLLMClient class.
        Args:
            pricing_data (dict): Dictionary with model pricing information.
            filter (str, optional): Filter string for model names. Defaults to None.
        """
        logger.info("Printing model pricing table")
        rich_table = Table(title="Model Usage Costs")

        rich_table.add_column("Model")
        rich_table.add_column("Prompt Cost (1K)")
        rich_table.add_column("Output Cost (1K)")
        rich_table.add_column("Total (1K)")

        sorted_data = sorted(
            pricing_data.items(), key=lambda x: sum(float(v) for v in x[1].values())
        )  # or x[1][3] if total is at index 3
        for model, prices in sorted_data:
            if filter:
                if filter not in model:
                    logger.debug(f"Excluding {model} due to filter")
                    continue
            logger.debug(f"Pricing for {model}: {prices}")
            pc = prices.get("prompt_tokens", 0)
            prompt_cost = float(pc) * 1000 if isinstance(pc, str) else pc
            oc = prices.get("completion_tokens", 0)
            output_cost = float(oc) * 1000 if isinstance(oc, str) else oc
            total = (prompt_cost + output_cost) / 2
            logger.debug(f"{prompt_cost} - {output_cost} - {total}")
            logger.debug(
                f"{pretty(prompt_cost)} - {pretty(output_cost)} - {pretty(total)}"
            )
            rich_table.add_row(
                model,
                fmt_colored(prompt_cost),
                fmt_colored(output_cost),
                fmt_colored(total),
            )

        console.print(rich_table)
        logger.debug("Pricing table printed.")

    def get_tokenizer(self, model_name: str):
        """
        Return an appropriate tokenizer encoding for the given model.
        Uses heuristics if the model is not recognized by tiktoken.
        Belongs to: BaseLLMClient class.
        Args:
            model_name (str): The model name to get the tokenizer for.
        Returns:
            tiktoken.Encoding: The tokenizer encoding object.
        """
        logger.debug(f"Getting tokenizer for model_name={model_name}")
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(
                f"Model '{model_name}' not found in tiktoken. Using fallback heuristics."
            )

        # 2. Model-specific heuristics
        model_name_lower = model_name.lower()

        if re.match(r"^o4-(nano|mini|small)", model_name_lower):
            # Known open-access models like O4 family — approximating as openai/gpt-like
            return tiktoken.get_encoding("cl100k_base")

        elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
            return tiktoken.get_encoding("cl100k_base")  # Fair approx

        elif "llama" in model_name_lower:
            return tiktoken.get_encoding("p50k_base")  # Closer to LLaMA tokenization

        elif "falcon" in model_name_lower:
            return tiktoken.get_encoding("p50k_base")

        elif "bloom" in model_name_lower:
            return tiktoken.get_encoding("r50k_base")

        else:
            # 3. Fallback to a universal tokenizer
            logger.warning(
                f"[WARN] Model '{model_name}' not recognized. Falling back to cl100k_base tokenizer."
            )
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages, model="openai/gpt-4"):
        """
        Count the number of tokens in the given messages for pricing/limits.
        Belongs to: BaseLLMClient class.
        Args:
            messages (list): List of message dicts to count tokens for.
            model (str, optional): Model name for tokenizer. Defaults to "openai/gpt-4".
        Returns:
            int: Number of tokens in the messages.
        """
        logger.debug(f"Counting tokens for model={model}")
        tokenizer = self.get_tokenizer("o4-mini")
        total = 0
        for _msg in messages:
            total += len(tokenizer.encode("Your text goes here"))
        logger.debug(f"Token count: {total}")
        return total

    def get_price(self, model, prompt_tokens, completion_tokens):
        """
        Calculate the price of a request given token usage.
        Belongs to: BaseLLMClient class.
        Args:
            model (str): The model name.
            prompt_tokens (int): Number of prompt tokens used.
            completion_tokens (int): Number of completion tokens used.
        Returns:
            float: The calculated price for the request.
        """
        logger.debug(
            f"Calculating price for model={model} prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
        )
        if model in self.price:
            return (
                prompt_tokens / 1000 * float(self.price[model]["prompt_tokens"])
            ) + (
                completion_tokens / 1000 * float(self.price[model]["completion_tokens"])
            )
        else:
            logger.warning(
                f"Model {model} not found in price dict, using default pricing."
            )
            return (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.03)

    def get_key(self, env_var: str):
        """
        Get an API key from environment variable, raise error if missing.
        """
        logger.debug(f"Fetching key from environment variable: {env_var}")
        api_key = os.getenv(env_var)
        if api_key is None:
            logger.error(f"Missing required environment variable: {env_var}")
            raise OSError(f"Missing required environment variable: {env_var}")
        else:
            self.api_key = api_key
            logger.debug(f"API key set for {env_var}")


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str = None,
        file: str = None,
        temperature: int = 0.7,
        system_prompt: str = None,
        output: str = "",
        max_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            file=file,
            max_tokens=max_tokens,
        )
        self.api_key = api_key
        # self.system_prompt = system_prompt
        # logger.debug(f"system prompt {self.system_prompt=}")
        self.client = OpenAI(api_key=self.api_key)
        if file:
            self.file = file
        else:
            self.file = None
        if output:
            self.output = output
            logger.debug(f"No output on screen, {self.output=}")

        # Load pricing data from user config directory
        pricing_data = load_pricing_data()
        self.price = pricing_data.get("openai", {})

        if not self.price:
            logger.warning(
                "No OpenAI pricing data found in config. Using basic fallback pricing."
            )
            # Basic fallback pricing for essential models
            self.price = {
                "gpt-4o-mini": {"prompt_tokens": 0.00015, "completion_tokens": 7.5e-05},
                "gpt-4o": {"prompt_tokens": 0.0025, "completion_tokens": 0.00125},
                "gpt-4": {"prompt_tokens": 0.03, "completion_tokens": 0.06},
                "gpt-3.5-turbo": {"prompt_tokens": 0.0005, "completion_tokens": 0.0015},
            }

    def build_payload(self, prompt: str) -> dict:
        """
        Build the payload for the OpenAI API request.
        Belongs to: OpenAIClient class.
        Args:
            prompt (str): The user prompt or query.
        Returns:
            dict: The payload for the OpenAI API request.
        """
        messages = self.create_prompt(prompt)
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def get_endpoint(self) -> str:
        """
        Return the OpenAI chat completions API endpoint URL.
        Belongs to: OpenAIClient class.
        Returns:
            str: The OpenAI API endpoint URL.
        """
        return "https://api.openai.com/v1/chat/completions"

    def get_usage(self, days=10):
        """
        Retrieve and display usage data from OpenAI for the past N days.
        Belongs to: OpenAIClient class.
        Args:
            days (int, optional): Number of days to fetch usage for. Defaults to 10.
        """
        if (admin_key := os.getenv("OPENAI_ADMIN_KEY")) is None:
            raise OSError("Missing required environment variable: OPENAI_ADMIN_KEY")
        # client = OpenAIClient(
        #     # This is the default and can be omitted
        #     model="nothing",
        #     api_key=admin_key,
        # )

        headers = {"Authorization": f"Bearer {admin_key}"}
        # Refer: https://platform.openai.com/docs/api-reference/usage/costs
        url = "https://api.openai.com/v1/organization/costs"
        end = int(time.time())
        params = {
            "start_time": end - (days * 86400),
            "group_by": "line_item",
            "limit": 120,
        }
        logger.debug(f"{params=}")
        table = []
        a = 0
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            logger.debug(f"{data=}")
            for i in data["data"]:
                for r in i["results"]:
                    logger.debug(f"New Line: {r=}")
                    table.append(
                        [r["line_item"], f"{r['amount']['value']:.8f}", r["project_id"]]
                    )
                    logger.debug(
                        f"""Adding [r["line_item"], f'{r["amount"]["value"]:.10f}', r["project_id"]])"""
                    )
                    a = a + r["amount"]["value"]
                    # print (f"New Total: {a}")

                # print(f"💸 Total Usage: ${data['total_usage'] / 100:.4f} (USD)")
        else:
            print("❌ Failed to fetch usage:", response.text)
        # Sort by total cost
        table.sort(key=lambda x: x[1])

        # Print the table
        headers = ["Model", "Total Cost", "Porject ID"]
        # print(tabulate(table, headers=headers, tablefmt="github"))
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
        print(f"\nTotal Cost: {a:.8f}")

    def print_model_pricing_table(self, filter=None):
        """
        Print a table of model pricing for OpenAI models, optionally filtered.
        Belongs to: OpenAIClient class.
        Args:
            filter (str, optional): Filter string for model names. Defaults to None.
        """
        super().print_model_pricing_table(self.price, filter=filter)
        print("Note: This could be incorrect as this is data provided with script")

    def generate_image(self, prompt, size):
        """
        Generate an image using the OpenAI image generation API.
        Belongs to: OpenAIClient class.
        Args:
            prompt (str): The image generation prompt.
        """
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            n=1,
            size=size,  # other options: "512x512", "256x256" (DALL·E 2), or "1024x1024" (DALL·E 3)
        )
        image_url = response.data[0].url
        output.info(
            f"Image URL: {image_url}", f"[green]🇺🇷 Image URL:[/green] {image_url}"
        )
        response = requests.get(image_url)
        if response.status_code == 200:
            if self.output:
                with open(self.output, "wb") as f:
                    f.write(response.content)
            output.success(f"Image written as {self.output}")
        else:
            output.error(f"Failed to download. Status code: {response.status_code}")


class OpenRouterClient(BaseLLMClient):
    def get_endpoint(self) -> str:
        """
        Return the OpenRouter API endpoint URL.
        Belongs to: OpenRouterClient class.
        Returns:
            str: The OpenRouter API endpoint URL.
        """
        return "https://openrouter.ai/api/v1/"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str = None,
        file: str = None,
        temperature: int = 0.7,
        system_prompt: str = None,
        output: str = "",
        max_tokens: int = 4096,
        **kwargs,
    ):
        """
        Initialize the OpenRouterClient instance.
        Belongs to: OpenRouterClient class.
        Args:
            model (str, optional): The model to use. Defaults to "openai/gpt-4o-mini".
            api_key (str, optional): The API key to use. Defaults to None.
            file (str, optional): The file to use. Defaults to None.
            temperature (int, optional): The temperature to use. Defaults to 0.7.
            system_prompt (str, optional): The system prompt to use. Defaults to None.
            output (str, optional): The output to use. Defaults to "".
            max_tokens (int, optional): The maximum number of tokens to use. Defaults to 4096.
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            file=file,
            max_tokens=max_tokens,
        )
        self.api_key = api_key
        logger.debug("using auto routing with lowest cost model")
        # self.system_prompt = system_prompt
        # logger.debug(f"system prompt {self.system_prompt=}")
        self.client = OpenAI(api_key=self.api_key, base_url=self.get_endpoint())
        logger.debug(f"File is set to {self.file}")
        if output:
            self.output = output
            logger.debug(f"No output on screen, {self.output=}")

        self.price = {}
        models = self.list_available_models(batch=True)
        for m in models.data:
            # logger.debug(f"{m}")
            self.price[m.id] = {
                "prompt_tokens": m.pricing["prompt"],
                "completion_tokens": m.pricing["completion"],
            }

    def get_usage(self, days=10):
        """
        Retrieve and display usage data from OpenRouter for the past N days.
        Belongs to: OpenRouterClient class.
        Args:
            days (int, optional): Number of days to fetch usage for. Defaults to 10.
        """
        url = f"{self.get_endpoint()}credits"
        r = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        d = r.json()["data"]
        print(f"Total Credits: {d['total_credits']}")
        print(f"Total Usage: {d['total_usage']:.6f}")

    def print_model_pricing_table(self, filter=None):
        """
        Print a table of model pricing for OpenRouter models, optionally filtered.
        Belongs to: OpenRouterClient class.
        Args:
            filter (str, optional): Filter string for model names. Defaults to None.
        """
        super().print_model_pricing_table(self.price, filter=filter)

    def send_request(self, prompt: str) -> str:
        """
        Send a chat completion request to the OpenRouter API and return the response.
        Belongs to: OpenRouterClient class.
        Args:
            prompt (str): The user prompt or query.
        Returns:
            str: The response from the OpenRouter API.
        """
        self.query = prompt
        logger.debug(f"User Prompt is set to {prompt}")
        prompt = self.create_prompt(prompt)
        logger.debug(f"Created prompt is : {prompt=}")

        # prompt_tokens = count_tokens(messages, model)
        prompt_tokens = self.count_tokens(prompt, self.model)
        self.prompt_tokens = prompt_tokens
        # max_total = 128000
        # max_tokens = min(4096, max_total - prompt_tokens)  # absolute safe cap
        params = {
            "extra_headers": {
                "HTTP-Referer": "https://blog.amit-agarwal.co.in",  # Your app's URL
                "X-Title": "Zapgpt AI Assistant",  # Your app's display name
            },
            "model": self.model,
            "messages": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 1.0,
        }
        logger.debug(f"Making request with {params=}")
        response = self.client.chat.completions.create(**params)
        logger.debug(f"{response=}")
        return response


class TogetherClient(OpenAIClient):
    def get_endpoint(self) -> str:
        return "https://api.together.xyz/v1/chat/completions"


class DeepInfraClient(OpenAIClient):
    def get_endpoint(self) -> str:
        return "https://api.deepinfra.com/v1/openai/chat/completions"


class ReplicateClient(BaseLLMClient):
    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key

    def build_payload(self, prompt: str) -> dict:
        return {"input": {"prompt": prompt}, "model": self.model}

    def get_headers(self) -> dict:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_endpoint(self) -> str:
        return "https://api.replicate.com/v1/predictions"

    def handle_response(self, response_json: dict) -> str:
        output = response_json.get("output", "")
        # self.track_usage(self.model, "replicate", 0)
        return output


class GithubClient(BaseLLMClient):
    def get_endpoint(self) -> str:
        """
        Return the GitHub AI inference API endpoint URL.
        Belongs to: GithubClient class.
        Returns:
            str: The GitHub AI API endpoint URL.
        """
        return "https://models.github.ai/inference/chat/completions"

    def __init__(
        self,
        model: str = "openai/gpt-4.1",
        api_key: str = None,
        file: str = None,
        temperature: float = 0.7,
        system_prompt: str = None,
        output: str = "",
        max_tokens: int = 4096,
        **kwargs,
    ):
        """
        Initialize a GithubClient instance.
        Belongs to: GithubClient class.
        Args:
            model (str, optional): The model to use. Defaults to "openai/gpt-4.1".
            api_key (str, optional): The API key to use. Defaults to None.
            file (str, optional): The file to use. Defaults to None.
            temperature (float, optional): The temperature to use. Defaults to 0.7.
            system_prompt (str, optional): The system prompt to use. Defaults to None.
            output (str, optional): The output file to use. Defaults to "".
            max_tokens (int, optional): The maximum number of tokens to use. Defaults to 4096.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            file=file,
            max_tokens=max_tokens,
        )
        self.api_key = api_key
        logger.debug("using auto routing with lowest cost model")
        # self.system_prompt = system_prompt
        # logger.debug(f"system prompt {self.system_prompt=}")
        self.client = OpenAI(api_key=self.api_key, base_url=self.get_endpoint())
        logger.debug(f"File is set to {self.file}")
        if output:
            self.output = output
            logger.debug(f"No output on screen, {self.output=}")

        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.api_key,
            "GitHub-Api-Version": "2022-11-28",
        }
        self.price = {}
        models = self.list_available_models(batch=True)
        for m in models:
            logger.debug(f"{m}")
            model = m["id"]
            self.price[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

    def get_usage(self, days=10):
        """
        Retrieve and display usage data from GitHub AI for the past N days.
        Belongs to: GithubClient class.
        Args:
            days (int, optional): Number of days to fetch usage for. Defaults to 10.
        """
        url = f"{self.get_endpoint()}credits"
        r = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        d = r.json()["data"]
        print(f"Total Credits: {d['total_credits']}")
        print(f"Total Usage: {d['total_usage']:.6f}")

    def print_model_pricing_table(self, filter=None):
        """
        Print a table of model pricing for GitHub AI models, optionally filtered.
        Belongs to: GithubClient class.
        Args:
            filter (str, optional): Filter string for model names. Defaults to None.
        """
        super().print_model_pricing_table(self.price, filter=filter)

    def list_available_models(self, batch=False, filter=None):
        """
        List available models from GitHub AI, optionally filtering.
        Belongs to: GithubClient class.
        Args:
            batch (bool, optional): If True, return raw model data. Defaults to False.
            filter (str, optional): Filter string for model names. Defaults to None.
        Returns:
            list or None: List of models if batch is True, otherwise prints a table.
        """
        models = requests.get(
            "https://models.github.ai/catalog/models", headers=self.headers
        ).json()
        logger.debug(f"All Models: {models}")
        rich_table = Table(title="Model List")

        # table = []
        # headers = ["ID", "Created", "Description", "Context Len", "Modality", "Supported Parameters" ]
        if batch:
            return models
        rich_table.add_column("ID")
        rich_table.add_column("Name")
        rich_table.add_column("Publisher")
        rich_table.add_column("Rate Limit Tier")
        rich_table.add_column("Input Modalities")
        rich_table.add_column("Output Modalities")
        print("\n📦 Available Github Models:")
        for m in models:
            # Convert created to humban-readable formatting
            # print(f"* {m.id}, Owner: {m.owned_by}, Created: {m.created}")
            # table.append([ m.id, m.created, m.description, m.context_length, m.architecture["modality"], m.supported_parameters ])
            if filter:
                logger.debug(f"Filter is set to {filter=}")
                if filter not in m["id"] and filter not in m["name"]:
                    logger.debug(f"Fitlering out {m['id']}")
                    continue
            rich_table.add_row(
                m["id"],
                m["name"],
                m["publisher"],
                m["rate_limit_tier"],
                ",".join(m["supported_input_modalities"]),
                ",".join(m["supported_output_modalities"]),
            )
        # print(tabulate(clean_table, headers=headers, tablefmt="fancy_grid", maxcolwidths=[20, 20, 35, 10, 10, 35, 10] ))
        # Table format options: plain, simple, grid, fancy_grid, github, pipe, orgtbl, mediawiki, rst, html, latex, jira, pretty
        console.print(rich_table)

    def send_request(self, prompt: str) -> str:
        self.query = prompt
        logger.debug(f"User Prompt is set to {prompt}")
        prompt = self.create_prompt(prompt)
        logger.debug(f"Created prompt is : {prompt=}")

        # prompt_tokens = count_tokens(messages, model)
        prompt_tokens = self.count_tokens(prompt, self.model)
        self.prompt_tokens = prompt_tokens
        # max_total = 128000
        # max_tokens = min(4096, max_total - prompt_tokens)  # absolute safe cap
        params = {
            "model": self.model,
            "messages": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 1.0,
        }
        logger.debug(f"Making request with {params=} using requests")
        # response = self.client.chat.completions.create(**params)
        response = requests.post(self.get_endpoint(), headers=self.headers, json=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Response Code: {response.status_code}, {response.text=}")
            return None
        logger.debug(f"{response=}")

        return response

    def handle_response(self, response: str) -> str:
        reply = response["choices"][0]["message"]["content"]
        completion_tokens = self.count_tokens(reply, model=self.model)
        total_tokens = completion_tokens
        cost = 0.0

        if self.output:
            with open(self.output, "w") as f:
                f.writelines(reply)
        else:
            # In quiet mode, show only the raw response
            if output and output.quiet_mode:
                print(self.highlight_code(reply, lang="markdown"))
            else:
                # Normal mode: show headers and usage info
                print("\n--- RESPONSE ---\n")
                print(self.highlight_code(reply, lang="markdown"))
                print("\n--- USAGE ---")
                print(f"Prompt tokens: {self.prompt_tokens}")
                print(f"Completion tokens: {completion_tokens}")
                print(f"Total tokens: {total_tokens}")
                print(f"Estimated cost: ${cost:.5f}")

        self.record_usage(
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            query=self.query,
            provider="github",
        )


class LocalClient(BaseLLMClient):
    def get_endpoint(self, url="localhost") -> str:
        """
        Return the GitHub AI inference API endpoint URL.
        Belongs to: LocalClient class.
        Returns:
            str: The GitHub AI API endpoint URL.
        """
        logger.debug("Base URL set to : {url}")
        return f"http://{url}/v1"

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        api_key: str = None,
        file: str = None,
        temperature: float = 0.7,
        system_prompt: str = None,
        output: str = "",
        max_tokens: int = 4096,
        url: str = "localhost:11434",
        **kwargs,
    ):
        """
        Initialize a LocalClient instance.
        Belongs to: LocalClient class.
        Args:
            model (str, optional): The model to use. Defaults to "openai/gpt-4.1".
            api_key (str, optional): The API key to use. Defaults to None.
            file (str, optional): The file to use. Defaults to None.
            temperature (float, optional): The temperature to use. Defaults to 0.7.
            system_prompt (str, optional): The system prompt to use. Defaults to None.
            output (str, optional): The output file to use. Defaults to "".
            max_tokens (int, optional): The maximum number of tokens to use. Defaults to 4096.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            file=file,
            max_tokens=max_tokens,
        )
        self.api_key = api_key
        logger.debug("using auto routing with lowest cost model")
        # self.system_prompt = system_prompt
        # logger.debug(f"system prompt {self.system_prompt=}")
        if api_key:
            self.client = OpenAI(
                api_key=self.api_key, base_url=self.get_endpoint(url=url)
            )
        else:
            self.client = OpenAI(base_url=self.get_endpoint(url=url))
        logger.debug(f"File is set to {self.file}")
        if output:
            self.output = output
            logger.debug(f"No output on screen, {self.output=}")
        ## No cost solution
        self.price = {}
        models = self.list_available_models(batch=True)
        for m in models.data:
            # logger.debug(f"{m}")
            self.price[m.id] = {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
            }


def get_prompt(filename):
    """
    Utility to read a prompt file and return its contents.
    """
    logger.debug(f"Prompt file is {filename}")
    if os.path.exists(filename):
        with open(filename) as f:
            system_prompt = f.read()
            logger.debug(f"Prompt {system_prompt}")
        return system_prompt
    else:
        logger.warning(f"Prompt file {filename} does not exist.")
        return ""


provider_map = {
    "openai": OpenAIClient,
    "openrouter": OpenRouterClient,
    "together": TogetherClient,
    "replicate": ReplicateClient,
    "deepinfra": DeepInfraClient,
    "github": GithubClient,
    "local": LocalClient,
}

# Mapping of providers to their required environment variables
provider_env_vars = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_KEY",
    "together": "TOGETHER_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "deepinfra": "DEEPINFRA_API_TOKEN",
    "github": "GITHUB_KEY",
    "local": "OLLAMA_KEY",
}


def query_llm(
    prompt: str,
    provider: str = "openai",
    model: str = None,
    system_prompt: str = None,
    use_prompt: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    quiet: bool = True,
) -> str:
    """
    Programmatic API to query LLM providers from Python scripts.

    Args:
        prompt (str): The user prompt/question
        provider (str): LLM provider (openai, openrouter, together, etc.)
        model (str, optional): Specific model to use
        system_prompt (str, optional): Custom system prompt
        use_prompt (str, optional): Use a predefined prompt template
        temperature (float): Response randomness (0.0-1.0)
        max_tokens (int): Maximum response tokens
        quiet (bool): Suppress all output except response

    Returns:
        str: The LLM response text

    Raises:
        EnvironmentError: If required API key is missing
        ValueError: If provider is not supported

    Example:
        >>> from zapgpt import query_llm
        >>> response = query_llm("What is Python?", provider="openai")
        >>> print(response)
    """
    # Validate provider
    if provider not in provider_map:
        raise ValueError(
            f"Unsupported provider: {provider}. Available: {list(provider_map.keys())}"
        )

    # Check API key
    required_env_var = provider_env_vars.get(provider)
    api_key = None
    if required_env_var:
        api_key = os.getenv(required_env_var)
        if not api_key:
            raise OSError(f"Missing required environment variable: {required_env_var}")

    # Load prompts for use_prompt functionality
    prompt_jsons = {}
    try:
        for prompt_file in glob.glob(os.path.join(USER_PROMPTS_DIR, "*.json")):
            name = os.path.splitext(os.path.basename(prompt_file))[0]
            with open(prompt_file, encoding="utf-8") as f:
                prompt_jsons[name] = json.load(f)
    except Exception as e:
        if not quiet:
            logger.error(f"Failed to load prompts: {e}")

    # Load prompts if using predefined prompt
    final_system_prompt = system_prompt
    final_model = model

    if use_prompt:
        if use_prompt in prompt_jsons:
            prompt_data = prompt_jsons[use_prompt]
            final_system_prompt = prompt_data.get("system_prompt", system_prompt)
            if not model:  # Only use prompt's model if not explicitly provided
                final_model = prompt_data.get("model")

            # Add common_base if it exists
            if use_prompt != "common_base" and "common_base" in prompt_jsons:
                common_base_prompt = prompt_jsons["common_base"].get(
                    "system_prompt", ""
                )
                if common_base_prompt and final_system_prompt:
                    final_system_prompt = (
                        f"{common_base_prompt}\n\n{final_system_prompt}"
                    )
        else:
            raise ValueError(
                f"Prompt '{use_prompt}' not found. Available: {list(prompt_jsons.keys())}"
            )

    # Set default model if none specified
    if not final_model:
        final_model = "openai/gpt-4o-mini"  # Default model

    # Create client
    client_class = provider_map[provider]
    llm_client = client_class(
        model=final_model,
        api_key=api_key,
        system_prompt=final_system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Send request and return response
    try:
        response = llm_client.send_request(prompt)
        return response
    except Exception as e:
        if not quiet:
            logger.error(f"Error querying {provider}: {e}")
        raise


class MyHelpFormatter(RichHelpFormatter):
    def __init__(self, prog):
        RichHelpFormatter.__init__(prog, rich_terminal=False, width=100)


epilog = Markdown(
    dedent(
        """
    ### Example usage:
      * `gpt "What's the capital of France?"`
      * `gpt "Refactor this function" --model openai/gpt-4`
      * `gpt --history`
      * `gpt --total`
      * `gpt --list-models`
      * `gpt "Give me a plan for a YouTube channel" --use-prompt`
    """
    )
)


def main():
    """
    Main entrypoint for the CLI application.
    Parses arguments, dispatches commands, and manages logging.
    """
    # Fix Windows encoding issues with Unicode characters
    import sys

    if sys.platform == "win32":
        # Set environment variables for Windows encoding
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONUTF8"] = "1"

        # Try to reconfigure stdout/stderr to UTF-8 (Python 3.7+)
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # Fallback for older Python versions or if reconfigure fails
            pass

    # Ensure the configuration directory and files exist and are up to date
    ensure_config_directory()

    # Gather available prompt choices from the prompts directory
    # Dynamically load all JSON prompt files from the prompts directory
    prompt_jsons = {}
    for prompt_file in glob.glob(os.path.join(USER_PROMPTS_DIR, "*.json")):
        name = os.path.splitext(os.path.basename(prompt_file))[0]
        with open(prompt_file, encoding="utf-8") as f:
            try:
                prompt_jsons[name] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load prompt file {prompt_file}: {e}")

    prompt_choices = list(prompt_jsons.keys())

    # Setup command-line argument parsing
    parser = ArgumentParser(
        description="ZapGPT CLI: Ask LLM models, track usage and cost.",
        formatter_class=RichHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("query", nargs="?", type=str, help="Your prompt or question")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai/gpt-4.1",
        help="Specify the model to use (default: openai/gpt-4o-mini). Available models can be listed using --list-models.",
    )
    parser.add_argument(
        "-up",
        "--use-prompt",
        type=match_abbreviation(prompt_choices),
        default=None,
        help=f"Specify a prompt type. Options: {', '.join(prompt_choices)}. Default is 'general'.",
    )
    parser.add_argument(
        "-xai",
        "--explain-AI",
        action="store_true",
        help="Explian the thought process for reaching the response.",
    )
    parser.add_argument(
        "-lsp",
        "--list-prompt",
        action="store_true",
        help="List all the prompts",
    )
    parser.add_argument(
        "-hi",
        "--history",
        action="store_true",
        help="Display the history of past queries.",
    )
    parser.add_argument(
        "-t",
        "--total",
        action="store_true",
        help="Show the total cost of all interactions.",
    )
    parser.add_argument(
        "-lm",
        "--list-models",
        action="store_true",
        help="List all available OpenAI models.",
    )
    parser.add_argument(
        "-lp",
        "--list-pricing",
        action="store_true",
        help="Show pricing for 1k input and output tokens",
    )
    parser.add_argument("-fl", "--flex", action="store_true", help="Enable flex mode")
    parser.add_argument(
        "-te",
        "--temp",
        type=float,
        default=0.3,
        help="Set the temperature for responses (0-1.0). Default is 0.3.",
    )
    parser.add_argument(
        "-u",
        "--usage",
        required=False,
        nargs="?",
        const=10,
        type=int,
        help="Specify days to retrieve usage from OpenAI API.",
    )
    parser.add_argument("-f", "--file", default=None, help="Path to the file")
    parser.add_argument(
        "-ll",
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all console output except the LLM response (useful for scripting).",
    )
    parser.add_argument(
        "-c", "--chat-id", default=None, help="Continue in same chat session"
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=provider_map.keys(),
        default="github",
        help="Select LLM provider",
    )
    parser.add_argument(
        "-mt",
        "--max-tokens",
        type=int,
        default=4096,
        help="Use low cost LLM for OpenRouter",
    )
    parser.add_argument(
        "-fi",
        "--filter",
        type=str,
        help="Set the logging level.",
    )
    parser.add_argument(
        "-ip",
        "--image-prompt",
        type=str,
        help="Prompt to generate image.",
    )
    parser.add_argument(
        "-is",
        "--image-size",
        type=str,
        default="256x256",
        help="Image Size. Options - 256x256, 512x512, 1024x1024, 1024x1536, 1536x1024 and auto",
    )
    parser.add_argument("-o", "--output", default=None, help="The output file to use.")
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show configuration directory information.",
    )
    parser.add_argument(
        "--show-prompt",
        type=str,
        help="Show the complete prompt that would be sent to LLM for the given prompt name.",
    )
    parser.add_argument(
        "-url",
        "--url",
        type=str,
        default="localhost:11434",
        help="Set the URL for local provider if not localhost:11434",
    )

    args = parser.parse_args()

    # Handle quiet mode FIRST - suppress all output except LLM response
    if args.quiet:
        logger.setLevel(logging.CRITICAL)  # Only show critical errors

    # Initialize global output handler
    global output
    output = OutputHandler(quiet_mode=args.quiet)

    logger.info("Starting zapgpt CLI main entry point")

    # Handle prompt listing early and exit
    if args.list_prompt:
        # User-facing output: show all prompt names
        for i in prompt_choices:
            print(f"* {i}")
        logger.info("Displayed all available prompts.")
        return

    logger.info(
        f"Parsed arguments: model={args.model}, provider={args.provider}, max_tokens={args.max_tokens}"
    )

    model = args.model

    # Print zapgpt logo/banner (user-facing, not logged) - only if not quiet
    if not args.quiet:
        console.print(
            f"""
[bold yellow]
        ███████╗
       ████████╗
      █████████╗
     ██████████╗
    ███████████╗
   ████████████╗
  █████████████╗
 ██████████████╗
███████████████╗
 ╚█████████████╝
  ╚███████████╝
   ╚█████████╝
    ╚███████╝
     ╚█████╝
      ╚███╝
       ╚█╝
[/bold yellow]
[bold blue]╔══════════════════════════════════════════════════╗
║ ⚡ [bold yellow]Zap[/bold yellow][bold white]GPT[/bold white] [dim]v{VERSION}[/dim] 🚀✨ Multi-provider AI automation 🛡️ ║
╚══════════════════════════════════════════════════╝[/bold blue]
            """,
            justify="center",
        )

        logger.debug(f"Arguments: {args}")
        try:
            logger.setLevel(getattr(logging, args.log_level.upper()))
            logger.debug(f"Log level is set to {args.log_level.upper()}")
        except Exception as e:
            logger.error(f"Invalid log level: {e}")

    # Check if output file already exists
    if args.output:
        if os.path.exists(args.output):
            logger.critical(f"File already exists {args.output}")
            sys.exit(-2)
        else:
            logger.debug(f"Setting output file as {args.output=}")
    system_prompt = None
    # If --use-prompt is set, load system_prompt and model from the prompt's JSON
    if args.use_prompt:
        prompt_name = args.use_prompt
        if prompt_name in prompt_jsons:
            # Load the selected prompt
            prompt_data = prompt_jsons[prompt_name]
            system_prompt = prompt_data.get("system_prompt", "")

            # Always prepend common_base if it's not the base prompt itself
            if prompt_name != "common_base" and "common_base" in prompt_jsons:
                common_base_prompt = prompt_jsons["common_base"].get(
                    "system_prompt", ""
                )
                if common_base_prompt:
                    system_prompt = f"{common_base_prompt}\n\n{system_prompt}"
                    logger.info(f"Combined 'common_base' with '{prompt_name}' prompt")

            # User-provided model takes precedence over prompt's model
            if args.model != "openai/gpt-4.1":  # User explicitly provided a model
                model = args.model
                logger.info(
                    f"Using user-specified model '{model}' (overriding prompt default)"
                )
            else:
                model = prompt_data.get("model", args.model)
                logger.info(f"Using model from prompt '{prompt_name}': '{model}'")

            if args.explain_AI:
                system_prompt += "\n\nImportant! Always start with a json of steps and thoughts that you took to get to response, json should have step number and your thought process.\n"
        else:
            logger.error(f"Prompt '{prompt_name}' not found in prompts directory.")
            system_prompt = ""
            model = args.model
    else:
        model = args.model

    # Prepare assistant_input if present
    assistant_input = None
    if args.use_prompt and prompt_name in prompt_jsons:
        assistant_input = prompt_jsons[prompt_name].get("assistant_input", None)

    # Handle commands that don't require API keys first
    if args.config:
        from .config import show_config_info

        show_config_info()
        return

    if args.show_prompt:
        show_complete_prompt(args.show_prompt, args.model)
        return

    # Now handle commands that need API access
    client_class = provider_map[args.provider]

    # Get the correct API key for the selected provider
    required_env_var = provider_env_vars.get(args.provider)
    api_key = None
    if required_env_var:
        api_key = os.getenv(required_env_var)
        if not api_key and args.provider != "local":
            logger.error(f"Missing required environment variable: {required_env_var}")
            output.error(f"Missing API key for {args.provider}")
            output.warning(f"Please set the environment variable: {required_env_var}")
            sys.exit(1)

    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File name {args.file} does not exists, exiting")
            sys.exit(1)

    llm_client = client_class(
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        temperature=args.temp,
        file=args.file,
        output=args.output,
        max_tokens=args.max_tokens,
        url=args.url,
    )

    if args.image_prompt:
        if args.provider != "openai":
            output.error("Only OpenAI is supported for Image generation")
            return False
        llm_client.generate_image(args.image_prompt, args.image_size)
        return

    if args.list_models:
        if args.file:
            batch = True
        else:
            batch = False
        models = llm_client.list_available_models(batch, filter=args.filter)
        if args.file:
            with open(args.file, "w") as f:
                f.writelines(json.dumps(models.model_dump()))
        return

    if args.list_pricing:
        # if hasattr(OpenAIClient, "price"):
        llm_client.print_model_pricing_table(filter=args.filter)
        # else:
        # print("⚠️ No pricing data available.")
        return

    if args.usage:
        llm_client.get_usage(args.usage)
        return

    if args.history:
        BaseLLMClient.show_history()
        return

    if args.total:
        BaseLLMClient.show_total_cost()
        return

    if args.query:
        # Compose conversation with optional assistant_input
        prompt = args.query
        if assistant_input:
            # If the LLM client supports chat history, add assistant_input as the first assistant message
            if hasattr(llm_client, "chat_history") and isinstance(
                llm_client.chat_history, list
            ):
                llm_client.chat_history.clear()
                llm_client.chat_history.append(
                    {"role": "assistant", "content": assistant_input}
                )
            else:
                # For clients that use prompt construction, prepend assistant_input to the prompt
                prompt = f"{assistant_input}\n\n{args.query}"
        try:
            response = llm_client.send_request(prompt)
            if response:
                logger.debug(f"Response: {response=}")
                llm_client.handle_response(response)
            else:
                logger.error("Bad Response from LLM")
                output.error("Bad Response from LLM")
        except Exception as e:
            logger.exception(f"❌ Error while querying: {e}")
        return

    # If no other action, show help
    parser.print_help()


if __name__ == "__main__":
    main()
