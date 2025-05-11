#!/usr/bin/python3
######################################################################
#
#      FileName: gpt
#
#
#        Author: Amit Agarwal
#   Description:
#       Version: 1.0
#       Created: 20250506 22:17:21
#      Revision: none
#        Author: Amit Agarwal (aka), <amit.agarwal@mobileum.com>
#       Company:
# Last modified: 20250506 22:17:21
#
######################################################################

import os
import sys
import argparse
import requests
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
import logging
import time
from pygments import highlight
from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.formatters import TerminalFormatter
from tabulate import tabulate
from openai import OpenAI, responses
from datetime import datetime
import tiktoken
import re


# Setup logging
console = Console()
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)], format="%(message)s")
logger = logging.getLogger("llm")

current_script_path =  str(Path(__file__).resolve().parent)
DB_FILE = os.path.expanduser(current_script_path + "/gpt_usage.db")

def pretty(x):
    return f"{x:.10f}".rstrip("0").rstrip(".")

def color_cost(value):
    try:
        num = float(value)
        if num == 0:
            return f"[green]{num:.10f}[/green]"
        else:
            return f"[cyan]{num:.10f}[/cyan]"
    except ValueError:
        return f"[red]Invalid[/red]"

def fmt_colored(value):
    num = float(value)
    color = "green" if num == 0 else "cyan"
    return f"[{color}]{num:.10f}[/{color}]"


def get_filenames_without_extension(folder_path):
    # List to hold the filenames without the .txt extension
    filenames = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .txt extension
        if filename.endswith('.txt'):
            # Remove the .txt extension and add to the list
            filenames.append(filename[:-4])  # Remove the last 4 characters ('.txt')

    return filenames

# def match_abbreviation(arg, choices):
#     """Match partial input to full option with ambiguity handling."""
#     arg = arg.casefold()
#     matches = [choice for choice in choices if choice.casefold().startswith(arg)]
#     if not matches:
#         raise argparse.ArgumentTypeError(f"Invalid choice: '{arg}' (expected one of {choices})")
#     elif len(matches) > 1:
#         raise argparse.ArgumentTypeError(f"Ambiguous choice: '{arg}' (matches: {matches})")
#     return matches[0]

def match_abbreviation(options: dict | list[str]):
    """
    Returns a function for argparse `type=` that matches partial input to full option key.
    Accepts either a list of strings or dict keys.
    """

    # Convert to list if a dict is passed
    valid_keys = list(options.keys()) if isinstance(options, dict) else list(options)

    def _match(value: str) -> str:
        value = value.strip().lower()
        matches = [key for key in valid_keys if key.lower().startswith(value)]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise argparse.ArgumentTypeError(
                f"Ambiguous input '{value}' → matches: {', '.join(matches)}"
            )
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid input '{value}' → expected one of: {', '.join(valid_keys)}"
            )

    return _match

class BaseLLMClient:

    def __init__(self, model: str, system_prompt: str = "", max_tokens: int = 1000, temperature: float = 0.7, output: str = "", file: str = None):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chat_history = []
        self.current_script_path = str(current_script_path)
        self.prompts_path=self.current_script_path + "/prompts"
        self.output = output
        logger.debug("File is set to {file=}")
        if file:
            self.file = file
        else:
            self.file = None
        logger.debug(f"Prompts path is {self.prompts_path=}")
        self.init_db()
        #logger.debug(f"{self.system_prompt=}")

    def init_db(self):

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS usage (model TEXT, provider TEXT, tokens INTEGER)''')
        conn.commit()
        conn.close()

    def record_usage(self,model: str, provider: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost: int, query: int):
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
                provider+":"+model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
                query,
            ),
        )
        conn.commit()
        conn.close()

    def create_prompt(self, query: str):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})
        if self.file:
            logger.debug(f"File is set to {self.file=}")
            try:
                with open(self.file, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except Exception as e:
                logger.critical(f"❌ Failed to read file: {e}")
                return
            messages.append({"role": "user", "content": f"File content:\n{file_content}"})
        logger.debug(f"Created prompt is : {messages=}")
        return messages

    def build_payload(self, prompt: str) -> Dict:
        raise NotImplementedError

    def get_headers(self) -> Dict:
        raise NotImplementedError

    def get_endpoint(self) -> str:
        raise NotImplementedError

    def send_request(self, prompt: str) -> str:

        self.query = prompt
        logger.debug(f"User Prompt is set to {prompt}")
        prompt = self.create_prompt(prompt)
        logger.debug(f"Created prompt is : {prompt=}")

        # prompt_tokens = count_tokens(messages, model)
        prompt_tokens = self.count_tokens(prompt, self.model)
        self.prompt_tokens = prompt_tokens
        max_total = 128000
        max_tokens = min(4096, max_total - prompt_tokens)  # absolute safe cap
        params = {
            "model":self.model, "messages":prompt, "temperature": self.temperature,
            "max_tokens":max_tokens, "top_p":1.0
        }
        logger.debug(f"Making request with {params=}")
        response = self.client.chat.completions.create(**params)
        logger.debug(f"{response=}")
        return response

    def handle_response(self, response_json: Dict) -> str:
        raise NotImplementedError

    def add_to_history(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})


    def highlight_code(self,code: str, lang: str = "python") -> str:
        lexer = get_lexer_by_name(lang, stripall=True)
        formatter = TerminalFormatter()
        return highlight(code, lexer, formatter)


    def show_history():
        conn = sqlite3.connect(DB_FILE)
        logger.debug(f"Opened {DB_FILE=}")
        cursor = conn.cursor()
        for row in cursor.execute(
            "SELECT timestamp, model, total_tokens, cost, query FROM usage ORDER BY timestamp DESC"
        ):
            print(
                f"[{row[0]}] model={row[1]} tokens={row[2]} cost=${row[3]:.5f}\nquery: {row[4][:60]}...\n"
            )
        conn.close()

    def show_total_cost():
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(cost) FROM usage")
        total = cursor.fetchone()[0] or 0.0
        print(f"\n💸 Total estimated cost so far: ${total:.5f}")
        conn.close()


    def list_available_models(self, batch=False):
        models = self.client.models.list()
        if batch:
            return models
        print("\n📦 Available OpenAI Models:")
        for m in models.data:
            # Convert created to humban-readable formatting
            m.created = datetime.fromtimestamp(m.created).strftime("%Y-%m-%d %H:%M:%S")
            print(f"* {m.id}, Owner: {m.owned_by}, Created: {m.created}")


    # def print_model_pricing_table(self,pricing_data):
    #     # Build list with calculated total cost per 1000 prompt + 1000 output tokens
    #     table = []
    #     for model, prices in pricing_data.items():
    #         logger.debug(f"{prices=}")
    #         pc = prices.get("prompt_tokens", 0)
    #         prompt_cost = float(pc) if isinstance(pc, str) else pc
    #         oc = prices.get("completion_tokens", 0)
    #         output_cost = float(oc) if isinstance(oc, str) else oc
    #         total = (prompt_cost + output_cost)
    #         logger.debug(f"{prompt_cost} - {output_cost} - {total}")
    #         logger.debug(f"{pretty(prompt_cost)} - {pretty(output_cost)} - {pretty(total)}")
    #         table.append([model, pretty(prompt_cost), pretty(output_cost), pretty(total)])

    #     # Sort by total cost
    #     table.sort(key=lambda x: x[3])

    #     # Print the table
    #     headers = ["Model", "Prompt ($/1k)", "Output ($/1k)", "Total ($/1k in + out)"]
    #     # print(tabulate(table, headers=headers, tablefmt="github"))
    #    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    def print_model_pricing_table(self,pricing_data):
        # Build list with calculated total cost per 1000 prompt + 1000 output tokens
        rich_table = Table(title="Model Usage Costs")

        rich_table.add_column("Model")
        rich_table.add_column("Prompt Cost")
        rich_table.add_column("Output Cost")
        rich_table.add_column("Total")

        sorted_data = sorted(pricing_data.items(), key=lambda x: sum(float(v) for v in x[1].values()))   # or x[1][3] if total is at index 3
        for model, prices in sorted_data:
            logger.debug(f"{prices=}")
            pc = prices.get("prompt_tokens", 0)
            prompt_cost = float(pc) if isinstance(pc, str) else pc
            oc = prices.get("completion_tokens", 0)
            output_cost = float(oc) if isinstance(oc, str) else oc
            total = (prompt_cost + output_cost)
            logger.debug(f"{prompt_cost} - {output_cost} - {total}")
            logger.debug(f"{pretty(prompt_cost)} - {pretty(output_cost)} - {pretty(total)}")
            rich_table.add_row(model, fmt_colored(prompt_cost), fmt_colored(output_cost), fmt_colored(total))

        console.print(rich_table)

    def get_tokenizer(self, model_name: str):
        """
        Return an appropriate tokenizer encoding for the given model.

        If the model is not recognized by tiktoken, it uses heuristics to choose
        a reasonable fallback.
        """

        # 1. Try the default tiktoken logic
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            pass  # Go to heuristic matching

        # 2. Model-specific heuristics
        model_name_lower = model_name.lower()

        if re.match(r"^o4-(nano|mini|small)", model_name_lower):
            # Known open-access models like O4 family — approximating as GPT-like
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
            warnings.warn(
                f"[WARN] Model '{model_name}' not recognized. Falling back to cl100k_base tokenizer."
            )
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages, model="gpt-4"):
        # encoding = tiktoken.encoding_for_model(model)
        # total = 0
        # for msg in messages:
        #     total += 4
        #     for key, value in msg.items():
        #         total += len(encoding.encode(value))
        # total += 2
        # return total
        tokenizer = self.get_tokenizer("o4-mini")
        total = 0
        for msg in messages:
            total += len(tokenizer.encode("Your text goes here"))
        return total


    def get_price(self,model, prompt_tokens, completion_tokens):
        if model in self.price:
            return (prompt_tokens / 1000 * self.price[model]["prompt_tokens"]) + (
                completion_tokens / 1000 * self.price[model]["completion_tokens"]
            )
        else:
            return (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.03)

    def get_key(self, env_var: str):
        api_key = os.getenv(env_var)
        if api_key is None:
            raise EnvironmentError(f"Missing required environment variable: {env_var}")
        else:
            self.api_key = api_key


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str  = "gpt-4o-mini", api_key: str = None, file : str = None, temperature: int = 0.7, system_prompt : str = None, output: str = "", **kwargs):
        super().__init__(model=model, system_prompt = system_prompt, temperature = temperature, file=file)
        self.api_key = api_key
        # self.system_prompt = system_prompt
        #logger.debug(f"system prompt {self.system_prompt=}")
        self.client = OpenAI(
            api_key = self.api_key
        )
        if file:
            self.file = file
        else:
            self.file = None
        if output:
            self.output = output
            logger.debug(f"No output on screen, {self.output=}")
        self.price = {
            "gpt-4o-audio-preview-2024-12-17": {
                "prompt_tokens": 0.04,
                "completion_tokens": 0.08,
            },
            "dall-e-3": {"prompt_tokens": 0.04, "completion_tokens": 0.04},
            "dall-e-2": {"prompt_tokens": 0.02, "completion_tokens": 0.02},
            "gpt-4o-audio-preview-2024-10-01": {
                "prompt_tokens": 0.04,
                "completion_tokens": 0.08,
            },
            "gpt-4-turbo-preview": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "text-embedding-3-small": {"prompt_tokens": 0.02, "completion_tokens": 0.02},
            "gpt-4-turbo": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "gpt-4-turbo-2024-04-09": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "gpt-4.1-nano": {"prompt_tokens": 0.0001, "completion_tokens": 0.0004},
            "gpt-4.1-nano-2025-04-14": {"prompt_tokens": 0.0001, "completion_tokens": 0.0004},
            "gpt-4o-realtime-preview-2024-10-01": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "gpt-4o-realtime-preview": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "babbage-002": {"prompt_tokens": 0.0005, "completion_tokens": 0.0005},
            "gpt-4": {"prompt_tokens": 0.03, "completion_tokens": 0.06},
            "text-embedding-ada-002": {"prompt_tokens": 0.0001, "completion_tokens": 0.0001},
            "chatgpt-4o-latest": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "gpt-4o-realtime-preview-2024-12-17": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "gpt-4o-mini-audio-preview": {"prompt_tokens": 0.01, "completion_tokens": 0.02},
            "gpt-4o-audio-preview": {"prompt_tokens": 0.04, "completion_tokens": 0.08},
            "o1-preview-2024-09-12": {"prompt_tokens": 0.05, "completion_tokens": 0.10},
            "gpt-4o-mini-realtime-preview": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "gpt-4.1-mini": {"prompt_tokens": 0.0008, "completion_tokens": 0.0032},
            "gpt-4o-mini-realtime-preview-2024-12-17": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "gpt-3.5-turbo-instruct-0914": {
                "prompt_tokens": 0.0015,
                "completion_tokens": 0.002,
            },
            "gpt-4o-mini-search-preview": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "gpt-4.1-mini-2025-04-14": {"prompt_tokens": 0.0008, "completion_tokens": 0.0032},
            "davinci-002": {"prompt_tokens": 0.02, "completion_tokens": 0.02},
            "gpt-3.5-turbo-1106": {"prompt_tokens": 0.0015, "completion_tokens": 0.002},
            "gpt-4o-search-preview": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "gpt-3.5-turbo-instruct": {"prompt_tokens": 0.0015, "completion_tokens": 0.002},
            "gpt-3.5-turbo": {"prompt_tokens": 0.0015, "completion_tokens": 0.002},
            "gpt-4o-mini-search-preview-2025-03-11": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "gpt-4-0125-preview": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "gpt-4o-2024-11-20": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "whisper-1": {"prompt_tokens": 0.006, "completion_tokens": 0.006},
            "gpt-4o-2024-05-13": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "gpt-3.5-turbo-16k": {"prompt_tokens": 0.003, "completion_tokens": 0.004},
            "gpt-image-1": {"prompt_tokens": 0.04, "completion_tokens": 0.04},
            "o1-preview": {"prompt_tokens": 0.05, "completion_tokens": 0.10},
            "gpt-4-0613": {"prompt_tokens": 0.03, "completion_tokens": 0.06},
            "text-embedding-3-large": {"prompt_tokens": 0.13, "completion_tokens": 0.13},
            "gpt-4o-mini-tts": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "gpt-4o-transcribe": {"prompt_tokens": 0.006, "completion_tokens": 0.006},
            "gpt-4.5-preview": {"prompt_tokens": 0.075, "completion_tokens": 0.075},
            "gpt-4.5-preview-2025-02-27": {"prompt_tokens": 0.075, "completion_tokens": 0.075},
            "gpt-4o-mini-transcribe": {"prompt_tokens": 0.006, "completion_tokens": 0.006},
            "gpt-4o-search-preview-2025-03-11": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "omni-moderation-2024-09-26": {
                "prompt_tokens": 0.0005,
                "completion_tokens": 0.0005,
            },
            "tts-1-hd": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "gpt-4o": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "tts-1-hd-1106": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "gpt-4o-mini": {"prompt_tokens": 0.00015, "completion_tokens": 0.0006},
            "gpt-4o-2024-08-06": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "gpt-4.1": {"prompt_tokens": 0.003, "completion_tokens": 0.012},
            "gpt-4.1-2025-04-14": {"prompt_tokens": 0.003, "completion_tokens": 0.012},
            "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.00015, "completion_tokens": 0.0006},
            "o1-mini": {"prompt_tokens": 0.01, "completion_tokens": 0.02},
            "gpt-4o-mini-audio-preview-2024-12-17": {
                "prompt_tokens": 0.01,
                "completion_tokens": 0.02,
            },
            "gpt-3.5-turbo-0125": {"prompt_tokens": 0.0015, "completion_tokens": 0.002},
            "o1-mini-2024-09-12": {"prompt_tokens": 0.01, "completion_tokens": 0.02},
            "tts-1": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "gpt-4-1106-preview": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "tts-1-1106": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "omni-moderation-latest": {"prompt_tokens": 0.0005, "completion_tokens": 0.0005},
        }

    def build_payload(self, prompt: str) -> Dict:
        messages = self.create_prompt(prompt)
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

    def get_endpoint(self) -> str:
        return "https://api.openai.com/v1/chat/completions"

    def handle_response(self, response: str) -> str:
        logger.debug(f"{response.model_dump()}")
        logger.debug(f"Chat ID: {response.id}")
        reply = response.choices[0].message.content
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        cost = self.get_price(self.model, self.prompt_tokens, completion_tokens)

        if self.output:
            with open(self.output, "w") as f:
                f.writelines(reply)
        else:
            print(f"\n--- RESPONSE ---\n")
            print(self.highlight_code(reply, lang="markdown"))
            print("\n--- USAGE ---")
            print(f"Chat ID: {response.id}")
            print(f"Prompt tokens: {self.prompt_tokens}")
            print(f"Completion tokens: {completion_tokens}")
            print(f"Total tokens: {total_tokens}")
            print(f"Estimated cost: ${cost:.5f}")

        self.record_usage(model=self.model, prompt_tokens=self.prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens, cost=cost, query=self.query, provider="openai")

    def get_usage(self, days=10):
        """
        Use OpenAI to get usage
        """
        if (admin_key := os.getenv("OPENAI_ADMIN_KEY")) is None:
            raise EnvironmentError(
                "Missing required environment variable: OPENAI_ADMIN_KEY"
            )
        client = OpenAIClient(
            # This is the default and can be omitted
            model = "nothing",
            api_key=admin_key
        )

        headers = {"Authorization": f"Bearer {admin_key}"}
        # Refer: https://platform.openai.com/docs/api-reference/usage/costs
        url = "https://api.openai.com/v1/organization/costs"
        end = int(time.time())
        params = {"start_time": end -( days * 86400), "group_by": "line_item", "limit":120}
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
                    table.append([r["line_item"], f'{r["amount"]["value"]:.8f}', r["project_id"]])
                    logger.debug(f'''Adding [r["line_item"], f'{r["amount"]["value"]:.10f}', r["project_id"]])''')
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


    def print_model_pricing_table(self):
        super().print_model_pricing_table(self.price)


class OpenRouterClient(BaseLLMClient):

    def get_endpoint(self) -> str:
        return "https://openrouter.ai/api/v1/"

    def __init__(self, model: str  = "gpt-4o-mini", api_key: str = None, file : str = None, temperature: int = 0.7, system_prompt : str = None, output: str = "",  **kwargs):
        super().__init__(model=model, system_prompt = system_prompt, temperature = temperature, file=file)
        self.get_key("OPENROUTER_KEY")
        # self.system_prompt = system_prompt
        #logger.debug(f"system prompt {self.system_prompt=}")
        self.client = OpenAI(
            api_key = self.api_key,
            base_url=self.get_endpoint()
        )
        logger.debug(f"File is set to {self.file}")
        if output:
            self.output = output
            logger.debug(f"No output on screen, {self.output=}")

        self.price = {}
        models = self.list_available_models(batch=True)
        for m in models.data:
            logger.debug(f"{m.pricing}")
            self.price[m.id] = { "prompt_tokens":m.pricing["prompt"], "completion_tokens": m.pricing["completion"]}



    def get_usage(self,days=10):
        url = f"{self.get_endpoint()}credits"
        r = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        d = r.json()["data"]
        print (f"Total Credits: {d['total_credits']}")
        print (f"Total Usage: {d['total_usage']:.6f}")

    def print_model_pricing_table(self):
        super().print_model_pricing_table(self.price)
    # def print_model_pricing_table(self,pricing_data= None):
    #     # Build list with calculated total cost per 1000 prompt + 1000 output tokens
    #     url = f"{self.get_endpoint()}pricing"
    #     r = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"})
    #     print (r.text)
    #     print (r.json())

    #     table = []
    #     for model, prices in pricing_data.items():
    #         prompt_cost = prices.get("prompt_tokens", 0)
    #         output_cost = prices.get("completion_tokens", 0)
    #         total = prompt_cost + output_cost
    #         table.append([model, prompt_cost, output_cost, total])

    #     # Sort by total cost
    #     table.sort(key=lambda x: x[3])

    #     # Print the table
    #     headers = ["Model", "Prompt ($/1k)", "Output ($/1k)", "Total ($/1k in + out)"]
    #     # print(tabulate(table, headers=headers, tablefmt="github"))
    #     print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

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

    def build_payload(self, prompt: str) -> Dict:
        return {
            "input": {"prompt": prompt},
            "model": self.model
        }

    def get_headers(self) -> Dict:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_endpoint(self) -> str:
        return f"https://api.replicate.com/v1/predictions"

    def handle_response(self, response_json: Dict) -> str:
        output = response_json.get("output", "")
        track_usage(self.model, "replicate", 0)
        return output


provider_map = {
    "openai": OpenAIClient,
    "openrouter": OpenRouterClient,
    "together": TogetherClient,
    "replicate": ReplicateClient,
    "deepinfra": DeepInfraClient
}

def main():
    #print(f"{current_script_path=}")
    prompt_choices = get_filenames_without_extension(str(current_script_path)+ "/prompts")
    parser = argparse.ArgumentParser(
        description="💬 GPT CLI Tracker: Ask OpenAI models, track usage and cost.",
        epilog="""
          gpt "What's the capital of France?"
          gpt "Refactor this function" --model gpt-4
          gpt --history
          gpt --total
          gpt --list-models
          gpt "Give me a plan for a YouTube channel" --use-prompt
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("query", nargs="?", type=str, help="Your prompt or question")
    parser.add_argument(
        "-m", "--model", type=str, default="gpt-4o-mini",
        help="Specify the model to use (default: gpt-4o-mini). Available models can be listed using --list-models."
    )
    parser.add_argument(
        "-up", "--use-prompt", type=match_abbreviation(prompt_choices),
        default="default", help=f"Specify a prompt type. Options: {','.join(prompt_choices)}. Default is 'general'."
    )
    parser.add_argument("-hi", "--history", action="store_true", help="Display the history of past queries.")
    parser.add_argument("-t", "--total", action="store_true", help="Show the total cost of all interactions.")
    parser.add_argument("-lm", "--list-models", action="store_true", help="List all available OpenAI models.")
    parser.add_argument("-lp", "--list-pricing", action="store_true", help="Show pricing for 1k input and output tokens")
    parser.add_argument("-fl", "--flex", action="store_true", help="Enable flex mode")
    parser.add_argument(
        "-te", "--temp", type=float, default=0.3,
        help="Set the temperature for responses (0-1.0). Default is 0.3."
    )
    parser.add_argument("-u", "--usage", type=int, help="Specify days to retrieve usage from OpenAI API.")
    parser.add_argument("-f", "--file", default=None, help="Path to the file")
    parser.add_argument(
        "-ll", "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    parser.add_argument("-c", "--chat-id", default=None, help="Continue in same chat session")
    parser.add_argument(
        "-p", "--provider", choices=provider_map.keys(), default="openrouter",
        help="Select LLM provider"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="The output file to use."
    )

    args = parser.parse_args()
    logger.debug(f"Arguments {args=}")
    try:
        logger.setLevel(getattr(logging, args.log_level.upper()))
    except Exception as e:
        logger.error(f"Invalid log level: {e}")
    #print(f"Log Level: {args=}")


    if args.output:
        if os.path.exists(args.output):
            logger.critical(f"File already exists {args.output}")
            sys.exit(-2)
        else:
            logger.debug(f"Setting output file as {args.output=}")
    system_prompt=None
    if args.use_prompt:
        filename=f"{current_script_path}/prompts/{args.use_prompt}.txt"
        logger.debug(f"Prompt file is {filename}")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                system_prompt = "\n" .join(f.readlines())
                logger.debug(f"System prompt is set as {system_prompt}")
        else:
            logger.critical(f"Prompt File {filename} does not exist")
            sys.exit(-1)
    client_class = provider_map[args.provider]
    llm_client = client_class(
        model=args.model,
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt=system_prompt,
        temperature=args.temp,
        file = args.file,
        output = args.output
    )


    if args.list_models:
        llm_client.list_available_models()
        return

    if args.list_pricing:
        #if hasattr(OpenAIClient, "price"):
        llm_client.print_model_pricing_table()
        #else:
            #print("⚠️ No pricing data available.")
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
        #print("🧠 Querying model...\n")
        try:
            response = llm_client.send_request(args.query)
            llm_client.handle_response(response)
        except Exception as e:
            logger.error(f"❌ Error while querying: {e}")
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
