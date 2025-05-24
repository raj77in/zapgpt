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
from openai.resources import BatchesWithRawResponse
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
from rich_argparse import RichHelpFormatter

# Setup logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True)],
    format="%(message)s",
)
logger = logging.getLogger("llm")

current_script_path = str(Path(__file__).resolve().parent)
DB_FILE = os.path.expanduser(current_script_path + "/gpt_usage.db")

model_prompts = {
    "red_team": {
        "model": "openai/gpt-4-turbo",
        "system_prompt": "You are a penetration tester in a red team. Use your knowledge of exploits and attack vectors to craft effective PoC exploits for services.",
        "assistant_input": "Sure, here's a Python PoC using `socket` to exploit the buffer overflow in the vulnerable service..."
    },
    "reverse_eng": {
        "model": "anthropic/claude-3-opus",
        "system_prompt": "Analyze the binary code and provide insights into potential vulnerabilities or weaknesses.",
        "assistant_input": "After disassembling the binary, I noticed a classic stack buffer overflow in the `auth()` function due to unchecked `strcpy` usage..."
    },
    "payload_creation": {
        "model": "mistralai/mixtral-8x7b-instruct",
        "system_prompt": "Create a payload for penetration testing that targets a specific system vulnerability.",
        "assistant_input": "Here’s a reverse shell payload crafted with `msfvenom` targeting Windows x64 over HTTPS..."
    },
    "cyber_awareness": {
        "model": "anthropic/claude-3-sonnet",
        "system_prompt": "Write a professional post on defending against ransomware attacks in corporate environments.",
        "assistant_input": "Ransomware defense starts with strong endpoint detection, frequent offline backups, and phishing-resistant email filtering."
    },
    "phishing_campaign": {
        "model": "openai/gpt-3.5-turbo",
        "system_prompt": "Assist in setting up a phishing campaign, including email crafting, social engineering tactics, and payload deployment.",
        "assistant_input": "Crafted a spoofed Outlook login page and an email that leverages urgency to trick the user into clicking..."
    },
    "vuln_assessment": {
        "model": "openai/gpt-4",
        "system_prompt": "Create a comprehensive vulnerability assessment report based on the findings from a recent penetration test.",
        "assistant_input": "The assessment identified three high-severity issues: unauthenticated RCE, outdated Apache server, and misconfigured S3 buckets..."
    },
    "exploit_dev": {
        "model": "anthropic/claude-3-opus",
        "system_prompt": "Provide guidance on developing an exploit for a discovered vulnerability, including code snippets and attack vectors.",
        "assistant_input": "You can exploit this heap overflow by corrupting adjacent chunks and redirecting execution flow to a ROP chain..."
    },
    "soc_reporting": {
        "model": "openai/gpt-3.5-turbo",
        "system_prompt": "Write a detailed report for a Security Operations Center (SOC) regarding the latest security incident.",
        "assistant_input": "At 03:21 UTC, our EDR detected anomalous PowerShell execution tied to lateral movement from host `SRV-DC-02`..."
    },
    "red_team_planning": {
        "model": "mistralai/mixtral-8x7b-instruct",
        "system_prompt": "Assist in planning a red team engagement, including attack vectors, reconnaissance, and post-exploitation strategies.",
        "assistant_input": "Initial recon will include passive OSINT on the target org’s domain. The engagement will move into phishing, privilege escalation, and persistence."
    },
    "incident_response": {
        "model": "openai/gpt-4-turbo",
        "system_prompt": "Create an incident response playbook for handling a security breach, including steps for containment, eradication, and recovery.",
        "assistant_input": "Step 1: Isolate impacted endpoints. Step 2: Identify and terminate malicious processes. Step 3: Deploy remediation scripts and begin forensic analysis..."
    },
    "malware_analysis": {
        "model": "anthropic/claude-3-opus",
        "system_prompt": "Analyze a given piece of malware and describe its behavior, including potential impacts and mitigation strategies.",
        "assistant_input": "This malware runs as a background process, periodically checks for a C2 server, and can exfiltrate browser credentials via HTTP POST..."
    },
    "threat_intel": {
        "model": "openai/gpt-4",
        "system_prompt": "Write a detailed threat intelligence report on the latest trends in cyber threats, including recommendations for defense.",
        "assistant_input": "Q1 2025 shows a surge in AI-assisted phishing and ransomware-as-a-service offerings. Zero-trust and MFA adoption are critical responses..."
    },
    "social_attack_sim": {
        "model": "anthropic/claude-3-sonnet",
        "system_prompt": "Simulate an attack on social media platforms, including crafting fake posts and exploiting user vulnerabilities.",
        "assistant_input": "Posted a fake giveaway with a link to a credential harvester mimicking a known influencer's site. Engagement rate is high within the first hour..."
    },
    "system_hardening": {
        "model": "openai/gpt-4-turbo",
        "system_prompt": "Provide guidance on hardening a system, including steps for securing the operating system, applications, and network.",
        "assistant_input": "Disable unused services, enable SELinux/AppArmor, enforce password policies, and segment your network using VLANs..."
    },
    "security_policy": {
        "model": "mistralai/mixtral-8x7b-instruct",
        "system_prompt": "Assist in creating a security policy for a company, covering areas such as access control, data protection, and incident response.",
        "assistant_input": "Policy recommends RBAC with MFA for all privileged access, encrypted data at rest and transit, and quarterly incident response drills..."
    },
    "blog_cyber_trends": {
        "model": "openai/gpt-4",
        "system_prompt": "Write a blog post about the latest trends in cybersecurity, focusing on emerging threats and mitigation techniques.",
        "assistant_input": "Cybersecurity in 2025 is seeing an AI-driven arms race. Defenders must embrace threat intelligence automation to stay ahead..."
    },
    "social_media_post": {
        "model": "openai/gpt-3.5-turbo",
        "system_prompt": "Write a short, engaging social media post on phishing prevention for a general audience.",
        "assistant_input": "⚠️ Don’t get hooked! Always double-check sender emails and links. Hover before you click. #PhishingAwareness #CyberSafe"
    },
    "code_review": {
        "model": "anthropic/claude-3-opus",
        "system_prompt": "Review the provided code snippet from a security perspective and identify potential vulnerabilities.",
        "assistant_input": "Line 24 uses `eval()` on untrusted input, which could lead to code injection. Consider using `ast.literal_eval` or a safer parsing method..."
    },
    "security_training": {
        "model": "openai/gpt-4-turbo",
        "system_prompt": "Create a training module for employees on how to recognize phishing attempts and other common cyber threats.",
        "assistant_input": "Module 1: Spotting phishing signs — mismatched URLs, urgent tone, strange attachments. Interactive quiz follows each section."
    },
    "security_scripts": {
        "model": "mistralai/mixtral-8x7b-instruct",
        "system_prompt": "Write a Python script to automate security tasks, such as vulnerability scanning or incident response actions.",
        "assistant_input": "Here's a Python script using `nmap` and `subprocess` to scan the internal network and log open ports into a CSV file..."
    },
    "pentest_summary": {
        "model": "openai/gpt-3.5-turbo",
        "system_prompt": "Summarize the findings of a recent penetration test, highlighting the most critical vulnerabilities and recommendations.",
        "assistant_input": "3 critical vulns were found: Unrestricted file upload, hardcoded credentials, and exposed management interfaces..."
    },
    "sql_injection": {
        "model": "openai/gpt-4",
        "system_prompt": "Create a guide for exploiting SQL injection vulnerabilities, including techniques for bypassing filters and extracting data.",
        "assistant_input": "Start with basic `OR '1'='1` injection. Use `UNION SELECT` to extract DB names. Bypass WAFs using inline comments or encodings..."
    }
}

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
        if filename.endswith(".txt"):
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

    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        output: str = "",
        file: str = None,
    ):
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

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS usage (model TEXT, provider TEXT, tokens INTEGER)"""
        )
        conn.commit()
        conn.close()

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

    def create_prompt(self, query: str):
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
                with open(self.file, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except Exception as e:
                logger.critical(f"❌ Failed to read file: {e}")
                return
            messages.append(
                {"role": "user", "content": f"File content:\n{file_content}"}
            )
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

    # def handle_response(self, response_json: Dict) -> str:
    #     raise NotImplementedError

    def handle_response(self, response: str) -> str:
        logger.debug(f"{response.model_dump()}")
        logger.debug(f"Chat ID: {response.id}")
        reply = response.choices[0].message.content
        completion_tokens = float(response.usage.completion_tokens)
        total_tokens = float(response.usage.total_tokens)
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
        self.chat_history.append({"role": role, "content": content})

    def highlight_code(self, code: str, lang: str = "python") -> str:
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

    def list_available_models(self, batch=False, filter= None):
        rich_table = Table(title="Model List")


        table=[]
        #headers = ["ID", "Created", "Description", "Context Len", "Modality", "Supported Parameters" ]
        models = self.client.models.list()
        if batch:
            return models
        rich_table.add_column("ID")
        rich_table.add_column("Created")
        if hasattr(models.data[0], 'context_length'):
            rich_table.add_column("Ctx Len")
            rich_table.add_column("Modality")
        print("\n📦 Available OpenAI Models:")
        for m in models.data:
            # Convert created to humban-readable formatting
            m.created = datetime.fromtimestamp(m.created).strftime("%Y-%m-%d %H:%M:%S")
            # print(f"* {m.id}, Owner: {m.owned_by}, Created: {m.created}")
            #table.append([ m.id, m.created, m.description, m.context_length, m.architecture["modality"], m.supported_parameters ])
            if filter:
                logger.debug(f"Filter is set to {filter=}")
                if filter not in m.id and filter not in m.name:
                    logger.debug(f"Fitlering out {m.id}")
                    continue
            if hasattr(m, 'context_length'):
                cl = f"{int(m.context_length / 1000)} K" if m.context_length > 1000 else str(m.context_length)
                logger.debug(f"{m.context_length=} and {cl=}")
                rich_table.add_row( m.id, m.created, cl , m.architecture["modality"])
            else:
                rich_table.add_row( m.id, m.created)
        #print(tabulate(clean_table, headers=headers, tablefmt="fancy_grid", maxcolwidths=[20, 20, 35, 10, 10, 35, 10] ))
        # Table format options: plain, simple, grid, fancy_grid, github, pipe, orgtbl, mediawiki, rst, html, latex, jira, pretty
        console.print(rich_table)

    def print_model_pricing_table(self, pricing_data, filter= None):
        # Build list with calculated total cost per 1000 prompt + 1000 output tokens
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
                    logger.debug("Exlcluding {model} due to filter")
                    continue
            logger.debug(f"{prices=}")
            pc = prices.get("prompt_tokens", 0)
            prompt_cost = float(pc) *1000 if isinstance(pc, str) else pc
            oc = prices.get("completion_tokens", 0)
            output_cost = float(oc) *1000 if isinstance(oc, str) else oc
            total = (prompt_cost + output_cost) /2
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
            warnings.warn(
                f"[WARN] Model '{model_name}' not recognized. Falling back to cl100k_base tokenizer."
            )
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages, model="openai/gpt-4"):
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

    def get_price(self, model, prompt_tokens, completion_tokens):
        if model in self.price:
            return (
                prompt_tokens / 1000 * float(self.price[model]["prompt_tokens"])
            ) + (
                completion_tokens / 1000 * float(self.price[model]["completion_tokens"])
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
            model=model, system_prompt=system_prompt, temperature=temperature, file=file, max_tokens=max_tokens
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

        self.price = {
            "openai/gpt-4o-audio-preview-2024-12-17": {
                "prompt_tokens": 0.04,
                "completion_tokens": 0.08,
            },
            "dall-e-3": {"prompt_tokens": 0.04, "completion_tokens": 0.04},
            "dall-e-2": {"prompt_tokens": 0.02, "completion_tokens": 0.02},
            "openai/gpt-4o-audio-preview-2024-10-01": {
                "prompt_tokens": 0.04,
                "completion_tokens": 0.08,
            },
            "openai/gpt-4-turbo-preview": {
                "prompt_tokens": 0.01,
                "completion_tokens": 0.03,
            },
            "text-embedding-3-small": {
                "prompt_tokens": 0.02,
                "completion_tokens": 0.02,
            },
            "openai/gpt-4-turbo": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "openai/gpt-4-turbo-2024-04-09": {
                "prompt_tokens": 0.01,
                "completion_tokens": 0.03,
            },
            "openai/gpt-4.1-nano": {
                "prompt_tokens": 0.0001,
                "completion_tokens": 0.0004,
            },
            "openai/gpt-4.1-nano-2025-04-14": {
                "prompt_tokens": 0.0001,
                "completion_tokens": 0.0004,
            },
            "openai/gpt-4o-realtime-preview-2024-10-01": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-4o-realtime-preview": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "babbage-002": {"prompt_tokens": 0.0005, "completion_tokens": 0.0005},
            "openai/gpt-4": {"prompt_tokens": 0.03, "completion_tokens": 0.06},
            "text-embedding-ada-002": {
                "prompt_tokens": 0.0001,
                "completion_tokens": 0.0001,
            },
            "chatopenai/gpt-4o-latest": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-4o-realtime-preview-2024-12-17": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-4o-mini-audio-preview": {
                "prompt_tokens": 0.01,
                "completion_tokens": 0.02,
            },
            "openai/gpt-4o-audio-preview": {
                "prompt_tokens": 0.04,
                "completion_tokens": 0.08,
            },
            "o1-preview-2024-09-12": {"prompt_tokens": 0.05, "completion_tokens": 0.10},
            "openai/gpt-4o-mini-realtime-preview": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-4.1-mini": {
                "prompt_tokens": 0.0008,
                "completion_tokens": 0.0032,
            },
            "openai/gpt-4o-mini-realtime-preview-2024-12-17": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-3.5-turbo-instruct-0914": {
                "prompt_tokens": 0.0015,
                "completion_tokens": 0.002,
            },
            "openai/gpt-4o-mini-search-preview": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-4.1-mini-2025-04-14": {
                "prompt_tokens": 0.0008,
                "completion_tokens": 0.0032,
            },
            "davinci-002": {"prompt_tokens": 0.02, "completion_tokens": 0.02},
            "openai/gpt-3.5-turbo-1106": {
                "prompt_tokens": 0.0015,
                "completion_tokens": 0.002,
            },
            "openai/gpt-4o-search-preview": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-3.5-turbo-instruct": {
                "prompt_tokens": 0.0015,
                "completion_tokens": 0.002,
            },
            "openai/gpt-3.5-turbo": {
                "prompt_tokens": 0.0015,
                "completion_tokens": 0.002,
            },
            "openai/gpt-4o-mini-search-preview-2025-03-11": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-4-0125-preview": {
                "prompt_tokens": 0.01,
                "completion_tokens": 0.03,
            },
            "openai/gpt-4o-2024-11-20": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "whisper-1": {"prompt_tokens": 0.006, "completion_tokens": 0.006},
            "openai/gpt-4o-2024-05-13": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-3.5-turbo-16k": {
                "prompt_tokens": 0.003,
                "completion_tokens": 0.004,
            },
            "openai/gpt-image-1": {"prompt_tokens": 0.04, "completion_tokens": 0.04},
            "o1-preview": {"prompt_tokens": 0.05, "completion_tokens": 0.10},
            "openai/gpt-4-0613": {"prompt_tokens": 0.03, "completion_tokens": 0.06},
            "text-embedding-3-large": {
                "prompt_tokens": 0.13,
                "completion_tokens": 0.13,
            },
            "openai/gpt-4o-mini-tts": {
                "prompt_tokens": 0.0006,
                "completion_tokens": 0.012,
            },
            "openai/gpt-4o-transcribe": {
                "prompt_tokens": 0.006,
                "completion_tokens": 0.006,
            },
            "openai/gpt-4.5-preview": {
                "prompt_tokens": 0.075,
                "completion_tokens": 0.075,
            },
            "openai/gpt-4.5-preview-2025-02-27": {
                "prompt_tokens": 0.075,
                "completion_tokens": 0.075,
            },
            "openai/gpt-4o-mini-transcribe": {
                "prompt_tokens": 0.006,
                "completion_tokens": 0.006,
            },
            "openai/gpt-4o-search-preview-2025-03-11": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "omni-moderation-2024-09-26": {
                "prompt_tokens": 0.0005,
                "completion_tokens": 0.0005,
            },
            "tts-1-hd": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "openai/gpt-4o": {"prompt_tokens": 0.0025, "completion_tokens": 0.01},
            "tts-1-hd-1106": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "openai/gpt-4o-mini": {
                "prompt_tokens": 0.00015,
                "completion_tokens": 0.0006,
            },
            "openai/gpt-4o-2024-08-06": {
                "prompt_tokens": 0.0025,
                "completion_tokens": 0.01,
            },
            "openai/gpt-4.1": {"prompt_tokens": 0.003, "completion_tokens": 0.012},
            "openai/gpt-4.1-2025-04-14": {
                "prompt_tokens": 0.003,
                "completion_tokens": 0.012,
            },
            "openai/gpt-4o-mini-2024-07-18": {
                "prompt_tokens": 0.00015,
                "completion_tokens": 0.0006,
            },
            "o1-mini": {"prompt_tokens": 0.01, "completion_tokens": 0.02},
            "openai/gpt-4o-mini-audio-preview-2024-12-17": {
                "prompt_tokens": 0.01,
                "completion_tokens": 0.02,
            },
            "openai/gpt-3.5-turbo-0125": {
                "prompt_tokens": 0.0015,
                "completion_tokens": 0.002,
            },
            "o1-mini-2024-09-12": {"prompt_tokens": 0.01, "completion_tokens": 0.02},
            "tts-1": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "openai/gpt-4-1106-preview": {
                "prompt_tokens": 0.01,
                "completion_tokens": 0.03,
            },
            "tts-1-1106": {"prompt_tokens": 0.0006, "completion_tokens": 0.012},
            "omni-moderation-latest": {
                "prompt_tokens": 0.0005,
                "completion_tokens": 0.0005,
            },
        }

    def build_payload(self, prompt: str) -> Dict:
        messages = self.create_prompt(prompt)
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def get_endpoint(self) -> str:
        return "https://api.openai.com/v1/chat/completions"

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
            model="nothing",
            api_key=admin_key,
        )

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
                        [r["line_item"], f'{r["amount"]["value"]:.8f}', r["project_id"]]
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

    def print_model_pricing_table(self, filter= None):
        super().print_model_pricing_table(self.price, filter= filter)
        print("Note: This could be incorrect as this is data provided with script")


class OpenRouterClient(BaseLLMClient):

    def get_endpoint(self) -> str:
        return "https://openrouter.ai/api/v1/"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str = None,
        file: str = None,
        temperature: int = 0.7,
        system_prompt: str = None,
        output: str = "",
        max_tokens : int = 4096,
        **kwargs,
    ):
        super().__init__(
            model=model, system_prompt=system_prompt, temperature=temperature, file=file, max_tokens=max_tokens
        )
        self.get_key("OPENROUTER_KEY")
        logger.debug ("using auto routing with lowest cost model")
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
            #logger.debug(f"{m}")
            self.price[m.id] = {
                "prompt_tokens": m.pricing["prompt"],
                "completion_tokens": m.pricing["completion"],
            }

    def get_usage(self, days=10):
        url = f"{self.get_endpoint()}credits"
        r = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        d = r.json()["data"]
        print(f"Total Credits: {d['total_credits']}")
        print(f"Total Usage: {d['total_usage']:.6f}")

    def print_model_pricing_table(self, filter=None):
        super().print_model_pricing_table(self.price, filter= filter)

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

    def build_payload(self, prompt: str) -> Dict:
        return {"input": {"prompt": prompt}, "model": self.model}

    def get_headers(self) -> Dict:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_endpoint(self) -> str:
        return f"https://api.replicate.com/v1/predictions"

    def handle_response(self, response_json: Dict) -> str:
        output = response_json.get("output", "")
        track_usage(self.model, "replicate", 0)
        return output

def get_prompt(filename):
    logger.debug(f"Prompt file is {filename}")
    if os.path.exists(filename):
        with open(filename, "r") as f:
            system_prompt = "\n".join(f.readlines())
            logger.debug(f"Prompt {system_prompt}")
        return system_prompt
    else:
        return ""

provider_map = {
    "openai": OpenAIClient,
    "openrouter": OpenRouterClient,
    "together": TogetherClient,
    "replicate": ReplicateClient,
    "deepinfra": DeepInfraClient,
}


def main():
    # print(f"{current_script_path=}")
    prompt_choices = get_filenames_without_extension(
        str(current_script_path) + "/prompts"
    )
    prompt_choices += model_prompts.keys()
    parser = argparse.ArgumentParser(
        description="💬 GPT CLI Tracker: Ask OpenAI models, track usage and cost.",
        formatter_class=RichHelpFormatter,
        epilog="""
          gpt "What's the capital of France?"
          gpt "Refactor this function" --model openai/gpt-4
          gpt --history
          gpt --total
          gpt --list-models
          gpt "Give me a plan for a YouTube channel" --use-prompt
        """,
        #formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("query", nargs="?", type=str, help="Your prompt or question")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        #default="openai/gpt-4o-mini",
        default="openai/gpt-3.5-turbo",
        help="Specify the model to use (default: openai/gpt-4o-mini). Available models can be listed using --list-models.",
    )
    parser.add_argument(
        "-up",
        "--use-prompt",
        type=match_abbreviation(prompt_choices),
        default="default",
        help=f"Specify a prompt type. Options: {', '.join(prompt_choices)}. Default is 'general'.",
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
        nargs='?',
        const=10,
        # default=10,
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
        "-c", "--chat-id", default=None, help="Continue in same chat session"
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=provider_map.keys(),
        default="openrouter",
        help="Select LLM provider",
    )
    parser.add_argument(
        "-mt",
        "--max-tokens",
        default=4096,
        help="Use low cost LLM for OpenRouter",
    )
    parser.add_argument(
        "-fi",
        "--filter",
        type=str,
        help="Set the logging level.",
    )
    parser.add_argument("-o", "--output", default=None, help="The output file to use.")

    args = parser.parse_args()
    model = args.model
    logger.debug(f"Arguments {args=}")
    try:
        logger.setLevel(getattr(logging, args.log_level.upper()))
    except Exception as e:
        logger.error(f"Invalid log level: {e}")
    # print(f"Log Level: {args=}")

    if args.output:
        if os.path.exists(args.output):
            logger.critical(f"File already exists {args.output}")
            sys.exit(-2)
        else:
            logger.debug(f"Setting output file as {args.output=}")
    system_prompt = None
    if args.use_prompt:
        logger.debug(f"Getting file {current_script_path}/prompts/common_base.txt")
        base_prompt = get_prompt(f"{current_script_path}/prompts/common_base.txt")
        logger.debug(f"Base prompt: {base_prompt}")
        sprompt = ""
        filename = f"{current_script_path}/prompts/{args.use_prompt}.txt"
        if os.path.exists(filename):
            sprompt = get_prompt(filename)
        else:
            sprompt = model_prompts[args.use_prompt]["system_prompt"]
            model = model_prompts[args.use_prompt]["model"]
            logger.critical(f"Prompt File {filename} does not exist")
            logger.debug(f"Using {system_prompt=}")
            logger.debug(f"Using {model=}")
            # sys.exit(-1)
        system_prompt = f"""
            {base_prompt}

            {sprompt}
        """
        logger.debug(f"Using system prompt : {system_prompt}")
    client_class = provider_map[args.provider]
    llm_client = client_class(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt=system_prompt,
        temperature=args.temp,
        file=args.file,
        output=args.output,
        max_tokens = args.max_tokens,
    )

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
        # print("🧠 Querying model...\n")
        try:
            response = llm_client.send_request(args.query)
            logger.debug(f"Response: {response=}")
            llm_client.handle_response(response)
        except Exception as e:
            logger.exception(f"❌ Error while querying: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
