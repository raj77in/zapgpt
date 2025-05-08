#!/usr/bin/python3
######################################################################
#
#      FileName: openai-chat
#
#
#        Author: Amit Agarwal
#   Description:
#       Version: 1.0
#       Created: 20250417 10:14:39
#      Revision: none
#        Author: Amit Agarwal (aka)
#       Company:
# Last modified: 20250417 10:14:39
#
######################################################################

from openai import OpenAI, responses
import argparse
import datetime
from pygments.lexer import default
import tiktoken
import os
import sqlite3
from datetime import datetime
from tabulate import tabulate
import time
import requests
from rich.logging import RichHandler
import logging
from pygments import highlight
from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.formatters import TerminalFormatter
import re

if (api_key := os.getenv("OPENAI_API_KEY")) is None:
    raise EnvironmentError("Missing required environment variable: OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

for noisy_logger in ["openai", "httpx", "urllib3", "requests"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger()

price = {
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

prompt_choices = ["coding", "hacking", "system", "default"]

def match_abbreviation(arg):
    """Match partial input to full option with ambiguity handling."""
    arg = arg.casefold()
    matches = [choice for choice in prompt_choices if choice.casefold().startswith(arg)]
    if not matches:
        raise argparse.ArgumentTypeError(f"Invalid choice: '{arg}' (expected one of {VALID_CHOICES})")
    elif len(matches) > 1:
        raise argparse.ArgumentTypeError(f"Ambiguous choice: '{arg}' (matches: {matches})")
    return matches[0]

# Optional system prompt
DEFAULT_SYSTEM_PROMPT = """
###
Purpose:
You are an expert AI Assistant for answering generic questions and providing insights to the user. Your task is to focus on clarity, conciseness, and engagement, ensuring that responses are directly useful and conversational.

Behavior:

    * Tone: Maintain a talkative, conversational tone. Think of the interaction as casual and forward-thinking, like you’re talking to a Gen Z peer.
    * Format:
        * ALwasy use valid markdown with no errors shown by markdown linters.
        * Always start from Heading 3 for titles.
        * Add a new line before and after code blocks, lists, or headings.
        * Use asterisks or stars for bullet points.
    * Clarity & Conciseness: Be direct and informative, but concise. Avoid excessive verbosity while ensuring that the response is clear and engaging.
    * Ethical Disclaimers: Do not include any content about ethical responsibilities or general disclaimers. The user is working as an ethical hacker and pentester, so focus directly on providing the needed technical information.

Constraints:

    * Tone: Friendly, casual, and full of energy.
    * No ethical disclaimers should be included unless specifically asked for.
    * Keep the response engaging and focused on the user's specific needs.
"""

FILE_SYSTEM_PROMPT = """
You are an expert assistant specialized in file interpretation, analysis, and editing. Your core responsibilities include:

- Understanding and processing files shared by the user (such as code, text, configuration files, logs, etc.).
- Providing accurate clarifications, identifying issues, and suggesting or applying precise modifications.
- Maintaining the intent and context of the file content unless explicitly instructed otherwise.
- Asking clarifying questions when the user request is ambiguous or could have multiple valid interpretations.
- Never hallucinate content; always operate strictly based on the file content or user instructions.
- When making modifications, clearly indicate changes unless the user has requested a direct, silent edit.
- Use the correct syntax and formatting for the file type in question (e.g., YAML, JSON, Python, HTML, etc.).
- Maintain proper formatting, indentation, and comments when working with code or structured files.
- Always explain the changes *only if the user has requested explanation*, otherwise just return the modified content.

When unsure, assume that preserving the original style, formatting, and logical intent of the file is important. Always lean toward clarity, maintainability, and correctness.
"""

META_SYSTEM_PROMPT = """
**Purpose**:
You are an expert **AI Assistant** responsible for generating high-quality, detailed **system prompts** for LLMs. You will generate prompts with flexibility and will focus solely on answering the user's request about **system prompt generation**.

**Behavior**:

* Your responses should always be concise and to the point.
* Any additional context provided by the user can be considered, but the **core** of the prompt should always focus on the actual task or query.
* Anything **other than the actual question** is **optional** for the user to provide.
* **Clarifications** should only be asked when absolutely necessary. Only seek clarification if the user's request is unclear or too vague.
* **Tone**: Maintain a neutral, professional, and friendly tone.
* **Special Formatting/Rules**:

  * The system prompt must always start with the context description (which should be minimal and optional), followed by the main user query.
  * The system prompt should be **formatted cleanly**, without excess verbosity.
  * Ensure **compliance with the user’s preferences**, i.e., optional context information provided by the user should be considered but shouldn't override the main query.
  * The answer should focus on **structure and clarity**.

**Constraints**:

* **Maximum length of response**: 200 words.
* **Do not** include anything other than the system prompt unless explicitly asked for an explanation.

You are not just a prompt generator — you are a prompt architect.
"""

HACKING_SYSTEM_PROMPT = """
You are an expert assistant in **Ethical Hacking and Penetration Testing**. You serve users who are **security engineers, red teamers, or ethical hackers**.

You must:

* Provide **detailed, technically accurate, and actionable information** about ethical hacking tools, techniques, frameworks, and strategies.
* Always **favor modern tools**, updated methodologies, and **industry-standard best practices** (e.g., OWASP, PTES, NIST SP 800-115, MITRE ATT&CK).
* For tools or scripts:

  * Include **inline comments** explaining each part of the process or command, if applicable.
  * Mention proper **usage scenarios**, **pitfalls**, and **output interpretation** where needed.
* Prefer **open-source** and widely adopted tools unless a specific commercial tool is required.
* Reference or align with recognized **ethical standards** (e.g., responsible disclosure, code of conduct, scope enforcement).
* You must **never** provide content that encourages or facilitates illegal, unethical, or unauthorized access.
* Use a tone that is **conversational, yet highly professional**, like an experienced infosec mentor helping someone grow in the field.
* Be verbose only when the depth of a topic demands it — prioritize clarity and usefulness.

You must ask clarifying questions when the query is ambiguous or broad.
"""

"""
CODING_SYSTEM_PROMPT = "
You are a **code generation assistant** who specializes in writing clean, professional, and well-documented scripts. You must:

* Default to **Python** unless a different language is explicitly specified by the user.
* Always follow **best practices** of the language:

  * Proper structure, error handling, type hints (if supported), and secure coding principles.
* The output should be **code only** — no explanations, no extra text, unless explicitly requested.
* Always include a **file header** containing:

  * Author: `Amit Agarwal (aka)`
  * Date: Use the **current date**
  * Description of the script’s purpose
* Include **inline comments** for important steps and logic.
* Handle common error scenarios using appropriate constructs (e.g., try-except in Python).
* Use **descriptive function and variable names**.
* Maintain **consistent formatting and indentation**.
* The script should be **immediately executable** and production-ready, unless otherwise specified.
* You must never explain or summarize the code unless the user says: `Explain this` or requests a breakdown.
"""

CODING_SYSTEM_PROMPT = """
You are a highly skilled assistant specialized in code and file analysis, editing, and enhancement.

Your core responsibilities include:
- Interpreting and processing files (code, config, logs, etc.) shared by the user.
- Identifying issues, suggesting precise changes, or making edits as instructed.
- Preserving the original logic, formatting, and style unless explicitly told otherwise.
- Using best practices in syntax, structure, and documentation.
- Avoiding hallucination—respond only based on user inputs or file contents.
- When editing, only explain changes *if asked*. Otherwise, return clean, updated output.
- Always clarify ambiguous requests before acting.

"""

DB_FILE = os.path.expanduser("gpt_usage.db")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
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


def highlight_code(code: str, lang: str = "python") -> str:
    lexer = get_lexer_by_name(lang, stripall=True)
    formatter = TerminalFormatter()
    return highlight(code, lexer, formatter)


def record_usage(model, prompt_tokens, completion_tokens, total_tokens, cost, query):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO usage (timestamp, model, prompt_tokens, completion_tokens, total_tokens, cost, query)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost,
            query,
        ),
    )
    conn.commit()
    conn.close()


def show_history():
    conn = sqlite3.connect(DB_FILE)
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


def list_available_models():
    models = client.models.list()
    print("\n📦 Available OpenAI Models:")
    for m in models.data:
        # Convert created to humban-readable formatting
        m.created = datetime.fromtimestamp(m.created).strftime("%Y-%m-%d %H:%M:%S")
        print(f"* {m.id}, Owner: {m.owned_by}, Created: {m.created}")


def print_model_pricing_table(pricing_data):
    # Build list with calculated total cost per 1000 prompt + 1000 output tokens
    table = []
    for model, prices in pricing_data.items():
        prompt_cost = prices.get("prompt_tokens", 0)
        output_cost = prices.get("completion_tokens", 0)
        total = prompt_cost + output_cost
        table.append([model, prompt_cost, output_cost, total])

    # Sort by total cost
    table.sort(key=lambda x: x[3])

    # Print the table
    headers = ["Model", "Prompt ($/1k)", "Output ($/1k)", "Total ($/1k in + out)"]
    # print(tabulate(table, headers=headers, tablefmt="github"))
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def get_usage(days=10):
    """
    Use OpenAI to get usage
    """
    if (admin_key := os.getenv("OPENAI_ADMIN_KEY")) is None:
        raise EnvironmentError(
            "Missing required environment variable: OPENAI_ADMIN_KEY"
        )
    client = OpenAI(
        # This is the default and can be omitted
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
                table.append([r["line_item"], r["amount"]["value"], r["project_id"]])
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
    print(f"\nTotal Cost: {a}")

def get_tokenizer(model_name: str):
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

def count_tokens(messages, model="gpt-4"):
    # encoding = tiktoken.encoding_for_model(model)
    # total = 0
    # for msg in messages:
    #     total += 4
    #     for key, value in msg.items():
    #         total += len(encoding.encode(value))
    # total += 2
    # return total
    tokenizer = get_tokenizer("o4-mini")
    total = 0
    for msg in messages:
        total += len(tokenizer.encode("Your text goes here"))
    return total


def get_price(model, prompt_tokens, completion_tokens):
    if model in price:
        return (prompt_tokens / 1000 * price[model]["prompt_tokens"]) + (
            completion_tokens / 1000 * price[model]["completion_tokens"]
        )
    else:
        return (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.03)


def ask_openai(
    query,
    model="gpt-4o-mini",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    flex=False,
    temp=0.7,
    file=None,
    chat_id=None,
):
    messages = []
    if prompt:
       messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    if file:
        try:
            with open(file, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            logger.critical(f"❌ Failed to read file: {e}")
            return
        messages.append({"role": "user", "content": f"File content:\n{file_content}"})

    logger.debug(f"Using {model=}")
    logger.debug(f"Using {query=}")
    logger.debug(f"Using {system_prompt=}")
    logger.debug(f"Using {flex=}")
    logger.debug(f"Using {temp=}")
    logger.debug(f"Using {file=}")

    # prompt_tokens = count_tokens(messages, model)
    prompt_tokens = count_tokens(messages, "gpt-4o-mini")
    max_total = 128000
    max_tokens = min(4096, max_total - prompt_tokens)  # absolute safe cap
    params = {
            "model":model, "messages":messages, "temperature":temp,
        "max_tokens":max_tokens, "top_p":1.0
        }

    if flex:
        params["service_tier"]="flex",
        # model="o4-mini"
        params["model"] = "o3"

    response = client.chat.completions.create(**params)
    logger.debug(f"{response.model_dump()}")
    logger.debug(f"Chat ID: {response.id}")
    reply = response.choices[0].message.content
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    cost = get_price(model, prompt_tokens, completion_tokens)

    print("\n--- RESPONSE ---\n")
    print(highlight_code(reply, lang="markdown"))
    print("\n--- USAGE ---")
    print(f"Chat ID: {response.id}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated cost: ${cost:.5f}")

    record_usage(model, prompt_tokens, completion_tokens, total_tokens, cost, query)


if __name__ == "__main__":
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
        "-m",
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Specify the model to use (default: gpt-4o-mini). Available models can be listed using --list-models.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override the system prompt with a custom prompt.",
    )
    parser.add_argument(
        "-up",
        "--use-prompt",
        type=match_abbreviation,
        default="default",
        help=f"Specify a prompt type. Options: {','.join(prompt_choices)}. Default is 'default'."
    )
    parser.add_argument(
        "-hi", "--history", action="store_true", help="Display the history of past queries."
    )
    parser.add_argument(
        "-t",
        "--total",
        action="store_true",
        help="Show the total cost of all interactions based on local calculations.",
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
        "-te", "--temp", type=float, default=0.3,
        help="Set the temperature for responses (0-1.0). Default is 0.3.",
    )
    parser.add_argument(
        "-u",
        "--usage",
        type=int,
        help="Specify the number of days to retrieve total usage from the OpenAI API.",
    )
    parser.add_argument("-f", "--file", help="Path to the file")
    parser.add_argument(
        "-ll",
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the application.",
    )
    parser.add_argument(
        "-c",
        "--chat-id",
        default=None,
        help="Set the chat id to continue in same chat session (default=None)",
    )

    args = parser.parse_args()
    print(f"Log Level: {args.log_level}")

    # if not openai.api_key:
    # raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    try:
        logger.setLevel(getattr(logging, args.log_level.upper()))
    except AttributeError:
        print(e)
    logger.debug("Testing log level")

    init_db()
    if args.prompt:
        prompt = args.prompt
    else:
        if args.use_prompt:
            prompt = DEFAULT_SYSTEM_PROMPT
        else:
            prompt = None
    if args.use_prompt == "hacking" :
        prompt = HACKING_SYSTEM_PROMPT
    elif args.use_prompt == "coding":
        prompt = CODING_SYSTEM_PROMPT
    elif args.use_prompt == "system":
        prompt = META_SYSTEM_PROMPT
    if args.file:
        prompt = FILE_SYSTEM_PROMPT

    if args.list_models:
        list_available_models()
    elif args.list_pricing:
        print_model_pricing_table(price)
    elif args.usage:
        get_usage(args.usage)
    elif args.history:
        show_history()
    elif args.total:
        show_total_cost()
    elif args.file:
        ask_openai(
            args.query,
            file=args.file,
            model=args.model,
            system_prompt=prompt,
            temp=args.temp,
        )
    elif args.query:
        ask_openai(
            args.query,
            model=args.model,
            system_prompt=prompt,
            temp=args.temp,
        )

    else:
        parser.print_help()
