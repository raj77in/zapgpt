# zapgpt

![Intro image](intro.png)

A minimalist CLI tool to chat with LLMs from your terminal. Supports multiple providers including OpenAI, OpenRouter, Together, Replicate, DeepInfra, and GitHub AI.

```plaintext
███████╗ █████╗ ██████╗  ██████╗ ██████╗ ████████╗
╚══███╔╝██╔══██╗██╔══██╗██╔════╝ ██╔══██╗╚══██╔══╝
  ███╔╝ ███████║██████╔╝██║  ███╗██████╔╝   ██║
 ███╔╝  ██╔══██║██╔═══╝ ██║   ██║██╔═══╝    ██║
███████╗██║  ██║██║     ╚██████╔╝██║        ██║
╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝        ╚═╝
         GPT on the CLI. Like a boss.
```

`zapgpt` is a minimalist CLI tool to chat with LLMs from your terminal. No bloated UI, just fast raw GPT magic, straight from the shell. With pre-cooked system prompt for Ethical hacking, code, file attachment and a good default prompt and usage tracking, I hope you find it useful. No extra features or frills. Modify it as you need it with a simple one file script.

Updated to version v2.

## Introduction video

[![Introduction](https://i.ytimg.com/vi/hpiVtj_gSD4/hqdefault.jpg)](https://www.youtube.com/watch?v=hpiVtj_gSD4)

## 💾 Requirements

* Python 3.8+
* `uv` (recommended - blazingly fast Python package manager)
* pip (alternative to uv)

## 🚀 Installation

### Option 1: Install with `uv` (⚡ Recommended)

```bash
uv tool install zapgpt
```

> **Why uv?** `uv` is blazingly fast and handles CLI tools perfectly. It installs zapgpt globally and manages dependencies automatically.

**Don't have uv?** Install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Option 2: Install from PyPI

```bash
uv tool install zapgpt
```

### Option 3: Development Installation

**With uv (recommended):**

```bash
git clone https://github.com/yourusername/zapgpt.git
cd zapgpt
uv sync
uv run zapgpt "test"

# Optional: Set up pre-commit hooks for code quality
./setup-pre-commit.sh
```

**With pip:**

```bash
git clone https://github.com/yourusername/zapgpt.git
cd zapgpt
pip install -e .
```

### Option 4: From Source (Classic method)

```bash
git clone https://github.com/yourusername/zapgpt.git
cd zapgpt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🔑 Environment Variables

ZapGPT only requires the API key for the provider you're using. Set the appropriate environment variable:

| Provider | Environment Variable | Get API Key |
|----------|---------------------|-------------|
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/account/api-keys) |
| OpenRouter | `OPENROUTER_KEY` | [openrouter.ai](https://openrouter.ai/keys) |
| Together | `TOGETHER_API_KEY` | [api.together.xyz](https://api.together.xyz/settings/api-keys) |
| Replicate | `REPLICATE_API_TOKEN` | [replicate.com](https://replicate.com/account/api-tokens) |
| DeepInfra | `DEEPINFRA_API_TOKEN` | [deepinfra.com](https://deepinfra.com/dash/api_keys) |
| GitHub | `GITHUB_KEY` | [github.com](https://github.com/settings/tokens) |

**Example:**

```bash
# For OpenAI (default provider)
export OPENAI_API_KEY="your-openai-api-key-here"

# For OpenRouter
export OPENROUTER_KEY="your-openrouter-key-here"
```

## 🧠 Usage

After installation, you can use `zapgpt` directly from the command line:

```bash
# Basic usage (uses OpenAI by default)
zapgpt "What's the meaning of life?"

# Use different providers
zapgpt --provider openrouter "Explain quantum computing"
zapgpt --provider together "Write a Python function"
zapgpt --provider github "Debug this code"

# Use specific models
zapgpt -m gpt-4o "Complex reasoning task"
zapgpt --provider openrouter -m anthropic/claude-3.5-sonnet "Creative writing"
```

### Interactive Mode

```bash
zapgpt  # Starts interactive mode
```

### Development Usage

**With uv:**

```bash
uv run zapgpt "Your question here"
```

**With Python:**

```bash
python -m zapgpt "Your question here"
# or
python zapgpt/main.py "Your question here"
```

### Quiet Mode (for Scripting)

```bash
# Suppress all output except the LLM response
zapgpt --quiet "What is the capital of France?"

# Perfect for shell scripts
RESPONSE=$(zapgpt -q "Summarize this in one word: Machine Learning")
echo "Result: $RESPONSE"
```

### File Input (for Automation)

```bash
# Send file contents to LLM
zapgpt --file /path/to/file.txt "Analyze this log file"

# Analyze command output
nmap -sV target.com > scan_results.txt
zapgpt -f scan_results.txt --use-prompt vuln_assessment "Analyze these scan results"

# Process multiple files
for file in *.log; do
    zapgpt -q -f "$file" "Summarize security events" >> summary.txt
done
```

### Automation Examples

```bash
# Penetration Testing Agent
#!/bin/bash
TARGET="example.com"

# 1. Reconnaissance
nmap -sV $TARGET > nmap_results.txt
RESPONSE=$(zapgpt -q -f nmap_results.txt --use-prompt vuln_assessment "Identify potential vulnerabilities")
echo "Vulnerabilities found: $RESPONSE"

# 2. Web Analysis
nikto -h $TARGET > nikto_results.txt
zapgpt -f nikto_results.txt "Prioritize these web vulnerabilities" > web_analysis.txt

# 3. Generate Report
zapgpt -q "Create executive summary" -f web_analysis.txt > final_report.md
```

```bash
# Log Analysis Agent
#!/bin/bash
# Monitor and analyze system logs
tail -n 100 /var/log/auth.log > recent_auth.log
ALERT=$(zapgpt -q -f recent_auth.log "Detect suspicious login attempts")

if [[ $ALERT == *"suspicious"* ]]; then
    echo "Security Alert: $ALERT" | mail -s "Security Alert" admin@company.com
fi
```

```bash
# Code Review Agent
#!/bin/bash
# Automated code review
for file in src/*.py; do
    REVIEW=$(zapgpt -q -f "$file" --use-prompt coding "Review this code for security issues")
    echo "File: $file" >> code_review.md
    echo "Review: $REVIEW" >> code_review.md
    echo "---" >> code_review.md
done
```

## 🐍 Programmatic API

ZapGPT can be imported and used in your Python scripts:

### Basic Usage

```python
from zapgpt import query_llm

# Simple query
response = query_llm("What is Python?", provider="openai")
print(response)

# With different provider
response = query_llm(
    "Explain quantum computing",
    provider="openrouter",
    model="anthropic/claude-3.5-sonnet"
)
```

### Advanced Usage

```python
from zapgpt import query_llm

# Use predefined prompts
code_review = query_llm(
    "Review this Python function: def hello(): print('hi')",
    provider="openai",
    use_prompt="coding",
    model="gpt-4o"
)

# Custom system prompt
response = query_llm(
    "Write a haiku about programming",
    provider="openai",
    system_prompt="You are a poetic programming mentor.",
    temperature=0.8
)

# Error handling
try:
    response = query_llm("Hello", provider="openai")
except EnvironmentError as e:
    print(f"Missing API key: {e}")
except ValueError as e:
    print(f"Invalid provider: {e}")
```

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | Required | Your question/prompt |
| `provider` | str | "openai" | LLM provider to use |
| `model` | str | None | Specific model (overrides prompt default) |
| `system_prompt` | str | None | Custom system prompt |
| `use_prompt` | str | None | Use predefined prompt template |
| `temperature` | float | 0.3 | Response randomness (0.0-1.0) |
| `max_tokens` | int | 4096 | Maximum response length |
| `quiet` | bool | True | Suppress logging output |

### Environment Variables

Set the appropriate API key for your chosen provider:

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'

from zapgpt import query_llm
response = query_llm("Hello world", provider="openai")
```

### Python Automation Examples

```python
# Penetration Testing Automation
import subprocess
from zapgpt import query_llm

def analyze_nmap_scan(target):
    # Run nmap scan
    result = subprocess.run(['nmap', '-sV', target], capture_output=True, text=True)

    # Analyze with LLM
    analysis = query_llm(
        f"Analyze this nmap scan: {result.stdout}",
        provider="openai",
        use_prompt="vuln_assessment"
    )
    return analysis

vulns = analyze_nmap_scan("example.com")
print(f"Vulnerabilities: {vulns}")
```

```python
# Log Analysis Agent
from zapgpt import query_llm

def monitor_logs(log_file):
    with open(log_file, 'r') as f:
        logs = f.read()

    alert = query_llm(
        f"Detect suspicious activity: {logs}",
        provider="openai",
        quiet=True
    )

    if "suspicious" in alert.lower():
        print(f"ALERT: {alert}")
        return True
    return False

# Monitor auth logs
monitor_logs('/var/log/auth.log')
```

## Usage Video

[![Using zapgpt for pentesting on Kali](https://i.ytimg.com/vi/vDTwIsEUheE/hqdefault.jpg)](https://www.youtube.com/watch?v=hpiVtj_gSD4)

## 🛠️ Features

* Context-aware prompts (memory)
* Easily customizable for your LLM endpoints
* Show your current usage.
* Optional pre-cooked system prompts.

## 📝 Configuration & Prompts

ZapGPT stores its configuration and prompts in `~/.config/zapgpt/`:

* **Configuration directory**: `~/.config/zapgpt/`
* **Prompts directory**: `~/.config/zapgpt/prompts/`
* **Database file**: `~/.config/zapgpt/gpt_usage.db`

### Managing Prompts

On first run, zapgpt automatically copies default prompts to your config directory. You can:

* **View config location**: `zapgpt --config`
* **List available prompts**: `zapgpt --list-prompt`
* **Use a specific prompt**: `zapgpt --use-prompt coding "Your question"`
* **Add custom prompts**: Create `.json` files in `~/.config/zapgpt/prompts/`
* **Modify existing prompts**: Edit the `.json` files in your prompts directory

### Default Prompts Included

* `coding` - Programming and development assistance
* `cyber_awareness` - Cybersecurity guidance
* `vuln_assessment` - Vulnerability assessment help
* `kalihacking` - Kali Linux and penetration testing
* `prompting` - Prompt engineering assistance
* `powershell` - PowerShell scripting help
* `default` - General purpose prompt
* `common_base` - Base prompt added to all others

### v2 Features

* Script now uses class and is much more well organized.
* Prompts are not hard-coded in the script. You can simply drop in any new
  system prompt in prompts folder and use it.
* ✅ **Multi-Provider Support**: Supports OpenAI, OpenRouter, Together, Replicate, DeepInfra, and GitHub AI
* ✅ **Easy Provider Switching**: Use `--provider` flag to switch between providers
* ✅ **Model Selection**: Override model with `-m` flag for any provider

## 🧪 Example

```bash
$ zapgpt "Summarize the Unix philosophy."
> Small is beautiful. Do one thing well. Write programs that work together.
```

## 🙌 Credits

Built with ❤️ by [Amit Agarwal aka](https://github.com/raj77in) — because LLMs deserve a good CLI.

## 🧙‍♂️ License

MIT — do whatever, just don't blame me if it becomes sentient.
