# zapgpt

A command line tool to use chatgpt - currently only OpenAI API is supported.

███████╗ █████╗ ██████╗  ██████╗ ██████╗ ████████╗
╚══███╔╝██╔══██╗██╔══██╗██╔════╝ ██╔══██╗╚══██╔══╝
  ███╔╝ ███████║██████╔╝██║  ███╗██████╔╝   ██║
 ███╔╝  ██╔══██║██╔═══╝ ██║   ██║██╔═══╝    ██║
███████╗██║  ██║██║     ╚██████╔╝██║        ██║
╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝        ╚═╝
         GPT on the CLI. Like a boss.

`zapgpt` is a minimalist CLI tool to chat with LLMs from your terminal. No bloated UI, just fast raw GPT magic, straight from the shell. With pre-cooked system prompt for Ethical hacking, code, file attachment and a good default prompt and usage tracking, I hope you find it useful. No extra features or frills. Modify it as you need it with a simple one file script.

## 💾 Requirements

* Python 3.8+
* pip
* `uv` (optional, fast venv & package manager)

## 🚀 Installation

### Option 1: Use `uv` (🔥 fast)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
````

### Option 2: Classic `venv` & `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🧠 Usage

```bash
python zapgpt.py "What's the meaning of life?"
```

Or keep it interactive:

```bash
python zapgpt.py
```

## 🛠️ Features

* Context-aware prompts (memory)
* Easily customizable for your LLM endpoints
* Show your current usage.
* Optional pre-cooked system prompts.

## 🧪 Example

```bash
$ python zapgpt.py "Summarize the Unix philosophy."
> Small is beautiful. Do one thing well. Write programs that work together.
```

## 🙌 Credits

Built with ❤️ by [Amit Agarwal aka](https://github.com/raj77in) — because LLMs deserve a good CLI.

## 🧙‍♂️ License

MIT — do whatever, just don't blame me if it becomes sentient.
