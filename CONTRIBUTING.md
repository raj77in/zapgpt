# Contributing to ZapGPT

Thank you for your interest in contributing to ZapGPT! This document provides guidelines and information for contributors.

## 🚀 Quick Start

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/zapgpt.git
   cd zapgpt
   ```

2. **Set up development environment (recommended)**
   ```bash
   # With uv (recommended)
   uv sync
   uv run zapgpt --help
   
   # Or with pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. **Set up API keys for testing**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # Add other provider keys as needed
   ```

## 🧪 Testing

### Run Tests
```bash
# Run verification tests
python verify_install.py

# Test specific functionality
uv run zapgpt --config
uv run zapgpt --list-prompt
uv run zapgpt --show-prompt coding
```

### Manual Testing
```bash
# Test different providers
uv run zapgpt --provider openai "Hello"
uv run zapgpt --provider openrouter "Hello"

# Test prompt system
uv run zapgpt --use-prompt coding "Write a hello world function"
```

## 📝 Code Style

### Python Code Standards
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for all functions and classes
- Keep functions focused and modular

### Commit Message Format
```
type(scope): description

Examples:
feat(providers): add support for Anthropic Claude
fix(config): resolve pricing file loading issue
docs(readme): update installation instructions
```

## 🔧 Adding New Features

### Adding a New LLM Provider

1. **Create provider class** in `zapgpt/main.py`:
   ```python
   class NewProviderClient(BaseLLMClient):
       def __init__(self, model: str, api_key: str, **kwargs):
           super().__init__(model=model, **kwargs)
           self.api_key = api_key
           # Provider-specific setup
       
       def send_request(self, prompt: str) -> str:
           # Implement API call
           pass
   ```

2. **Add to provider mapping**:
   ```python
   provider_map = {
       # ... existing providers
       "newprovider": NewProviderClient,
   }
   
   provider_env_vars = {
       # ... existing vars
       "newprovider": "NEWPROVIDER_API_KEY",
   }
   ```

3. **Update documentation** in README.md

### Adding New Prompts

1. Create JSON file in `zapgpt/prompts/`:
   ```json
   {
       "system_prompt": "Your system prompt here",
       "model": "openai/gpt-4o-mini",
       "assistant_input": "Optional assistant input"
   }
   ```

2. Test with: `zapgpt --use-prompt yourprompt "test"`

## 🐛 Bug Reports

### Before Submitting
- Check existing issues
- Test with latest version
- Provide minimal reproduction case

### Bug Report Template
```markdown
**Description**
Brief description of the bug

**Steps to Reproduce**
1. Run command: `zapgpt ...`
2. Expected: ...
3. Actual: ...

**Environment**
- OS: 
- Python version:
- ZapGPT version:
- Provider used:

**Logs**
```
Include relevant logs with -ll DEBUG
```

## 🎯 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why would this feature be useful?

**Proposed Implementation**
Any ideas on how this could be implemented?
```

## 📋 Pull Request Process

1. **Fork and create branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write code following style guidelines
   - Add tests if applicable
   - Update documentation

3. **Test thoroughly**
   ```bash
   python verify_install.py
   # Test your specific changes
   ```

4. **Submit PR**
   - Clear title and description
   - Reference any related issues
   - Include testing instructions

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] No breaking changes (or clearly documented)

## 🏗️ Architecture Overview

### Key Components
- **`main.py`**: Core application logic and provider implementations
- **`config.py`**: Configuration management and user directory handling
- **`cli.py`**: Command-line interface setup
- **`prompts/`**: Default prompt templates
- **`default_pricing.json`**: Default pricing data for providers

### Design Principles
- **User-friendly**: Configuration in standard locations
- **Extensible**: Easy to add new providers
- **Reliable**: Graceful error handling
- **Fast**: Efficient API calls and caching where appropriate

## 🤝 Community

### Getting Help
- Open an issue for bugs or questions
- Check existing documentation first
- Be respectful and constructive

### Code of Conduct
- Be respectful to all contributors
- Focus on constructive feedback
- Help newcomers get started
- Maintain a welcoming environment

## 📄 License

By contributing to ZapGPT, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to ZapGPT! 🚀
