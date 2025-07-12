# 🔧 ZapGPT Configuration System

ZapGPT now uses a proper user configuration system that follows XDG Base Directory standards.

## 📁 Configuration Directory Structure

```
~/.config/zapgpt/
├── prompts/                    # User-customizable prompts
│   ├── coding.json
│   ├── cyber_awareness.json
│   ├── vuln_assessment.json
│   ├── kalihacking.json
│   ├── prompting.json
│   ├── powershell.json
│   ├── default.json
│   └── common_base.json
└── gpt_usage.db               # Usage tracking database
```

## 🚀 How It Works

### First Run
1. **Automatic Setup**: On first run, zapgpt creates `~/.config/zapgpt/`
2. **Default Prompts**: Copies 8 default prompts from the package to user config
3. **Database Creation**: Creates the usage tracking database in the config directory

### Subsequent Runs
- **Loads User Prompts**: Reads prompts from `~/.config/zapgpt/prompts/`
- **User Customization**: Users can modify existing prompts or add new ones
- **Persistent Storage**: All data stays in the user's config directory

## 🛠️ User Commands

### View Configuration
```bash
zapgpt --config
```
Shows:
- Configuration directory location
- Prompts directory location
- Database file location
- List of available prompts

### Manage Prompts
```bash
# List available prompts
zapgpt --list-prompt

# Use a specific prompt
zapgpt --use-prompt coding "Refactor this function"

# Add custom prompts (create .json files in ~/.config/zapgpt/prompts/)
```

## 📝 Custom Prompt Format

Create `.json` files in `~/.config/zapgpt/prompts/` with this structure:

```json
{
    "system_prompt": "Your custom system prompt here",
    "model": "openai/gpt-4o-mini",
    "assistant_input": "Optional assistant input"
}
```

## ✅ Benefits

1. **User-Friendly**: Prompts are easily accessible and modifiable
2. **Persistent**: Configuration survives package updates
3. **Standard Compliant**: Follows XDG Base Directory specification
4. **Portable**: Users can backup/share their config directory
5. **Clean**: No configuration files cluttering the package directory

## 🔄 Migration from Old System

The old system stored prompts in the package directory. The new system:
- ✅ Automatically copies default prompts to user config
- ✅ Maintains backward compatibility during development
- ✅ Provides clear migration path for existing users

## 🎯 For Developers

When developing zapgpt:
- Prompts are loaded from `~/.config/zapgpt/prompts/`
- Database is stored in `~/.config/zapgpt/gpt_usage.db`
- Default prompts are bundled with the package for first-time setup
- Configuration system handles both development and installed scenarios

This approach ensures that user customizations are preserved while maintaining a clean package structure for PyPI distribution.
