# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-07-13

### 🎉 Major Release - Complete Rewrite

This is a major release with significant architectural improvements and new features.

### Added
- **Multi-Provider Support**: OpenAI, OpenRouter, Together, Replicate, DeepInfra, GitHub AI
- **User Configuration System**: All config stored in `~/.config/zapgpt/`
- **Customizable Prompts**: User can modify and add prompts in config directory
- **Customizable Pricing**: Pricing data now user-configurable
- **Provider-Specific API Keys**: Only requires API key for selected provider
- **Prompt Preview**: `--show-prompt` to see complete prompt before sending
- **Model Override**: `-m` flag now takes precedence over prompt defaults
- **Configuration Display**: `--config` shows all configuration paths and status
- **Professional Package Structure**: Proper Python package with modules
- **Rich CLI Output**: Beautiful terminal output with colors and formatting

### Changed
- **BREAKING**: Configuration moved from package directory to `~/.config/zapgpt/`
- **BREAKING**: Database moved to `~/.config/zapgpt/gpt_usage.db`
- **BREAKING**: Environment variable validation now provider-specific
- **Improved**: Installation now supports `uv tool install` (recommended)
- **Improved**: Better error messages and user guidance
- **Improved**: Modular code architecture for maintainability

### Fixed
- **Critical**: No longer requires all provider API keys to be set
- **Critical**: Proper model override priority (CLI flag > prompt default)
- **Improved**: Better error handling and user feedback
- **Improved**: Consistent logging throughout application

### Technical Improvements
- Proper Python package structure with entry points
- XDG Base Directory compliance for configuration
- Extensible provider architecture
- Comprehensive test suite
- Professional documentation
- CI/CD ready structure

## [2.0.0] - 2025-06-20

### Added
- Initial PyPI package structure
- Basic multi-provider framework
- Prompt management system

## [1.0.0] - 2025-05-06

### Added
- Initial release
- OpenAI integration
- Basic CLI functionality
