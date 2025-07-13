#!/bin/bash
# Setup script for pre-commit hooks in ZapGPT project

set -e  # Exit on any error

echo "🔧 Setting up pre-commit hooks for ZapGPT..."
echo ""
echo "Choose configuration:"
echo "1) Full (comprehensive checks including tests, type checking, security)"
echo "2) Simple (fast formatting and basic checks)"
echo "3) Development (recommended - linting only, no formatting conflicts)"
echo ""
read -p "Enter choice (1, 2, or 3, default: 3): " choice
choice=${choice:-3}

if [ "$choice" = "1" ]; then
    echo "📝 Using full configuration..."
    # Keep existing .pre-commit-config.yaml
elif [ "$choice" = "2" ]; then
    echo "📝 Using simplified configuration..."
    cp .pre-commit-config-simple.yaml .pre-commit-config.yaml
else
    echo "📝 Using development-friendly configuration (recommended)..."
    cp .pre-commit-config-dev.yaml .pre-commit-config.yaml
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository. Please run this from the project root."
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install uv first."
    echo "   Visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    uv add --dev pre-commit
else
    echo "✅ pre-commit is already installed"
fi

# Install the pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
uv run pre-commit install

# Install commit-msg hook for conventional commits (optional)
echo "📝 Installing commit-msg hook..."
uv run pre-commit install --hook-type commit-msg

# Run pre-commit on all files to ensure everything works
echo "🧪 Running pre-commit on all files (this may take a while)..."
uv run pre-commit run --all-files || {
    echo "⚠️  Some pre-commit checks failed. This is normal for the first run."
    echo "   The hooks will automatically fix many issues."
    echo "   Please review the changes and commit them."
}

echo ""
echo "🎉 Pre-commit hooks setup complete!"
echo ""
echo "📋 What happens now:"
echo "   • Before each commit, the following checks will run automatically:"
echo ""
echo "📚 Documentation:"
echo "   - Setup guide: docs/pre-commit-setup.md"
echo "   - Troubleshooting: docs/TROUBLESHOOTING.md"
echo ""
echo "🚀 Happy coding with automated quality checks!"
