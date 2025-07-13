#!/usr/bin/env python3
"""
Test suite for ZapGPT CLI functionality
"""

import os
import pytest
import subprocess
import tempfile
from pathlib import Path

# Test data directory
TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent


class TestCLIBasics:
    """Test basic CLI functionality"""
    
    def test_cli_help(self):
        """Test that CLI help works"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        assert 'zapgpt' in result.stdout.lower()
        assert 'usage' in result.stdout.lower()
    
    def test_cli_version_info(self):
        """Test that CLI shows version information"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        # Should contain version or description
        assert any(word in result.stdout.lower() for word in ['version', 'gpt', 'llm'])


class TestCLIFlags:
    """Test CLI flags and options"""
    
    def test_quiet_flag(self):
        """Test --quiet flag is available"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        assert '--quiet' in result.stdout or '-q' in result.stdout
    
    def test_file_flag(self):
        """Test --file flag is available"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        assert '--file' in result.stdout or '-f' in result.stdout
    
    def test_provider_flag(self):
        """Test --provider flag is available"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        assert '--provider' in result.stdout or '-p' in result.stdout
    
    def test_model_flag(self):
        """Test --model flag is available"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        assert '--model' in result.stdout or '-m' in result.stdout


class TestCLIFileInput:
    """Test CLI file input functionality"""
    
    def test_file_input_structure(self):
        """Test that file input works structurally"""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Test log entry\n")
            f.write("2025-01-13 01:30:00 INFO: Sample log message\n")
            f.write("2025-01-13 01:30:05 WARNING: Test warning\n")
            temp_file = f.name
        
        try:
            # Set dummy API key
            env = os.environ.copy()
            env['OPENAI_API_KEY'] = 'dummy_key_for_testing'
            
            # Test file input (will fail at API call but structure should work)
            result = subprocess.run(
                ['python', '-m', 'zapgpt', '--file', temp_file, '--quiet', 'Analyze this file'],
                capture_output=True, text=True, cwd=PROJECT_DIR, env=env
            )
            
            # Should either succeed or fail with authentication error (not file error)
            if result.returncode != 0:
                error_output = result.stderr.lower()
                # Should be API-related error, not file-related
                assert not any(word in error_output for word in ['file not found', 'no such file', 'permission denied'])
        
        finally:
            # Cleanup
            os.unlink(temp_file)


class TestCLIPrompts:
    """Test CLI prompt functionality"""
    
    def test_list_prompts(self):
        """Test --list-prompt functionality"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--list-prompt'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        # Should list some prompts
        assert len(result.stdout.strip()) > 0
    
    def test_show_prompt_help(self):
        """Test --show-prompt flag exists"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        assert '--show-prompt' in result.stdout
    
    def test_use_prompt_help(self):
        """Test --use-prompt flag exists"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        assert '--use-prompt' in result.stdout


class TestCLIConfiguration:
    """Test CLI configuration functionality"""
    
    def test_config_command(self):
        """Test --config command works"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--config'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        # Should show configuration information
        output = result.stdout.lower()
        assert any(word in output for word in ['config', 'directory', 'path'])


class TestCLIProviders:
    """Test CLI provider functionality"""
    
    def test_provider_options_in_help(self):
        """Test that provider options are shown in help"""
        result = subprocess.run(
            ['python', '-m', 'zapgpt', '--help'], 
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        
        assert result.returncode == 0
        # Should mention some providers
        output = result.stdout.lower()
        assert any(provider in output for provider in ['openai', 'openrouter', 'together'])


if __name__ == "__main__":
    pytest.main([__file__])
