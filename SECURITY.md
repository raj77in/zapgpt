# Security Policy

## Supported Versions

We actively support the following versions of ZapGPT with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 3.x.x   | ✅ Yes             |
| 2.x.x   | ❌ No              |
| 1.x.x   | ❌ No              |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in ZapGPT, please report it responsibly.

### How to Report

1. **Do NOT create a public issue** for security vulnerabilities
2. **Email**: Send details to [your-email@domain.com] with subject "ZapGPT Security Issue"
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium/Low: Next release cycle

### Security Best Practices for Users

#### API Key Security
- **Never commit API keys** to version control
- **Use environment variables** for API keys
- **Rotate keys regularly**
- **Use least-privilege access** when possible

#### Safe Usage
```bash
# ✅ Good - API key in environment
export OPENAI_API_KEY="your-key"
zapgpt "Your query"

# ❌ Bad - API key in command (visible in history)
zapgpt --api-key "your-key" "Your query"
```

#### Configuration Security
- ZapGPT stores configuration in `~/.config/zapgpt/`
- **Protect your config directory** with appropriate file permissions
- **Review custom prompts** before using them
- **Be cautious with pricing data** modifications

#### Network Security
- ZapGPT makes HTTPS requests to LLM providers
- **Verify SSL certificates** are properly validated
- **Use trusted networks** when possible
- **Monitor API usage** for unexpected activity

### Known Security Considerations

#### API Key Handling
- API keys are stored in memory during execution
- Keys are passed to provider clients securely
- No API keys are logged or stored persistently

#### User Input
- User prompts are sent directly to LLM providers
- **Be mindful of sensitive information** in prompts
- ZapGPT does not modify or log user prompts

#### Configuration Files
- Configuration files are stored in user's home directory
- **Ensure proper file permissions** on config directory
- Custom prompts and pricing data should be reviewed

### Security Features

#### Environment Variable Validation
- Only required API keys are validated
- Clear error messages for missing keys
- No fallback to insecure defaults

#### Configuration Isolation
- User configuration isolated from package
- No system-wide configuration files
- Clean separation of user data

#### Provider Security
- HTTPS-only communication with all providers
- Proper error handling for API failures
- No credential caching or persistence

### Vulnerability Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 1-2**: Acknowledgment sent
3. **Day 3-7**: Initial assessment and triage
4. **Day 8-30**: Fix development and testing
5. **Day 31**: Public disclosure and release

### Security Updates

Security updates will be:
- **Released immediately** for critical vulnerabilities
- **Announced** in release notes and README
- **Tagged** with security labels in GitHub releases
- **Documented** in CHANGELOG.md

### Contact

For security-related questions or concerns:
- **Email**: [your-email@domain.com]
- **Subject**: "ZapGPT Security"

For general questions, please use GitHub issues.

---

Thank you for helping keep ZapGPT secure! 🔒
