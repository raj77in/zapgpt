# ZapGPT + OWASP ZAP Integration Guide

This guide shows how to integrate ZapGPT with OWASP ZAP to enhance vulnerability analysis using AI-powered insights.

## Overview

ZapGPT can be integrated with OWASP ZAP to:
- Provide AI-powered analysis of security vulnerabilities
- Generate detailed remediation guidance
- Create executive summaries for management
- Automate vulnerability triage and prioritization
- Enhance security reporting with contextual insights

## Integration Methods

### 1. Post-Scan Analysis with ZapGPT

Use ZapGPT to analyze ZAP's findings and get AI-powered insights:

```bash
# Run ZAP scan and export results
zap-cli quick-scan --self-contained --start-options '-config api.disablekey=true' http://example.com
zap-cli report -o zap-report.json -f json

# Use ZapGPT to analyze the vulnerabilities
zapgpt -q "Analyze this OWASP ZAP security report and provide prioritized remediation recommendations: $(cat zap-report.json)"
```

### 2. Custom ZAP Script Integration

Create a ZAP script that calls ZapGPT for each vulnerability:

```python
# ZAP Script (Python)
import subprocess
import json

def analyze_vulnerability_with_ai(vuln_data):
    prompt = f"""
    Analyze this security vulnerability:
    - Name: {vuln_data['name']}
    - Risk: {vuln_data['risk']}
    - Description: {vuln_data['description']}
    - URL: {vuln_data['url']}

    Provide:
    1. Detailed explanation
    2. Exploitation scenarios
    3. Specific remediation steps
    4. Code examples if applicable
    """

    result = subprocess.run([
        'zapgpt', '-q', prompt
    ], capture_output=True, text=True)

    return result.stdout

# Use in ZAP passive/active scan rules
```

### 3. ZAP Automation Framework Integration

Integrate ZapGPT into ZAP's automation framework:

```yaml
# automation.yaml
env:
  contexts:
    - name: "AI-Enhanced Scan"
      urls:
        - "http://example.com"

jobs:
  - type: spider
    parameters:
      context: "AI-Enhanced Scan"

  - type: activeScan
    parameters:
      context: "AI-Enhanced Scan"

  - type: report
    parameters:
      template: "traditional-json"
      reportFile: "/tmp/zap-report.json"

  # Custom job to analyze with AI
  - type: script
    parameters:
      action: "analyze-with-ai"
      script: |
        import subprocess
        result = subprocess.run(['zapgpt', '-q', 'Analyze ZAP findings: ' + open('/tmp/zap-report.json').read()])
```

### 4. Real-time Vulnerability Triage

Create a monitoring script that processes ZAP alerts in real-time:

```bash
#!/bin/bash
# zap-ai-monitor.sh

# Monitor ZAP alerts and analyze with AI
while true; do
    # Get latest alerts from ZAP API
    NEW_ALERTS=$(curl -s "http://localhost:8080/JSON/core/view/alerts/" | jq '.alerts[] | select(.id > '${LAST_ID:-0}')')

    if [ ! -z "$NEW_ALERTS" ]; then
        echo "$NEW_ALERTS" | while read alert; do
            # Analyze each new alert with ZapGPT
            ANALYSIS=$(echo "Analyze this security alert and provide remediation guidance: $alert" | zapgpt -q)

            # Send to your notification system
            echo "AI Analysis: $ANALYSIS" | notify-send "ZAP Alert Analysis"
        done
    fi

    sleep 30
done
```

### 5. Enhanced Reporting

Generate AI-enhanced security reports:

```bash
#!/bin/bash
# enhanced-zap-report.sh

# Generate standard ZAP report
zap-cli report -o raw-report.json -f json

# Create AI-enhanced executive summary
zapgpt -q "Create an executive summary of this security assessment for management, including business impact and prioritized recommendations: $(cat raw-report.json)" > executive-summary.md

# Generate technical remediation guide
zapgpt -q "Create a detailed technical remediation guide for developers based on these ZAP findings: $(cat raw-report.json)" > technical-guide.md

# Combine into final report
echo "# Security Assessment Report" > final-report.md
echo "" >> final-report.md
cat executive-summary.md >> final-report.md
echo "" >> final-report.md
echo "## Technical Details" >> final-report.md
cat technical-guide.md >> final-report.md
```

### 6. ZAP Add-on Development

You could develop a ZAP add-on that integrates ZapGPT:

```java
// ZAP Add-on (Java)
public class AIAnalysisExtension extends ExtensionAdaptor {

    public void analyzeAlert(Alert alert) {
        String prompt = String.format(
            "Analyze vulnerability: %s\nRisk: %s\nURL: %s\nDescription: %s",
            alert.getName(), alert.getRisk(), alert.getUri(), alert.getDescription()
        );

        ProcessBuilder pb = new ProcessBuilder("zapgpt", "-q", prompt);
        Process process = pb.start();
        String aiAnalysis = readProcessOutput(process);

        // Add AI analysis to alert or create new alert with enhanced info
        alert.setOtherInfo(alert.getOtherInfo() + "\n\nAI Analysis:\n" + aiAnalysis);
    }
}
```

### 7. Vulnerability Research Mode

Use ZapGPT for deep vulnerability research:

```bash
# Research specific vulnerability types
zapgpt "Explain SQL injection vulnerabilities in detail, including advanced techniques, bypass methods, and comprehensive prevention strategies"

# Analyze custom payloads
zapgpt "Analyze this XSS payload and suggest variations: <script>alert(1)</script>"

# Generate test cases
zapgpt "Generate comprehensive test cases for testing authentication bypass vulnerabilities"
```

## Practical Examples

### Example 1: Automated Vulnerability Analysis

```bash
#!/bin/bash
# auto-analyze.sh

TARGET_URL="$1"
if [ -z "$TARGET_URL" ]; then
    echo "Usage: $0 <target-url>"
    exit 1
fi

echo "Starting automated security analysis for: $TARGET_URL"

# Step 1: Run ZAP baseline scan
echo "Running ZAP baseline scan..."
zap-baseline.py -t "$TARGET_URL" -J zap-baseline.json

# Step 2: Analyze findings with AI
echo "Analyzing findings with AI..."
zapgpt -q "Analyze these ZAP security findings and provide a prioritized action plan: $(cat zap-baseline.json)" > ai-analysis.md

# Step 3: Generate executive summary
echo "Generating executive summary..."
zapgpt -q "Create a 2-paragraph executive summary of this security assessment for business stakeholders: $(cat zap-baseline.json)" > executive-summary.txt

# Step 4: Create remediation checklist
echo "Creating remediation checklist..."
zapgpt -q "Create a developer checklist for fixing these security issues: $(cat zap-baseline.json)" > remediation-checklist.md

echo "Analysis complete! Check the generated files:"
echo "- ai-analysis.md: Detailed AI analysis"
echo "- executive-summary.txt: Business summary"
echo "- remediation-checklist.md: Developer action items"
```

### Example 2: Continuous Security Monitoring

```python
#!/usr/bin/env python3
# zap-ai-monitor.py

import requests
import subprocess
import json
import time
import logging

class ZAPAIMonitor:
    def __init__(self, zap_url="http://localhost:8080", check_interval=60):
        self.zap_url = zap_url
        self.check_interval = check_interval
        self.last_alert_id = 0

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_new_alerts(self):
        """Fetch new alerts from ZAP API"""
        try:
            response = requests.get(f"{self.zap_url}/JSON/core/view/alerts/")
            alerts = response.json()['alerts']

            new_alerts = [alert for alert in alerts if int(alert['id']) > self.last_alert_id]

            if new_alerts:
                self.last_alert_id = max(int(alert['id']) for alert in new_alerts)

            return new_alerts
        except Exception as e:
            self.logger.error(f"Error fetching alerts: {e}")
            return []

    def analyze_with_ai(self, alert):
        """Analyze alert with ZapGPT"""
        prompt = f"""
        Analyze this security alert:
        - Alert: {alert['alert']}
        - Risk: {alert['risk']}
        - Confidence: {alert['confidence']}
        - URL: {alert['url']}
        - Description: {alert['description']}

        Provide:
        1. Severity assessment
        2. Potential impact
        3. Immediate actions needed
        4. Long-term remediation
        """

        try:
            result = subprocess.run(['zapgpt', '-q', prompt],
                                  capture_output=True, text=True, timeout=30)
            return result.stdout.strip()
        except Exception as e:
            self.logger.error(f"Error analyzing with AI: {e}")
            return "AI analysis failed"

    def process_alert(self, alert):
        """Process a single alert"""
        self.logger.info(f"Processing alert: {alert['alert']} (Risk: {alert['risk']})")

        ai_analysis = self.analyze_with_ai(alert)

        # Log the analysis
        self.logger.info(f"AI Analysis for {alert['alert']}:\n{ai_analysis}")

        # You can add integrations here:
        # - Send to Slack/Teams
        # - Create JIRA tickets
        # - Update security dashboard
        # - Send email notifications

        return ai_analysis

    def run(self):
        """Main monitoring loop"""
        self.logger.info("Starting ZAP AI Monitor...")

        while True:
            try:
                new_alerts = self.get_new_alerts()

                for alert in new_alerts:
                    self.process_alert(alert)

                if new_alerts:
                    self.logger.info(f"Processed {len(new_alerts)} new alerts")

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

if __name__ == "__main__":
    monitor = ZAPAIMonitor()
    monitor.run()
```

## Benefits of Integration

1. **Enhanced Analysis**: AI provides context and detailed explanations for vulnerabilities
2. **Prioritization**: AI can help prioritize fixes based on business context
3. **Remediation Guidance**: Specific, actionable remediation steps
4. **Executive Reporting**: AI can translate technical findings into business language
5. **Learning Tool**: Helps security teams understand vulnerabilities better
6. **Automation**: Reduces manual analysis time
7. **Consistency**: Standardized analysis approach across all findings
8. **Speed**: Rapid analysis of large numbers of vulnerabilities

## Getting Started

### Prerequisites

1. **Install ZapGPT**:
   ```bash
   pip install zapgpt
   ```

2. **Install OWASP ZAP**:
   - Download from [official website](https://www.zaproxy.org/download/)
   - Or use Docker: `docker run -p 8080:8080 owasp/zap2docker-stable`

3. **Set up API keys**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   # or other supported providers
   ```

### Quick Start

1. **Test basic integration**:
   ```bash
   # Run a simple ZAP scan
   zap-baseline.py -t http://example.com -J zap-report.json

   # Analyze with ZapGPT
   zapgpt -q "Summarize the key security findings: $(cat zap-report.json)"
   ```

2. **Try the automated analysis script**:
   ```bash
   chmod +x auto-analyze.sh
   ./auto-analyze.sh http://your-target.com
   ```

3. **Set up continuous monitoring**:
   ```bash
   python3 zap-ai-monitor.py
   ```

## Best Practices

1. **Use Quiet Mode**: Use `zapgpt -q` for clean output suitable for automation
2. **Structured Prompts**: Create consistent prompt templates for different analysis types
3. **Rate Limiting**: Be mindful of API rate limits when processing many alerts
4. **Error Handling**: Implement proper error handling for production use
5. **Security**: Keep API keys secure and use environment variables
6. **Logging**: Implement comprehensive logging for troubleshooting
7. **Testing**: Test integrations thoroughly before production deployment

## Troubleshooting

### Common Issues

1. **ZAP API not accessible**:
   - Ensure ZAP is running with API enabled
   - Check firewall settings
   - Verify API key configuration

2. **ZapGPT not found**:
   - Ensure ZapGPT is installed and in PATH
   - Check virtual environment activation

3. **API rate limits**:
   - Implement delays between requests
   - Use batch processing for large datasets
   - Consider using multiple API keys

4. **Large JSON files**:
   - Filter reports to relevant findings only
   - Process in chunks for very large reports
   - Use file-based processing for memory efficiency

## Advanced Use Cases

### Custom Vulnerability Scoring

```bash
# Create custom risk scoring based on business context
zapgpt "Analyze these vulnerabilities and score them 1-10 based on impact to an e-commerce platform: $(cat zap-report.json)"
```

### Compliance Mapping

```bash
# Map findings to compliance frameworks
zapgpt "Map these security findings to OWASP Top 10 and PCI DSS requirements: $(cat zap-report.json)"
```

### Threat Modeling

```bash
# Generate threat models based on findings
zapgpt "Based on these ZAP findings, create a threat model for this application including attack vectors and mitigations: $(cat zap-report.json)"
```

## Contributing

If you develop useful ZAP + ZapGPT integrations, please consider:
1. Sharing scripts and examples
2. Contributing to the documentation
3. Reporting issues and improvements
4. Creating reusable templates and tools

## Resources

- [OWASP ZAP Documentation](https://www.zaproxy.org/docs/)
- [ZAP API Documentation](https://www.zaproxy.org/docs/api/)
- [ZapGPT Documentation](../README.md)
- [ZAP Automation Framework](https://www.zaproxy.org/docs/automate/)
