# ZapGPT + Burp Suite Integration Guide

This guide shows how to integrate ZapGPT with Burp Suite to enhance vulnerability analysis using AI-powered insights.

## Overview

ZapGPT can be integrated with Burp Suite to:
- Analyze Burp scan results with AI
- Generate detailed remediation guidance
- Create executive summaries from technical findings
- Automate vulnerability triage and prioritization
- Enhance security reporting with contextual insights

## Integration Methods

### 1. Burp Suite Extension Development

Create a Burp extension that calls ZapGPT for vulnerability analysis:

```java
// BurpAIAnalyzer.java
public class BurpAIAnalyzer implements BurpExtension, IScannerCheck {
    private IBurpExtenderCallbacks callbacks;
    private IExtensionHelpers helpers;

    @Override
    public void registerExtenderCallbacks(IBurpExtenderCallbacks callbacks) {
        this.callbacks = callbacks;
        this.helpers = callbacks.getHelpers();

        callbacks.setExtensionName("AI Vulnerability Analyzer");
        callbacks.registerScannerCheck(this);
    }

    @Override
    public List<IScanIssue> doActiveScan(IHttpRequestResponse baseRequestResponse,
                                        IScannerInsertionPoint insertionPoint) {
        // Analyze findings with ZapGPT
        String analysis = analyzeWithAI(baseRequestResponse);

        // Create enhanced scan issue with AI insights
        return Arrays.asList(new AIEnhancedScanIssue(baseRequestResponse, analysis));
    }

    private String analyzeWithAI(IHttpRequestResponse requestResponse) {
        try {
            String prompt = buildAnalysisPrompt(requestResponse);
            ProcessBuilder pb = new ProcessBuilder("zapgpt", "-q", prompt);
            Process process = pb.start();
            return readProcessOutput(process);
        } catch (Exception e) {
            return "AI analysis failed: " + e.getMessage();
        }
    }
}
```

### 2. Post-Scan Analysis

Export Burp results and analyze with ZapGPT:

```bash
#!/bin/bash
# burp-ai-analysis.sh

# Export Burp scan results (XML format)
BURP_REPORT="$1"
if [ -z "$BURP_REPORT" ]; then
    echo "Usage: $0 <burp-report.xml>"
    exit 1
fi

echo "Analyzing Burp Suite findings with AI..."

# Convert XML to readable format and analyze
zapgpt -q "Analyze this Burp Suite security report and provide prioritized remediation recommendations: $(cat $BURP_REPORT)" > ai-analysis.md

# Generate executive summary
zapgpt -q "Create an executive summary of this Burp Suite security assessment for management: $(cat $BURP_REPORT)" > executive-summary.md

# Create developer checklist
zapgpt -q "Create a developer remediation checklist based on these Burp findings: $(cat $BURP_REPORT)" > remediation-checklist.md

echo "Analysis complete! Generated files:"
echo "- ai-analysis.md"
echo "- executive-summary.md"
echo "- remediation-checklist.md"
```

### 3. Burp Collaborator Integration

Monitor Burp Collaborator interactions with AI analysis:

```python
#!/usr/bin/env python3
# burp-collaborator-ai.py

import requests
import subprocess
import json
import time
import base64

class BurpCollaboratorAI:
    def __init__(self, collaborator_server, polling_interval=30):
        self.server = collaborator_server
        self.interval = polling_interval
        self.seen_interactions = set()

    def get_interactions(self):
        """Fetch new Collaborator interactions"""
        try:
            # This would integrate with Burp's Collaborator API
            # Implementation depends on your Burp setup
            response = requests.get(f"{self.server}/interactions")
            return response.json().get('interactions', [])
        except Exception as e:
            print(f"Error fetching interactions: {e}")
            return []

    def analyze_interaction(self, interaction):
        """Analyze interaction with AI"""
        prompt = f"""
        Analyze this Burp Collaborator interaction:
        - Type: {interaction.get('type')}
        - Client IP: {interaction.get('client_ip')}
        - Query: {interaction.get('query')}
        - Time: {interaction.get('time')}

        Determine:
        1. Vulnerability type and severity
        2. Potential attack vectors
        3. Remediation steps
        4. Business impact
        """

        try:
            result = subprocess.run(['zapgpt', '-q', prompt],
                                  capture_output=True, text=True, timeout=30)
            return result.stdout.strip()
        except Exception as e:
            return f"AI analysis failed: {e}"

    def monitor(self):
        """Main monitoring loop"""
        print("Starting Burp Collaborator AI monitoring...")

        while True:
            try:
                interactions = self.get_interactions()

                for interaction in interactions:
                    interaction_id = interaction.get('id')
                    if interaction_id not in self.seen_interactions:
                        print(f"New interaction detected: {interaction_id}")
                        analysis = self.analyze_interaction(interaction)
                        print(f"AI Analysis:\n{analysis}\n")
                        self.seen_interactions.add(interaction_id)

                time.sleep(self.interval)

            except KeyboardInterrupt:
                print("Monitoring stopped")
                break

if __name__ == "__main__":
    monitor = BurpCollaboratorAI("http://your-collaborator-server")
    monitor.monitor()
```

### 4. Burp Suite Professional API Integration

Use Burp's REST API with ZapGPT:

```python
#!/usr/bin/env python3
# burp-api-ai.py

import requests
import subprocess
import json
import time

class BurpAPIAI:
    def __init__(self, burp_url="http://localhost:1337", api_key=None):
        self.burp_url = burp_url
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def get_scan_issues(self, task_id=None):
        """Get scan issues from Burp"""
        url = f"{self.burp_url}/v0.1/scan"
        if task_id:
            url += f"/{task_id}/issues"
        else:
            url += "/issues"

        response = requests.get(url, headers=self.headers)
        return response.json().get('issues', [])

    def analyze_issue_with_ai(self, issue):
        """Analyze single issue with AI"""
        prompt = f"""
        Analyze this Burp Suite finding:
        - Issue Type: {issue.get('issue_type')}
        - Severity: {issue.get('severity')}
        - Confidence: {issue.get('confidence')}
        - URL: {issue.get('origin')}
        - Description: {issue.get('issue_detail')}

        Provide:
        1. Detailed technical explanation
        2. Exploitation scenarios
        3. Business impact assessment
        4. Step-by-step remediation
        """

        try:
            result = subprocess.run(['zapgpt', '-q', prompt],
                                  capture_output=True, text=True, timeout=45)
            return result.stdout.strip()
        except Exception as e:
            return f"AI analysis failed: {e}"

    def generate_enhanced_report(self, output_file="enhanced-burp-report.md"):
        """Generate AI-enhanced report"""
        issues = self.get_scan_issues()

        with open(output_file, 'w') as f:
            f.write("# AI-Enhanced Burp Suite Security Report\n\n")

            # Executive summary
            summary_prompt = f"Create an executive summary of these {len(issues)} security findings: {json.dumps(issues[:5])}"  # Sample for summary
            summary = subprocess.run(['zapgpt', '-q', summary_prompt],
                                   capture_output=True, text=True).stdout
            f.write("## Executive Summary\n\n")
            f.write(summary + "\n\n")

            # Detailed analysis
            f.write("## Detailed Findings\n\n")
            for i, issue in enumerate(issues, 1):
                f.write(f"### Finding {i}: {issue.get('issue_type')}\n\n")
                f.write(f"**Severity:** {issue.get('severity')}\n")
                f.write(f"**URL:** {issue.get('origin')}\n\n")

                # AI analysis
                analysis = self.analyze_issue_with_ai(issue)
                f.write("**AI Analysis:**\n")
                f.write(analysis + "\n\n")
                f.write("---\n\n")

        print(f"Enhanced report generated: {output_file}")

if __name__ == "__main__":
    analyzer = BurpAPIAI()
    analyzer.generate_enhanced_report()
```

### 5. Intruder Payload Analysis

Analyze Intruder attack results with AI:

```bash
#!/bin/bash
# burp-intruder-ai.sh

# Export Intruder results to CSV
INTRUDER_CSV="$1"
if [ -z "$INTRUDER_CSV" ]; then
    echo "Usage: $0 <intruder-results.csv>"
    exit 1
fi

echo "Analyzing Burp Intruder results with AI..."

# Analyze successful payloads
zapgpt -q "Analyze these Burp Intruder attack results and identify successful payloads and attack patterns: $(cat $INTRUDER_CSV)" > intruder-analysis.md

# Generate payload recommendations
zapgpt -q "Based on these Intruder results, suggest additional payloads and attack vectors to test: $(cat $INTRUDER_CSV)" > payload-recommendations.md

echo "Analysis complete!"
```

### 6. Repeater Request Analysis

Analyze individual requests in Repeater:

```python
#!/usr/bin/env python3
# burp-repeater-ai.py

import sys
import subprocess

def analyze_request_response(request_file, response_file):
    """Analyze HTTP request/response pair"""

    with open(request_file, 'r') as f:
        request_data = f.read()

    with open(response_file, 'r') as f:
        response_data = f.read()

    prompt = f"""
    Analyze this HTTP request/response pair for security issues:

    REQUEST:
    {request_data}

    RESPONSE:
    {response_data}

    Identify:
    1. Potential vulnerabilities
    2. Security headers analysis
    3. Input validation issues
    4. Authentication/authorization flaws
    5. Recommended security improvements
    """

    try:
        result = subprocess.run(['zapgpt', '-q', prompt],
                              capture_output=True, text=True, timeout=60)
        return result.stdout.strip()
    except Exception as e:
        return f"Analysis failed: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 burp-repeater-ai.py <request.txt> <response.txt>")
        sys.exit(1)

    analysis = analyze_request_response(sys.argv[1], sys.argv[2])
    print("AI Analysis:")
    print("=" * 50)
    print(analysis)
```

## Practical Examples

### Example 1: Automated Burp Report Enhancement

```bash
#!/bin/bash
# enhance-burp-report.sh

BURP_XML="$1"
OUTPUT_DIR="enhanced-report"

mkdir -p "$OUTPUT_DIR"

# Generate different types of analysis
zapgpt -q "Create a technical security assessment summary from this Burp report: $(cat $BURP_XML)" > "$OUTPUT_DIR/technical-summary.md"

zapgpt -q "Create an executive briefing for management from this Burp security scan: $(cat $BURP_XML)" > "$OUTPUT_DIR/executive-briefing.md"

zapgpt -q "Generate a developer action plan with specific remediation steps: $(cat $BURP_XML)" > "$OUTPUT_DIR/developer-action-plan.md"

zapgpt -q "Map these findings to OWASP Top 10 and provide compliance guidance: $(cat $BURP_XML)" > "$OUTPUT_DIR/compliance-mapping.md"

echo "Enhanced reports generated in $OUTPUT_DIR/"
```

### Example 2: Real-time Burp Monitoring

```python
#!/usr/bin/env python3
# burp-realtime-monitor.py

import time
import subprocess
import json
from datetime import datetime

class BurpRealtimeMonitor:
    def __init__(self, log_file="/tmp/burp-monitor.log"):
        self.log_file = log_file
        self.processed_issues = set()

    def log_analysis(self, issue_type, analysis):
        """Log AI analysis results"""
        timestamp = datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {issue_type}\n")
            f.write(f"{analysis}\n")
            f.write("-" * 80 + "\n")

    def quick_analyze(self, issue_summary):
        """Quick AI analysis for real-time monitoring"""
        prompt = f"Quickly assess this security finding and provide severity and immediate actions: {issue_summary}"

        try:
            result = subprocess.run(['zapgpt', '-q', prompt],
                                  capture_output=True, text=True, timeout=20)
            return result.stdout.strip()
        except Exception as e:
            return f"Quick analysis failed: {e}"

    def monitor_new_issues(self):
        """Monitor for new issues (implement based on your Burp setup)"""
        print("Starting real-time Burp monitoring...")

        while True:
            try:
                # This would integrate with your Burp monitoring mechanism
                # Could be file watching, API polling, etc.

                # Placeholder for new issue detection
                time.sleep(10)

            except KeyboardInterrupt:
                print("Monitoring stopped")
                break

if __name__ == "__main__":
    monitor = BurpRealtimeMonitor()
    monitor.monitor_new_issues()
```

## Benefits of Integration

1. **Enhanced Vulnerability Analysis**: AI provides detailed context for Burp findings
2. **Automated Triage**: Prioritize vulnerabilities based on business impact
3. **Improved Reporting**: Generate executive and technical reports automatically
4. **Learning Tool**: Understand complex vulnerabilities better
5. **Time Savings**: Reduce manual analysis effort
6. **Consistency**: Standardized analysis approach
7. **Custom Insights**: Tailored analysis based on your application context

## Getting Started

### Prerequisites

1. **Install ZapGPT**:
   ```bash
   pip install zapgpt
   ```

2. **Burp Suite Setup**:
   - Burp Suite Professional (for API access)
   - Enable REST API in Burp settings
   - Configure API key if required

3. **Set up API keys**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

### Quick Start

1. **Export Burp scan results** (XML format)

2. **Run basic analysis**:
   ```bash
   zapgpt -q "Analyze this Burp Suite report: $(cat burp-report.xml)"
   ```

3. **Use enhancement script**:
   ```bash
   ./enhance-burp-report.sh burp-report.xml
   ```

## Best Practices

1. **Use Quiet Mode**: Use `zapgpt -q` for automation-friendly output
2. **Structured Analysis**: Create consistent prompt templates
3. **Rate Limiting**: Respect API limits when processing many findings
4. **Error Handling**: Implement robust error handling
5. **Security**: Protect API keys and sensitive data
6. **Testing**: Thoroughly test integrations before production use

## Advanced Use Cases

### Custom Vulnerability Scoring
```bash
zapgpt "Score these Burp findings 1-10 for a financial application: $(cat burp-report.xml)"
```

### Attack Chain Analysis
```bash
zapgpt "Identify potential attack chains from these Burp findings: $(cat burp-report.xml)"
```

### Compliance Mapping
```bash
zapgpt "Map these Burp findings to PCI DSS requirements: $(cat burp-report.xml)"
```

## Troubleshooting

### Common Issues

1. **Burp API not accessible**: Check API configuration and firewall
2. **Large XML files**: Process in chunks or filter results
3. **Timeout errors**: Increase timeout values for complex analysis
4. **Memory issues**: Use streaming for large datasets

## Resources

- [Burp Suite Documentation](https://portswigger.net/burp/documentation)
- [Burp Extensions API](https://portswigger.net/burp/extender)
- [ZapGPT Documentation](../README.md)
- [Burp Suite Professional](https://portswigger.net/burp/pro)
