# Security Policy

## Supported Versions

We take security seriously and provide security updates for the following versions of GenOps AI:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Security Considerations

GenOps AI is designed with security in mind, particularly since it handles sensitive AI telemetry data:

### **Data Handling**
- **No API Keys Stored**: GenOps AI never stores or logs API keys from AI providers
- **Telemetry Only**: Only telemetry metadata (costs, tokens, models) is captured, never prompt/response content by default
- **Configurable Redaction**: Built-in support for redacting sensitive information in telemetry
- **Local Processing**: All governance decisions are made locally, no data sent to external services

### **OpenTelemetry Security**
- **Standard Compliance**: Follows OpenTelemetry security best practices
- **Transport Security**: OTLP exports use TLS by default
- **Authentication**: Supports standard OTLP authentication headers
- **Sampling**: Configurable sampling reduces data exposure

### **Provider Security**
- **Graceful Failures**: Provider failures don't expose sensitive information
- **Timeout Handling**: Proper timeout handling prevents hanging connections
- **Error Sanitization**: Error messages are sanitized to prevent information leakage

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them responsibly by:

**🔒 [Creating a private security issue](https://github.com/KoshiHQ/GenOps-AI/security/advisories/new)** or using GitHub Issues

Please include the following information:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths** of source file(s) related to the manifestation of the issue
- **Location** of the affected source code (tag/branch/commit or direct URL)
- **Special configuration** required to reproduce the issue
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact** of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

## Response Process

1. **Acknowledgment**: We'll acknowledge receipt of your vulnerability report within 48 hours
2. **Investigation**: We'll investigate and validate the issue within 5 business days
3. **Fix Development**: We'll develop a fix and coordinate disclosure timeline
4. **Release**: We'll release a security update and publish a security advisory
5. **Recognition**: We'll publicly recognize your contribution (if desired)

## Security Updates

Security updates will be:

- **Released promptly** for critical vulnerabilities
- **Announced** through GitHub security advisories
- **Documented** in release notes with severity information
- **Backported** to supported versions when applicable

## Bug Bounty

We don't currently offer a formal bug bounty program, but we're grateful for security research and will:

- **Publicly recognize** responsible disclosure (with permission)
- **Provide attribution** in security advisories and release notes
- **Consider** offering swag or other recognition for significant contributions

## Best Practices for Users

### **Deployment Security**

1. **API Key Management**
   - Store AI provider API keys securely (environment variables, key vaults)
   - Rotate API keys regularly
   - Use least-privilege access for API keys
   - Never commit API keys to version control

2. **Network Security**
   - Use TLS for all OTLP exports
   - Restrict network access to telemetry endpoints
   - Use VPNs or private networks for sensitive deployments
   - Validate OTLP endpoint certificates

3. **Access Control**
   - Limit access to GenOps configuration
   - Use proper authentication for observability platforms
   - Implement role-based access control where possible
   - Audit access to governance data

### **Configuration Security**

1. **Sensitive Data Protection**
   ```python
   # Configure redaction for sensitive content
   genops.init(
       redact_patterns=["password", "ssn", "credit_card"],
       redact_user_content=True,  # Redact user prompts
       max_content_length=100     # Limit content capture
   )
   ```

2. **Sampling Configuration**
   ```python
   # Use sampling in production to limit data exposure
   genops.init(
       sampling_rate=0.1,  # Sample 10% of requests
       sensitive_operations_only=False  # Don't sample sensitive ops
   )
   ```

3. **OTLP Security**
   ```python
   # Always use TLS and authentication
   genops.init(
       exporter_type="otlp",
       otlp_endpoint="https://secure-endpoint.com",  # HTTPS only
       otlp_headers={
           "Authorization": "Bearer your-secure-token",
           "X-Custom-Auth": "your-auth-header"
       }
   )
   ```

### **Development Security**

1. **Dependency Management**
   - Keep GenOps AI updated to the latest version
   - Regularly update AI provider SDKs
   - Use dependency scanning tools
   - Pin dependency versions in production

2. **Testing Security**
   - Use mock providers in tests (never real API keys)
   - Test error handling paths
   - Validate input sanitization
   - Test timeout and failure scenarios

3. **Code Reviews**
   - Review GenOps configurations for sensitive data exposure
   - Validate OTLP endpoint security
   - Check for hardcoded credentials
   - Ensure proper error handling

## Security Architecture

### **Data Flow Security**

```
AI Application
    ↓ (telemetry metadata only)
GenOps AI SDK
    ↓ (TLS/authenticated)
OTLP Exporter
    ↓ (TLS/authenticated)
Observability Platform
```

### **Threat Model**

GenOps AI protects against:

- **API Key Exposure**: Never logs or stores provider API keys
- **Content Leakage**: Configurable content redaction and sampling
- **Man-in-the-Middle**: TLS for all external communications
- **Unauthorized Access**: Authentication for OTLP exports
- **Data Injection**: Input validation and sanitization
- **Resource Exhaustion**: Timeouts and circuit breakers

## Compliance Considerations

GenOps AI supports compliance requirements:

- **GDPR**: Data minimization and configurable data retention
- **HIPAA**: Healthcare data protection through redaction
- **SOC 2**: Audit logging and access controls
- **PCI DSS**: Credit card data redaction
- **Custom**: Configurable data governance policies

For specific compliance questions, create a [GitHub Issue](https://github.com/KoshiHQ/GenOps-AI/issues) with the "compliance" label.

## Contact

For security questions or concerns:

- **Security Issues**: [GitHub Security Tab](https://github.com/KoshiHQ/GenOps-AI/security/advisories/new)
- **Compliance Questions**: [GitHub Issues with compliance label](https://github.com/KoshiHQ/GenOps-AI/issues/new?labels=compliance)
- **General Security**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)

---

**Thank you for helping keep GenOps AI and our community safe!** 🔒