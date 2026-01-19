# Security Patterns for Kubetorch Integration

This document outlines security best practices implemented in the GenOps Kubetorch integration to ensure safe and secure operation.

## Subprocess Usage

All subprocess calls use absolute executable paths resolved via `shutil.which()` to prevent arbitrary command execution and path injection attacks.

### Pattern

```python
import shutil
import subprocess  # nosec B404 - subprocess required for CLI validation

# Resolve absolute path to executable
kubectl_path = shutil.which('kubectl')
if not kubectl_path:
    # Handle missing executable gracefully
    return

# Use absolute path with explicit security settings
subprocess.run(
    [kubectl_path, 'version', '--client'],  # nosec B607 - validated absolute path
    capture_output=True,
    check=True,
    timeout=5,
    shell=False  # Explicit shell=False prevents shell injection
)
```

### Rationale

- **`shutil.which()`**: Securely resolves partial paths to absolute paths
- **Absolute paths**: Prevents PATH manipulation attacks
- **`shell=False`**: Prevents shell injection vulnerabilities
- **Path validation**: Checks executable exists before execution
- **Timeout**: Prevents indefinite blocking on subprocess calls

### Bandit Suppressions

- **B404**: subprocess module import - Required for CLI tool validation in setup checks
- **B607**: Partial executable path - Mitigated by using `shutil.which()` for absolute path resolution

## Random Number Generation

The Kubetorch integration does **not** use the `random` module. All randomness requirements (if any) should follow these guidelines:

### Non-Cryptographic Use (Sampling, Statistics)

For telemetry sampling or non-security-critical randomness:

```python
import random  # nosec B311 - using for sampling rates, not cryptography

# Sampling rate logic (non-security-critical)
if random.random() < sampling_rate:  # nosec B311
    # Track telemetry
    pass
```

### Cryptographic Use (Tokens, Keys, Secrets)

For security-sensitive operations, **always use the `secrets` module**:

```python
import secrets

# Generate cryptographically secure random values
token = secrets.token_urlsafe(32)
random_int = secrets.randbelow(100)
```

### When to Use Each

- **`random` module**: Sampling rates, load balancing, non-security telemetry decisions
- **`secrets` module**: Authentication tokens, API keys, session IDs, encryption keys

## Security Review Checklist

Before adding new functionality, verify:

- [ ] All subprocess calls use absolute paths via `shutil.which()`
- [ ] All subprocess calls have `shell=False` explicitly set
- [ ] All subprocess calls have appropriate timeouts
- [ ] Cryptographic operations use `secrets` module, not `random`
- [ ] All `# nosec` comments include justification
- [ ] Error handling doesn't leak sensitive information
- [ ] File operations validate paths and permissions
- [ ] External data is validated before processing

## Bandit Configuration

This integration passes Bandit security scanning with the following justified suppressions:

```python
# B404: subprocess import - Required for validation
import subprocess  # nosec B404

# B607: Partial path - Resolved to absolute path
subprocess.run([abs_path, ...])  # nosec B607
```

## Reporting Security Issues

If you discover a security vulnerability in the Kubetorch integration:

1. **Do not** create a public GitHub issue
2. Email security@genops.ai with details
3. Include reproduction steps and potential impact
4. Allow reasonable time for patching before disclosure

## Security Testing

Run security scans before committing:

```bash
# Bandit security scan
bandit -r src/genops/providers/kubetorch/ -ll

# Check for high/medium severity issues (should be 0)
bandit -r src/genops/providers/kubetorch/ -f json -o bandit-report.json
```

## References

- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE-78: OS Command Injection](https://cwe.mitre.org/data/definitions/78.html)
- [CWE-330: Insufficient Randomness](https://cwe.mitre.org/data/definitions/330.html)
