# CONTRIBUTING.md

Thank you for your interest in contributing to **GenOps AI** —
the open-source governance framework for AI systems built alongside [OpenLLMetry](https://github.com/traceloop/openllmetry) on [OpenTelemetry](https://opentelemetry.io).

We welcome contributions from developers, researchers, and teams who share our mission to bring visibility, accountability, and governance to AI systems.

## Overview

GenOps AI is part of the **Koshi ecosystem**, maintained by the GenOps AI community and distributed by [KoshiHQ](https://getkoshi.ai).
It extends the OpenTelemetry standard to include AI governance telemetry: cost, policy, compliance, and evaluation semantics.

Our open-core model follows the same proven pattern as:

- **OpenTelemetry** → Honeycomb
- **OpenLLMetry** → Traceloop  
- **GenOps AI** → Koshi

## How to Contribute

There are many ways to contribute:

| Type | Examples |
|------|----------|
| **Code Contributions** | Provider adapters, processors, exporters, SDK improvements |
| **Documentation** | Tutorials, integration guides, API references |
| **Ecosystem Adapters** | Support for new observability or LLM platforms |
| **Community Engagement** | Blog posts, demos, issue triage, or RFC discussions |
| **Governance RFCs** | Propose new telemetry semantics or extensions |

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/KoshiHQ/GenOps-AI.git
   cd GenOps-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -e .[dev]
   ```

3. **Run tests**
   ```bash
   pytest
   ```

4. **Format & lint**
   ```bash
   ruff format
   ruff check
   mypy src/
   ```

5. **Open a Pull Request**  
   PRs should target the `main` branch and follow our [Pull Request Guidelines](#pull-request-guidelines).

## Repository Structure

```
GenOps-AI/
├── src/genops/         # Core telemetry and adapter logic
├── tests/              # Unit and integration tests  
├── examples/           # Example integrations and demos
├── docs/               # Documentation source files
├── processors/         # OTel collector processors
├── exporters/          # OTLP exporters
└── CONTRIBUTING.md     # This file
```

## Pull Request Guidelines

### General Rules

- **Keep PRs focused and atomic** — one purpose per PR.
- **Write clear commit messages** describing your changes.
- **Include or update tests** when modifying functionality.
- **Update documentation and examples** when relevant.

### Review Process

1. A maintainer will triage your PR.
2. Reviews will check code quality, test coverage, and alignment with governance semantics.
3. Approved PRs will be merged into `main`.

For architectural or standards-level changes, please open an RFC first (see below).

## RFCs: Proposing New Standards

GenOps AI evolves through open RFCs, similar to OpenTelemetry's SIG process.

To propose a new governance signal, adapter, or processor:

1. Create a new GitHub issue using the **RFC Proposal** template.
2. Include background, motivation, and an implementation outline.
3. Label it `rfc` and link any relevant discussions.
4. Maintainers and the community will review and provide feedback.
5. Accepted RFCs are merged into `/docs/specs` and released in the next minor version.

## Code Standards

We follow modern Python conventions for quality and maintainability:

| Tool | Purpose |
|------|---------|
| **pytest** | Unit and integration testing |
| **ruff** | Linting and formatting |
| **mypy** | Static type checking |
| **pyproject.toml** | Unified project configuration |
| **90%+ coverage goal** | Required for merges to core modules |

Please ensure all contributions:

- ✅ Use type hints
- ✅ Include docstrings  
- ✅ Pass CI tests
- ✅ Avoid unnecessary external dependencies

## Contribution Focus Areas

Contributors are encouraged to focus on three primary areas:

| Category | Description |
|----------|-------------|
| **LLM Provider Adapters** | Extend `genops.providers.*` for new model APIs (OpenAI, Anthropic, Bedrock, Mistral, Gemini, etc.) |
| **Observability Exporters** | Add OTLP integrations for Datadog, Honeycomb, Grafana Tempo, Splunk, and others |
| **Collector Processors** | Build OTel Collector extensions for cost attribution, redaction, or policy evaluation |

## Community Engagement

- **Participate in discussions** via Issues
- **Provide feedback** on schema evolution and integration design
- **Submit example PRs** to the `examples/` directory
- **Improve or extend documentation**

Top contributors will be recognized in release notes and documentation.

## Governance Model

- **Core Maintainers**: GenOps AI Community
- **Primary Steward**: [KoshiHQ](https://getkoshi.ai)
- **License**: Apache 2.0 (permissive and community-friendly)

Decisions are made transparently via:

- Public GitHub discussions and PRs
- Open RFC proposals and votes
- Regular milestone planning

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).
All contributors are expected to foster a respectful, inclusive, and collaborative environment.

## Attribution

GenOps AI draws inspiration and architectural guidance from:

- [OpenTelemetry](https://opentelemetry.io)
- [OpenLLMetry](https://github.com/traceloop/openllmetry)
- [Vercel / Next.js](https://nextjs.org)
- [Sentry](https://sentry.io)

## Contact

For general inquiries or partnership opportunities:  
**Email**: [hello@genopsai.org](mailto:hello@genopsai.org)

**Maintainers**: GenOps AI Community / [KoshiHQ](https://getkoshi.ai)

---

## TL;DR

**Fork** → **Build** → **Test** → **PR**

- Keep PRs small and focused
- Discuss major changes via RFCs
- Include type hints, docstrings, and tests
- Help us shape the future of AI governance telemetry