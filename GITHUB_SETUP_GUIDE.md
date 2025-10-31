# GitHub Repository Setup Guide

This guide contains all the configurations needed to optimize GenOps AI for community contributions.

## Step 1: Create GitHub Labels

Go to: `https://github.com/KoshiHQ/GenOps-AI/labels`

### Core Labels (Create these first)
```
good first issue - #7057ff - Issues good for newcomers
help wanted - #008672 - Extra attention is needed
documentation - #0075ca - Improvements or additions to documentation
bug - #d73a4a - Something isn't working
enhancement - #a2eeef - New feature or request
ci-fix - #f9d0c4 - CI test fixes needed
```

### Priority Labels
```
priority: high - #b60205 - Critical issues
priority: medium - #fbca04 - Important but not urgent
priority: low - #0e8a16 - Nice to have
```

### Skill Level Labels
```
difficulty: beginner - #c2e0c6 - Good for new contributors
difficulty: intermediate - #bfd4f2 - Requires some experience
difficulty: advanced - #d4c5f9 - Complex changes needed
```

### Category Labels
```
provider - #5319e7 - AI provider integrations
dashboard - #1d76db - Observability dashboards
governance - #b794f6 - AI governance patterns
```

## Step 2: Create Issue Templates

Create folder: `.github/ISSUE_TEMPLATE/`

### File: `.github/ISSUE_TEMPLATE/bug_report.yml`
```yaml
name: 🐛 Bug Report
description: Report a bug or unexpected behavior in GenOps AI
title: "[Bug]: "
labels: ["bug", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report! 🐛
        
        **Quick note:** If you're having trouble with AI costs or governance setup, check our [examples](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples) first.

  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Describe what you were trying to do and what went wrong.
      placeholder: "I was trying to set up cost attribution for my OpenAI calls, but..."
    validations:
      required: true

  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected behavior
      description: What did you expect to happen instead?
      placeholder: "I expected to see cost data in my telemetry..."
    validations:
      required: true

  - type: textarea
    id: reproduction
    attributes:
      label: Steps to reproduce
      description: Please provide step-by-step instructions to reproduce the issue.
      placeholder: |
        1. Install GenOps with `pip install genops[openai]`
        2. Set up instrumentation with `instrument_openai(...)`
        3. Make an API call...
        4. Check telemetry output...
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: |
        Please provide information about your environment.
      value: |
        - GenOps version: 
        - Python version: 
        - AI Provider (OpenAI/Anthropic): 
        - Operating System: 
        - Observability Platform (if applicable): 
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: Relevant logs or error messages
      description: If applicable, add any error messages or logs.
      render: text
    validations:
      required: false

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our [Code of Conduct](https://github.com/KoshiHQ/GenOps-AI/blob/main/CODE_OF_CONDUCT.md)
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

### File: `.github/ISSUE_TEMPLATE/feature_request.yml`
```yaml
name: 🚀 Feature Request
description: Suggest a new feature or enhancement for GenOps AI
title: "[Feature]: "
labels: ["enhancement", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature! 🚀
        
        **Note:** Before requesting a feature, check if it aligns with our [governance focus](https://github.com/KoshiHQ/GenOps-AI/blob/main/README.md#-what-is-genops-ai).

  - type: textarea
    id: problem
    attributes:
      label: Problem or use case
      description: What governance problem would this feature solve?
      placeholder: "As a DevOps engineer, I need to track AI costs per customer because..."
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed solution
      description: How would you like this feature to work?
      placeholder: "I'd like GenOps to automatically tag telemetry with customer_id..."
    validations:
      required: true

  - type: dropdown
    id: category
    attributes:
      label: Feature category
      description: Which area does this feature belong to?
      options:
        - Cost Attribution
        - Policy Enforcement
        - Compliance & Auditing
        - Provider Integration
        - Dashboard & Observability
        - Documentation
        - Other
    validations:
      required: true

  - type: checkboxes
    id: contribution
    attributes:
      label: Contribution
      description: Would you be interested in contributing to this feature?
      options:
        - label: I'd like to work on this feature
        - label: I can help with testing
        - label: I can help with documentation

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our [Code of Conduct](https://github.com/KoshiHQ/GenOps-AI/blob/main/CODE_OF_CONDUCT.md)
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

### File: `.github/ISSUE_TEMPLATE/documentation.yml`
```yaml
name: 📚 Documentation
description: Report missing, unclear, or incorrect documentation
title: "[Docs]: "
labels: ["documentation", "good first issue"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for helping improve our documentation! 📚
        
        Documentation improvements are a great way to contribute to the project.

  - type: dropdown
    id: doc-type
    attributes:
      label: Documentation type
      description: What kind of documentation needs improvement?
      options:
        - README or Getting Started
        - API Documentation
        - Examples or Tutorials
        - Integration Guides
        - Troubleshooting
        - Other
    validations:
      required: true

  - type: textarea
    id: issue
    attributes:
      label: What's missing or unclear?
      description: Describe the documentation issue.
      placeholder: "The cost attribution example doesn't show how to..."
    validations:
      required: true

  - type: textarea
    id: suggestion
    attributes:
      label: Suggested improvement
      description: How could we improve this documentation?
      placeholder: "It would be helpful to add an example showing..."
    validations:
      required: false

  - type: textarea
    id: context
    attributes:
      label: Additional context
      description: Any other context about the documentation issue?
    validations:
      required: false
```

## Step 3: Create Good First Issues

Create these issues immediately after setting up labels:

### Issue 1: Fix failing integration test
```markdown
Title: Fix failing integration test in test_end_to_end.py
Labels: good first issue, ci-fix, help wanted

**Description:**
Our CI integration test is currently failing on some builds. This is a great issue for contributors who enjoy debugging!

**What's happening:**
The integration test in `tests/integration/test_end_to_end.py` occasionally fails on CI.

**Steps to investigate:**
1. Check the GitHub Actions logs
2. Look for patterns in when it fails vs succeeds
3. Run the test locally: `python -m pytest tests/integration/test_end_to_end.py -v`

**Expected outcome:**
- Integration test passes consistently
- Documentation of any fixes applied

**Good for:**
- Contributors familiar with Python testing
- Those who enjoy detective work!
- Anyone wanting to improve project stability

**Resources:**
- [Contributing Guide](CONTRIBUTING.md)
- [Test documentation](docs/development/)
```

### Issue 2: Add cost calculation examples
```markdown
Title: Add cost calculation examples for different AI models
Labels: good first issue, documentation, help wanted

**Description:**
Help developers understand GenOps cost calculations by adding clear examples.

**What's needed:**
Add examples to `examples/` showing:
1. GPT-4 vs GPT-3.5 cost comparison
2. Claude model cost calculations  
3. Cost per customer scenario

**Files to update:**
- Create `examples/cost_calculations.py`
- Update `examples/README.md` with new example

**Good for:**
- First-time contributors
- Those who want to help others learn
- Documentation enthusiasts

**Acceptance criteria:**
- Working Python examples
- Clear comments explaining calculations
- Tests for the examples
```

### Issue 3: Create Azure OpenAI setup guide
```markdown
Title: Create Azure OpenAI integration guide
Labels: good first issue, documentation, provider

**Description:**
Many users want to use GenOps with Azure OpenAI instead of OpenAI directly. Let's help them!

**What's needed:**
Create documentation showing:
1. How to configure GenOps for Azure OpenAI
2. Any differences in cost calculation
3. Example code

**Files to create:**
- `docs/integrations/azure-openai.md`
- `examples/azure_openai_setup.py`

**Resources:**
- [Azure OpenAI docs](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [OpenAI provider code](src/genops/providers/openai.py)

**Good for:**
- Contributors familiar with Azure
- Documentation writers
- Those who want to expand platform support
```

## Step 4: Update CONTRIBUTING.md

Add this section to CONTRIBUTING.md:

```markdown
## 🎯 Quick Start for Contributors

### 5-Minute Wins
Perfect for your first contribution:
- Fix typos in documentation
- Improve code comments
- Add examples to existing files
- Update badges or links

### 15-Minute Tasks
Great for building confidence:
- Add tests for existing functions
- Improve error messages
- Create simple documentation pages
- Fix CI test issues

### Bigger Challenges
For ongoing contributors:
- New AI provider integrations
- Dashboard templates
- Performance improvements
- Advanced governance patterns

### Finding Your First Issue
1. Browse [good first issues](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
2. Check [help wanted](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
3. Look for [CI fixes](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Aci-fix) if you enjoy debugging!

**Not sure which issue to pick?** Comment on an issue asking for guidance - we're here to help!
```

## Implementation Priority

1. **Immediate (Today):**
   - Create GitHub labels
   - Create the 3 good first issues above
   
2. **This Week:**
   - Add issue templates
   - Update CONTRIBUTING.md
   
3. **Ongoing:**
   - Monitor issues for community engagement
   - Create more good first issues as needed
   - Celebrate contributors!

## Success Metrics

Track these weekly:
- Number of new contributors
- Issues labeled with "good first issue" 
- Community engagement on issues
- Time to first response on issues

The goal is 2-3 new contributors within 2 weeks of implementing this setup.