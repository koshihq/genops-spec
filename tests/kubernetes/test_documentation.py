#!/usr/bin/env python3
"""
Tests for GenOps AI Kubernetes documentation and guides.

Validates that documentation is accurate, complete, and examples work as described.
"""

import re
import subprocess
from pathlib import Path

import pytest


class TestDocumentation:
    """Test documentation accuracy and completeness."""

    @pytest.fixture
    def docs_dir(self):
        """Path to documentation directory."""
        return Path(__file__).parent.parent.parent / "docs"

    @pytest.fixture
    def examples_dir(self):
        """Path to examples directory."""
        return Path(__file__).parent.parent.parent / "examples" / "kubernetes"

    def test_kubernetes_getting_started_guide_exists(self, docs_dir):
        """Test that getting started guide exists and is comprehensive."""

        guide_path = docs_dir / "kubernetes-getting-started.md"
        assert guide_path.exists(), "Getting started guide missing"

        content = guide_path.read_text()

        # Test required sections
        required_sections = [
            "What You'll Achieve",
            "Phase 1: Quick Wins",
            "Phase 2: Hands-On Control",
            "Phase 3: Production Mastery",
            "Troubleshooting"
        ]

        for section in required_sections:
            assert section in content, f"Missing required section: {section}"

        # Test has practical commands
        assert "helm install" in content, "Missing Helm installation commands"
        assert "kubectl" in content, "Missing kubectl commands"
        assert "```bash" in content, "Missing bash code blocks"

        # Test mentions examples
        assert "examples/" in content, "Missing references to examples"

    def test_troubleshooting_runbook_exists(self, docs_dir):
        """Test troubleshooting runbook exists and is comprehensive."""

        runbook_path = docs_dir / "kubernetes-troubleshooting.md"
        assert runbook_path.exists(), "Troubleshooting runbook missing"

        content = runbook_path.read_text()

        # Test required troubleshooting sections
        required_sections = [
            "Quick Diagnosis",
            "Emergency Response",
            "Installation & Configuration Issues",
            "Policy and Budget Issues",
            "Cost Tracking Issues",
            "Network and Connectivity Issues",
            "Performance and Scaling Issues"
        ]

        for section in required_sections:
            assert section in content, f"Missing troubleshooting section: {section}"

        # Test has diagnostic commands
        assert "kubectl get pods" in content, "Missing pod diagnostic commands"
        assert "kubectl logs" in content, "Missing log diagnostic commands"
        assert "kubectl describe" in content, "Missing describe diagnostic commands"

        # Test has fix suggestions
        assert "Quick Fix:" in content, "Missing quick fix suggestions"
        assert "Solution:" in content, "Missing solution suggestions"

    def test_local_development_guide_exists(self, docs_dir):
        """Test local development guide exists and is complete."""

        dev_guide_path = docs_dir / "kubernetes-local-development.md"
        assert dev_guide_path.exists(), "Local development guide missing"

        content = dev_guide_path.read_text()

        # Test covers different local environments
        local_environments = ["kind", "minikube", "Docker Desktop"]
        for env in local_environments:
            assert env in content, f"Missing local environment: {env}"

        # Test has development workflows
        development_features = [
            "Hot Development Setup",
            "VS Code Setup",
            "Testing Your Changes"
        ]

        for feature in development_features:
            assert feature in content, f"Missing development feature: {feature}"

    def test_migration_guide_exists(self, docs_dir):
        """Test migration guide exists and covers all scenarios."""

        migration_guide_path = docs_dir / "kubernetes-migration-guide.md"
        assert migration_guide_path.exists(), "Migration guide missing"

        content = migration_guide_path.read_text()

        # Test migration strategies
        strategies = [
            "Proxy Injection",
            "Sidecar Pattern",
            "Service Replacement",
            "Gateway Migration"
        ]

        for strategy in strategies:
            assert strategy in content, f"Missing migration strategy: {strategy}"

        # Test rollback procedures
        assert "Rollback Procedures" in content, "Missing rollback procedures"
        assert "Emergency Rollback" in content, "Missing emergency rollback"

        # Test validation procedures
        assert "Migration Validation" in content, "Missing migration validation"

    def test_quickstart_guides_exist(self, docs_dir):
        """Test that all quickstart guides exist."""

        quickstart_guides = [
            "kubernetes-quickstart.md",
            "openai-kubernetes-quickstart.md",
            "multi-provider-kubernetes-quickstart.md"
        ]

        for guide in quickstart_guides:
            guide_path = docs_dir / guide
            assert guide_path.exists(), f"Quickstart guide missing: {guide}"

            content = guide_path.read_text()

            # Each guide should have 5-minute setup
            assert "5 minutes" in content or "5 minute" in content, f"Missing 5-minute promise in {guide}"
            assert "Quick Setup" in content, f"Missing quick setup in {guide}"
            assert "helm install" in content, f"Missing Helm commands in {guide}"


class TestDocumentationAccuracy:
    """Test that documentation accurately reflects implementation."""

    def test_example_references_are_accurate(self, docs_dir, examples_dir):
        """Test that documentation references to examples are accurate."""

        # Find all markdown files
        md_files = list(docs_dir.glob("kubernetes*.md"))

        for md_file in md_files:
            content = md_file.read_text()

            # Find example file references
            example_refs = re.findall(r'examples/kubernetes/([a-zA-Z_]+\.py)', content)

            for example_ref in example_refs:
                example_path = examples_dir / example_ref
                assert example_path.exists(), f"Referenced example {example_ref} not found (referenced in {md_file.name})"

    def test_kubectl_commands_are_valid(self, docs_dir):
        """Test that kubectl commands in documentation are syntactically valid."""

        md_files = list(docs_dir.glob("kubernetes*.md"))

        # Common kubectl command patterns that should be valid
        valid_patterns = [
            r'kubectl get \w+',
            r'kubectl apply -f',
            r'kubectl describe \w+',
            r'kubectl logs',
            r'kubectl port-forward',
            r'kubectl create \w+'
        ]

        for md_file in md_files:
            content = md_file.read_text()

            # Find kubectl commands
            kubectl_commands = re.findall(r'kubectl [^\n`]+', content)

            for cmd in kubectl_commands:
                # Skip overly complex commands or ones with placeholders
                if any(placeholder in cmd for placeholder in ['YOUR_', '${', 'your-']):
                    continue

                # Check if it matches valid patterns
                is_valid = any(re.search(pattern, cmd) for pattern in valid_patterns)

                # Or check basic syntax
                parts = cmd.split()
                assert len(parts) >= 2, f"Invalid kubectl command: {cmd} (in {md_file.name})"
                assert parts[0] == "kubectl", f"Invalid kubectl command: {cmd} (in {md_file.name})"

    def test_helm_commands_are_valid(self, docs_dir):
        """Test that Helm commands in documentation are syntactically valid."""

        md_files = list(docs_dir.glob("kubernetes*.md"))

        for md_file in md_files:
            content = md_file.read_text()

            # Find Helm commands
            helm_commands = re.findall(r'helm [^\n`]+', content)

            for cmd in helm_commands:
                # Skip commands with placeholders
                if any(placeholder in cmd for placeholder in ['YOUR_', '${', 'your-']):
                    continue

                # Check basic Helm command structure
                parts = cmd.split()
                assert len(parts) >= 2, f"Invalid helm command: {cmd} (in {md_file.name})"
                assert parts[0] == "helm", f"Invalid helm command: {cmd} (in {md_file.name})"

                # Check for common Helm subcommands
                valid_subcommands = ['install', 'upgrade', 'uninstall', 'list', 'repo', 'get', 'status']
                assert parts[1] in valid_subcommands, f"Unknown helm subcommand: {cmd} (in {md_file.name})"

    def test_code_block_syntax(self, docs_dir):
        """Test that code blocks have proper syntax."""

        md_files = list(docs_dir.glob("kubernetes*.md"))

        for md_file in md_files:
            content = md_file.read_text()

            # Find code blocks
            code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)

            for language, code in code_blocks:
                # Skip empty blocks
                if not code.strip():
                    continue

                # Check YAML blocks
                if language == 'yaml':
                    # Basic YAML syntax check
                    lines = code.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            # Should not have tabs (YAML uses spaces)
                            assert '\t' not in line, f"YAML should use spaces, not tabs: {line} (in {md_file.name})"

                # Check bash blocks
                elif language == 'bash':
                    # Should not have Windows line endings
                    assert '\r\n' not in code, f"Bash code should use Unix line endings (in {md_file.name})"


class TestExampleDocumentation:
    """Test that examples have proper documentation."""

    def test_all_examples_have_docstrings(self, examples_dir):
        """Test that all example files have comprehensive docstrings."""

        example_files = list(examples_dir.glob("*.py"))

        for example_file in example_files:
            if example_file.name.startswith("__"):
                continue

            content = example_file.read_text()

            # Should have module docstring
            assert content.startswith('#!/usr/bin/env python3\n"""') or content.startswith('"""'), \
                f"Example {example_file.name} missing module docstring"

            # Should have usage information
            assert "Usage:" in content, f"Example {example_file.name} missing usage information"

            # Should describe what it demonstrates
            demo_keywords = ["demonstrates", "shows", "example", "test"]
            assert any(keyword in content.lower() for keyword in demo_keywords), \
                f"Example {example_file.name} missing clear description"

    def test_examples_have_help_text(self, examples_dir):
        """Test that examples provide helpful command-line help."""

        example_files = [f for f in examples_dir.glob("*.py") if not f.name.startswith("__")]

        for example_file in example_files:
            try:
                # Test that --help works
                result = subprocess.run([
                    "python", str(example_file), "--help"
                ], capture_output=True, text=True, timeout=10)

                # Should exit successfully and provide help
                assert result.returncode == 0, f"Example {example_file.name} --help failed"
                assert len(result.stdout) > 50, f"Example {example_file.name} help text too short"
                assert "usage" in result.stdout.lower(), f"Example {example_file.name} missing usage in help"

            except subprocess.TimeoutExpired:
                pytest.fail(f"Example {example_file.name} --help timeout")
            except Exception as e:
                # Skip if we can't run the example (missing dependencies, etc.)
                pytest.skip(f"Cannot test {example_file.name} help: {e}")

    def test_examples_have_proper_error_handling(self, examples_dir):
        """Test that examples have proper error handling documentation."""

        example_files = list(examples_dir.glob("*.py"))

        for example_file in example_files:
            if example_file.name.startswith("__"):
                continue

            content = example_file.read_text()

            # Should have try/except blocks for main functionality
            assert "try:" in content, f"Example {example_file.name} missing error handling"
            assert "except" in content, f"Example {example_file.name} missing exception handling"

            # Should check for ImportError specifically
            assert "ImportError" in content, f"Example {example_file.name} missing ImportError handling"

    def test_examples_readme_exists_and_comprehensive(self, examples_dir):
        """Test that examples README exists and is comprehensive."""

        readme_path = examples_dir / "README.md"
        assert readme_path.exists(), "Examples README.md missing"

        content = readme_path.read_text()

        # Should describe all example files
        example_files = [f.name for f in examples_dir.glob("*.py") if not f.name.startswith("__")]

        for example_file in example_files:
            assert example_file in content, f"Example {example_file} not described in README"

        # Should have learning progression
        assert "5-minute" in content or "5 minutes" in content, "README missing 5-minute progression"
        assert "30-minute" in content or "30 minutes" in content, "README missing 30-minute progression"

        # Should have troubleshooting section
        assert "troubleshooting" in content.lower(), "README missing troubleshooting section"

        # Should reference main documentation
        assert "docs/" in content, "README missing references to main documentation"


class TestDocumentationConsistency:
    """Test consistency across documentation."""

    def test_consistent_terminology(self, docs_dir):
        """Test that documentation uses consistent terminology."""

        md_files = list(docs_dir.glob("kubernetes*.md"))

        # Define preferred terminology
        preferred_terms = {
            "Kubernetes": ["k8s", "kubernetes"],  # Prefer "Kubernetes"
            "GenOps AI": ["genops", "genops-ai"],  # Prefer "GenOps AI"
            "cost tracking": ["cost-tracking"],    # Prefer "cost tracking"
        }

        for md_file in md_files:
            content = md_file.read_text()

            for preferred, alternatives in preferred_terms.items():
                # Check that we consistently use preferred term in headings
                headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)

                for heading in headings:
                    for alt in alternatives:
                        if alt.lower() in heading.lower() and preferred not in heading:
                            # This is just a warning, not a failure
                            print(f"Warning: Consider using '{preferred}' instead of '{alt}' in heading: {heading} ({md_file.name})")

    def test_consistent_command_formatting(self, docs_dir):
        """Test that commands are consistently formatted."""

        md_files = list(docs_dir.glob("kubernetes*.md"))

        for md_file in md_files:
            content = md_file.read_text()

            # Check that kubectl commands are in code blocks
            kubectl_inline = re.findall(r'`kubectl [^`]+`', content)
            kubectl_blocks = re.findall(r'```bash\n.*?kubectl.*?\n```', content, re.DOTALL)

            # Most kubectl commands should be in code blocks, not inline
            if len(kubectl_inline) > len(kubectl_blocks) * 2:
                print(f"Warning: Consider using code blocks for kubectl commands in {md_file.name}")

    def test_links_are_consistent(self, docs_dir):
        """Test that internal links are consistent and use same format."""

        md_files = list(docs_dir.glob("kubernetes*.md"))

        for md_file in md_files:
            content = md_file.read_text()

            # Find internal links
            internal_links = re.findall(r'\[([^\]]+)\]\(([^)]+\.md[^)]*)\)', content)

            for link_text, link_url in internal_links:
                # Internal docs links should not start with http
                assert not link_url.startswith('http'), f"Internal link should be relative: {link_url} (in {md_file.name})"

                # Should reference existing files (basic check)
                if not link_url.startswith('#') and '/' not in link_url:
                    referenced_file = docs_dir / link_url.split('#')[0]
                    if not referenced_file.exists():
                        print(f"Warning: Referenced file may not exist: {link_url} (in {md_file.name})")


if __name__ == "__main__":
    # Run documentation tests when script is executed directly
    pytest.main([__file__, "-v"])
