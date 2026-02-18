#!/usr/bin/env python3
"""
README Format Validation Script for GenOps AI

Validates that the README.md integration list follows the established format patterns:
- ✅ [Name](link) (<a href="external" target="_blank">↗</a>)
- ☐ Name (<a href="external" target="_blank">↗</a>)

This script prevents the recurring issue of adding descriptive text to integration entries,
which violates the established README formatting standards.
"""

import re
import sys
from pathlib import Path
from typing import NamedTuple


class ValidationError(NamedTuple):
    """Represents a README formatting validation error."""

    line_number: int
    line_content: str
    error_type: str
    suggestion: str


class ValidationResult(NamedTuple):
    """Results of README validation."""

    is_valid: bool
    errors: list[ValidationError]
    total_lines_checked: int


# Integration list format patterns
INTEGRATION_PATTERN = re.compile(
    r'^- ✅ \[([^\]]+)\]\(([^)]+)\) \(<a href="([^"]+)" target="_blank">↗</a>\)$'
)

PLANNED_PATTERN = re.compile(r'^- ☐ (.+?) \(<a href="([^"]+)" target="_blank">↗</a>\)$')

# Violation patterns - these are FORBIDDEN
VIOLATION_PATTERN = re.compile(
    r'^- ✅ \[([^\]]+)\]\(([^)]+)\) \(<a href="([^"]+)" target="_blank">↗</a>\) - (.+)$'
)

SECTION_HEADER_PATTERN = re.compile(r"^###?\s+.*$")
COMMENT_PATTERN = re.compile(r"^<!--.*-->$")
EMPTY_LINE_PATTERN = re.compile(r"^\s*$")


def validate_readme_format(readme_path: Path) -> ValidationResult:
    """
    Validate README.md integration list format.

    Returns ValidationResult with any format violations found.
    """
    if not readme_path.exists():
        return ValidationResult(
            is_valid=False,
            errors=[
                ValidationError(
                    0, "", "missing_file", f"README file not found: {readme_path}"
                )
            ],
            total_lines_checked=0,
        )

    errors = []
    total_lines_checked = 0
    in_integrations_section = False

    with open(readme_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or COMMENT_PATTERN.match(line) or EMPTY_LINE_PATTERN.match(line):
            continue

        # Check if we're entering the integrations section
        if (
            line == "### 🧠 AI & LLM Ecosystem"
            or line == "### 🏗️ Platform & Infrastructure"
        ):
            in_integrations_section = True
            continue

        # Check if we're leaving the integrations section
        if in_integrations_section and SECTION_HEADER_PATTERN.match(line):
            in_integrations_section = False
            continue

        # Only validate lines in the integrations sections
        if not in_integrations_section:
            continue

        # Skip lines that don't look like integration entries
        if not line.startswith("- "):
            continue

        total_lines_checked += 1

        # Check for violations - descriptive text after integration entry
        violation_match = VIOLATION_PATTERN.match(line)
        if violation_match:
            name = violation_match.group(1)
            descriptive_text = violation_match.group(4)

            errors.append(
                ValidationError(
                    line_number=line_num,
                    line_content=line,
                    error_type="descriptive_text_violation",
                    suggestion=f"Remove descriptive text '- {descriptive_text}' from [{name}] entry. "
                    f"Integration entries must only contain name and links.",
                )
            )
            continue

        # Validate correct patterns
        if line.startswith("- ✅"):
            if not INTEGRATION_PATTERN.match(line):
                errors.append(
                    ValidationError(
                        line_number=line_num,
                        line_content=line,
                        error_type="invalid_completed_format",
                        suggestion="Completed integration format must be: "
                        '- ✅ [Name](internal-link) (<a href="external-link" target="_blank">↗</a>)',
                    )
                )
        elif line.startswith("- ☐"):
            if not PLANNED_PATTERN.match(line):
                errors.append(
                    ValidationError(
                        line_number=line_num,
                        line_content=line,
                        error_type="invalid_planned_format",
                        suggestion="Planned integration format must be: "
                        '- ☐ Name (<a href="external-link" target="_blank">↗</a>)',
                    )
                )
        else:
            # Unknown integration format
            errors.append(
                ValidationError(
                    line_number=line_num,
                    line_content=line,
                    error_type="unknown_format",
                    suggestion="Integration entries must start with '- ✅' or '- ☐'",
                )
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        total_lines_checked=total_lines_checked,
    )


def print_validation_results(result: ValidationResult, readme_path: Path) -> None:
    """Print human-readable validation results."""
    print(f"\n📋 README Format Validation Results for {readme_path}")
    print(f"📊 Lines checked: {result.total_lines_checked}")

    if result.is_valid:
        print("✅ All integration entries follow the correct format!")
        print("\n✨ No formatting violations found. README is properly formatted.")
        return

    print(f"❌ Found {len(result.errors)} formatting violation(s):")
    print()

    # Group errors by type for better reporting
    errors_by_type = {}
    for error in result.errors:
        if error.error_type not in errors_by_type:
            errors_by_type[error.error_type] = []
        errors_by_type[error.error_type].append(error)

    for error_type, errors in errors_by_type.items():
        if error_type == "descriptive_text_violation":
            print("🚨 CRITICAL: Descriptive text violations (most common issue):")
            for error in errors:
                print(f"   Line {error.line_number}: {error.line_content[:80]}...")
                print(f"   💡 Fix: {error.suggestion}")
                print()
        else:
            print(f"🔧 Format Issues ({error_type}):")
            for error in errors:
                print(f"   Line {error.line_number}: {error.line_content[:80]}...")
                print(f"   💡 Fix: {error.suggestion}")
                print()

    print("📚 README Formatting Standards:")
    print(
        '   ✅ Completed: - ✅ [Name](link) (<a href="external" target="_blank">↗</a>)'
    )
    print('   ☐ Planned:   - ☐ Name (<a href="external" target="_blank">↗</a>)')
    print("   ❌ NEVER add descriptive text after integration entries!")
    print()
    print("📖 See CLAUDE.md for complete formatting guidelines")


def main() -> int:
    """Main validation function."""
    if len(sys.argv) > 1:
        readme_path = Path(sys.argv[1])
    else:
        # Default to README.md in current directory
        readme_path = Path("README.md")

    result = validate_readme_format(readme_path)
    print_validation_results(result, readme_path)

    # Return appropriate exit code for CI/CD integration
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
