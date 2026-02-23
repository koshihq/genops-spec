"""Tag validation and enforcement for GenOps AI attribution."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for tag validation violations."""

    WARNING = "warning"  # Log warning, allow operation
    ERROR = "error"  # Log error, allow operation but mark as invalid
    BLOCK = "block"  # Raise exception, prevent operation


@dataclass
class ValidationRule:
    """A single tag validation rule."""

    name: str
    attribute: str
    rule_type: str  # 'required', 'pattern', 'enum', 'length', 'custom'
    severity: ValidationSeverity
    description: str

    # Rule-specific parameters
    pattern: str | None = None
    allowed_values: set[str] | None = None
    min_length: int | None = None
    max_length: int | None = None
    validator_func: Callable[[Any], bool] | None = None
    error_message: str | None = None


@dataclass
class ValidationResult:
    """Result of tag validation."""

    valid: bool
    violations: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    cleaned_attributes: dict[str, Any]


class TagValidator:
    """Validates and enforces attribution tag quality and compliance."""

    def __init__(self):
        self.rules: list[ValidationRule] = []
        self.enabled = True

        # Set up default validation rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up default validation rules for common attribution patterns."""

        # Team validation
        self.add_rule(
            ValidationRule(
                name="team_required",
                attribute="team",
                rule_type="required",
                severity=ValidationSeverity.WARNING,
                description="Team is required for proper cost attribution",
                error_message="Team attribute should be specified for cost tracking",
            )
        )

        self.add_rule(
            ValidationRule(
                name="team_format",
                attribute="team",
                rule_type="pattern",
                severity=ValidationSeverity.WARNING,
                description="Team names should follow kebab-case format",
                pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
                error_message="Team should be lowercase with hyphens (e.g., 'platform-engineering')",
            )
        )

        # Customer ID validation
        self.add_rule(
            ValidationRule(
                name="customer_id_format",
                attribute="customer_id",
                rule_type="pattern",
                severity=ValidationSeverity.ERROR,
                description="Customer IDs should follow standard format",
                pattern=r"^[a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?$",
                error_message="Customer ID should be alphanumeric with hyphens/underscores",
            )
        )

        # Environment validation
        self.add_rule(
            ValidationRule(
                name="environment_enum",
                attribute="environment",
                rule_type="enum",
                severity=ValidationSeverity.WARNING,
                description="Environment should be standard value",
                allowed_values={
                    "production",
                    "staging",
                    "development",
                    "test",
                    "local",
                },
                error_message="Environment should be one of: production, staging, development, test, local",
            )
        )

        # Feature validation
        self.add_rule(
            ValidationRule(
                name="feature_length",
                attribute="feature",
                rule_type="length",
                severity=ValidationSeverity.WARNING,
                description="Feature names should be reasonable length",
                min_length=2,
                max_length=50,
                error_message="Feature name should be 2-50 characters",
            )
        )

        # User ID validation
        self.add_rule(
            ValidationRule(
                name="user_id_not_empty",
                attribute="user_id",
                rule_type="custom",
                severity=ValidationSeverity.ERROR,
                description="User ID should not be empty string",
                validator_func=lambda x: (
                    x is None or (isinstance(x, str) and len(x.strip()) > 0)
                ),
                error_message="User ID should not be empty string",
            )
        )

    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules.append(rule)
        logger.debug(f"Added validation rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove a validation rule by name."""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.debug(f"Removed validation rule: {rule_name}")

    def enable(self):
        """Enable tag validation."""
        self.enabled = True
        logger.info("Tag validation enabled")

    def disable(self):
        """Disable tag validation."""
        self.enabled = False
        logger.info("Tag validation disabled")

    def validate(self, attributes: dict[str, Any]) -> ValidationResult:
        """
        Validate attribution attributes against all rules.

        Args:
            attributes: Dictionary of attribution attributes to validate

        Returns:
            ValidationResult with validation status and any violations
        """
        if not self.enabled:
            return ValidationResult(
                valid=True, violations=[], warnings=[], cleaned_attributes=attributes
            )

        violations = []
        warnings = []
        cleaned_attributes = attributes.copy()

        # Apply each validation rule
        for rule in self.rules:
            try:
                violation = self._apply_rule(rule, attributes)
                if violation:
                    if rule.severity == ValidationSeverity.WARNING:
                        warnings.append(violation)
                    else:
                        violations.append(violation)
            except Exception as e:
                logger.error(f"Error applying validation rule {rule.name}: {e}")
                violations.append(
                    {
                        "rule": rule.name,
                        "attribute": rule.attribute,
                        "severity": "error",
                        "message": f"Validation rule failed: {e}",
                    }
                )

        # Determine overall validity (only blocking violations make it invalid)
        blocking_violations = [v for v in violations if v.get("severity") == "block"]
        valid = len(blocking_violations) == 0

        result = ValidationResult(
            valid=valid,
            violations=violations,
            warnings=warnings,
            cleaned_attributes=cleaned_attributes,
        )

        # Log results
        if violations or warnings:
            logger.info(
                f"Tag validation: {len(violations)} violations, {len(warnings)} warnings"
            )
            for violation in violations:
                logger.error(f"Validation violation: {violation}")
            for warning in warnings:
                logger.warning(f"Validation warning: {warning}")

        return result

    def _apply_rule(
        self, rule: ValidationRule, attributes: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply a single validation rule to attributes."""

        attr_value = attributes.get(rule.attribute)

        if rule.rule_type == "required":
            if attr_value is None or attr_value == "":
                return {
                    "rule": rule.name,
                    "attribute": rule.attribute,
                    "severity": rule.severity.value,
                    "message": rule.error_message or f"{rule.attribute} is required",
                    "value": attr_value,
                }

        # Skip other validations if attribute is not present (unless required)
        if attr_value is None:
            return None

        if rule.rule_type == "pattern" and rule.pattern:
            if isinstance(attr_value, str) and not re.match(rule.pattern, attr_value):
                return {
                    "rule": rule.name,
                    "attribute": rule.attribute,
                    "severity": rule.severity.value,
                    "message": rule.error_message
                    or f"{rule.attribute} does not match required pattern",
                    "value": attr_value,
                    "expected_pattern": rule.pattern,
                }

        elif rule.rule_type == "enum" and rule.allowed_values:
            if attr_value not in rule.allowed_values:
                return {
                    "rule": rule.name,
                    "attribute": rule.attribute,
                    "severity": rule.severity.value,
                    "message": rule.error_message
                    or f"{rule.attribute} must be one of: {', '.join(rule.allowed_values)}",
                    "value": attr_value,
                    "allowed_values": list(rule.allowed_values),
                }

        elif rule.rule_type == "length":
            if isinstance(attr_value, str):
                length = len(attr_value)
                if rule.min_length and length < rule.min_length:
                    return {
                        "rule": rule.name,
                        "attribute": rule.attribute,
                        "severity": rule.severity.value,
                        "message": rule.error_message
                        or f"{rule.attribute} must be at least {rule.min_length} characters",
                        "value": attr_value,
                        "actual_length": length,
                        "min_length": rule.min_length,
                    }
                if rule.max_length and length > rule.max_length:
                    return {
                        "rule": rule.name,
                        "attribute": rule.attribute,
                        "severity": rule.severity.value,
                        "message": rule.error_message
                        or f"{rule.attribute} must be no more than {rule.max_length} characters",
                        "value": attr_value,
                        "actual_length": length,
                        "max_length": rule.max_length,
                    }

        elif rule.rule_type == "custom" and rule.validator_func:
            try:
                if not rule.validator_func(attr_value):
                    return {
                        "rule": rule.name,
                        "attribute": rule.attribute,
                        "severity": rule.severity.value,
                        "message": rule.error_message
                        or f"{rule.attribute} failed custom validation",
                        "value": attr_value,
                    }
            except Exception as e:
                return {
                    "rule": rule.name,
                    "attribute": rule.attribute,
                    "severity": "error",
                    "message": f"Custom validation failed: {e}",
                    "value": attr_value,
                }

        return None

    def enforce(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """
        Validate attributes and enforce rules, raising exceptions for blocking violations.

        Args:
            attributes: Attribution attributes to validate

        Returns:
            Cleaned attributes if validation passes

        Raises:
            TagValidationError: If there are blocking validation violations
        """
        result = self.validate(attributes)

        # Check for blocking violations
        blocking_violations = [
            v for v in result.violations if v.get("severity") == "block"
        ]
        if blocking_violations:
            raise TagValidationError(
                f"Tag validation blocked operation with {len(blocking_violations)} violations",
                violations=blocking_violations,
                warnings=result.warnings,
            )

        return result.cleaned_attributes


class TagValidationError(Exception):
    """Exception raised when tag validation blocks an operation."""

    def __init__(
        self,
        message: str,
        violations: list[dict[str, Any]],
        warnings: list[dict[str, Any]],
    ):
        super().__init__(message)
        self.violations = violations
        self.warnings = warnings


# Global validator instance
_global_validator = TagValidator()


def get_validator() -> TagValidator:
    """Get the global tag validator instance."""
    return _global_validator


def validate_tags(attributes: dict[str, Any]) -> ValidationResult:
    """Convenience function to validate tags using the global validator."""
    return _global_validator.validate(attributes)


def enforce_tags(attributes: dict[str, Any]) -> dict[str, Any]:
    """Convenience function to enforce tag validation using the global validator."""
    return _global_validator.enforce(attributes)


def add_validation_rule(rule: ValidationRule):
    """Add a validation rule to the global validator."""
    _global_validator.add_rule(rule)


def remove_validation_rule(rule_name: str):
    """Remove a validation rule from the global validator."""
    _global_validator.remove_rule(rule_name)


# Common validation rule templates
def create_required_rule(
    attribute: str, severity: ValidationSeverity = ValidationSeverity.WARNING
) -> ValidationRule:
    """Create a required field validation rule."""
    return ValidationRule(
        name=f"{attribute}_required",
        attribute=attribute,
        rule_type="required",
        severity=severity,
        description=f"{attribute} is required for proper attribution",
        error_message=f"{attribute} attribute is required",
    )


def create_enum_rule(
    attribute: str,
    allowed_values: set[str],
    severity: ValidationSeverity = ValidationSeverity.WARNING,
) -> ValidationRule:
    """Create an enum validation rule."""
    return ValidationRule(
        name=f"{attribute}_enum",
        attribute=attribute,
        rule_type="enum",
        severity=severity,
        description=f"{attribute} must be one of allowed values",
        allowed_values=allowed_values,
        error_message=f"{attribute} must be one of: {', '.join(allowed_values)}",
    )


def create_pattern_rule(
    attribute: str,
    pattern: str,
    description: str,
    severity: ValidationSeverity = ValidationSeverity.WARNING,
) -> ValidationRule:
    """Create a pattern validation rule."""
    return ValidationRule(
        name=f"{attribute}_pattern",
        attribute=attribute,
        rule_type="pattern",
        severity=severity,
        description=description,
        pattern=pattern,
        error_message=f"{attribute} does not match required format",
    )
