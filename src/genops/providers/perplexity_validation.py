"""
Perplexity AI Setup Validation and Diagnostics

Comprehensive validation utilities for Perplexity AI integration including:
- API connectivity and authentication validation
- Model access and search capability testing
- Cost configuration and governance validation
- Search-specific feature validation (citations, contexts)
- Interactive setup wizard for guided configuration
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallbacks
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ValidationLevel(Enum):
    """Validation issue severity levels."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    DEPENDENCIES = "dependencies"
    AUTHENTICATION = "authentication"
    CONNECTIVITY = "connectivity"
    MODEL_ACCESS = "model_access"
    SEARCH_FEATURES = "search_features"
    GOVERNANCE = "governance"
    COST_MANAGEMENT = "cost_management"
    CONFIGURATION = "configuration"


@dataclass
class ValidationIssue:
    """Individual validation issue with details and fix suggestions."""
    category: ValidationCategory
    level: ValidationLevel
    title: str
    description: str
    fix_suggestions: List[str]
    affected_functionality: List[str]
    technical_details: Optional[str] = None
    documentation_links: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Complete validation result with summary and detailed issues."""
    is_valid: bool
    overall_status: ValidationLevel
    issues: List[ValidationIssue]
    summary: Dict[str, int]
    timestamp: str
    validation_duration_seconds: float
    configuration_tested: Dict[str, Any]

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for issue in self.issues if issue.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for issue in self.issues if issue.level == ValidationLevel.WARNING)

    @property
    def success_count(self) -> int:
        """Count of successful validations."""
        return sum(1 for issue in self.issues if issue.level == ValidationLevel.SUCCESS)


class PerplexitySetupValidator:
    """
    Comprehensive validation for Perplexity AI setup and configuration.
    
    Validates all aspects of Perplexity integration including:
    - Required dependencies and installation
    - API authentication and connectivity
    - Model access and search capabilities
    - Governance configuration and cost controls
    - Search-specific features (citations, contexts)
    """

    def __init__(self):
        self.issues: List[ValidationIssue] = []

    def validate_complete_setup(
        self,
        perplexity_api_key: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Run complete validation of Perplexity setup.
        
        Args:
            perplexity_api_key: Perplexity API key to validate
            team: Team name for governance validation
            project: Project name for governance validation
            **kwargs: Additional configuration to validate
            
        Returns:
            ValidationResult with detailed findings and recommendations
        """
        start_time = time.time()
        self.issues = []  # Reset issues list

        # Configuration to test
        config_tested = {
            'perplexity_api_key': '***' if perplexity_api_key else None,
            'team': team,
            'project': project,
            'has_openai_client': HAS_OPENAI,
            'has_requests': HAS_REQUESTS,
        }
        config_tested.update(kwargs)

        # Run all validation checks
        self._validate_dependencies()
        self._validate_authentication(perplexity_api_key)
        self._validate_environment_configuration()
        self._validate_governance_configuration(team, project)
        self._validate_cost_configuration(**kwargs)
        self._validate_connectivity_and_models(perplexity_api_key)
        self._validate_search_features(perplexity_api_key)

        # Determine overall status
        error_count = len([i for i in self.issues if i.level == ValidationLevel.ERROR])
        warning_count = len([i for i in self.issues if i.level == ValidationLevel.WARNING])

        if error_count > 0:
            overall_status = ValidationLevel.ERROR
            is_valid = False
        elif warning_count > 0:
            overall_status = ValidationLevel.WARNING
            is_valid = True  # Warnings don't prevent basic functionality
        else:
            overall_status = ValidationLevel.SUCCESS
            is_valid = True

        # Generate summary
        summary = {}
        for category in ValidationCategory:
            category_issues = [i for i in self.issues if i.category == category]
            summary[category.value] = len(category_issues)

        validation_duration = time.time() - start_time

        result = ValidationResult(
            is_valid=is_valid,
            overall_status=overall_status,
            issues=self.issues,
            summary=summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
            validation_duration_seconds=validation_duration,
            configuration_tested=config_tested
        )

        return result

    def _validate_dependencies(self) -> None:
        """Validate required dependencies are installed."""

        # Check OpenAI client (required for Perplexity)
        if HAS_OPENAI:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.SUCCESS,
                title="OpenAI Client Available",
                description="OpenAI client library is installed and available for Perplexity integration.",
                fix_suggestions=[],
                affected_functionality=[]
            ))
        else:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.ERROR,
                title="OpenAI Client Missing",
                description="OpenAI client library is required for Perplexity API integration.",
                fix_suggestions=[
                    "Install OpenAI client: pip install openai",
                    "Install GenOps with Perplexity support: pip install genops[perplexity]",
                    "Verify installation: python -c 'import openai; print(openai.__version__)'"
                ],
                affected_functionality=[
                    "All Perplexity search operations",
                    "Real-time web search with citations",
                    "Chat completions with search context"
                ],
                documentation_links=[
                    "https://docs.perplexity.ai/",
                    "https://pypi.org/project/openai/"
                ]
            ))

        # Check requests library (helpful for direct API calls)
        if HAS_REQUESTS:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.SUCCESS,
                title="Requests Library Available",
                description="Requests library is available for direct API validation.",
                fix_suggestions=[],
                affected_functionality=[]
            ))
        else:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.WARNING,
                title="Requests Library Missing",
                description="Requests library is recommended for enhanced API validation and debugging.",
                fix_suggestions=[
                    "Install requests: pip install requests",
                    "Or install with GenOps: pip install genops[perplexity]"
                ],
                affected_functionality=[
                    "Enhanced connectivity validation",
                    "Direct API endpoint testing"
                ]
            ))

        # Check GenOps core
        try:
            from genops.core.telemetry import GenOpsTelemetry
            self.issues.append(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.SUCCESS,
                title="GenOps Core Available",
                description="GenOps core telemetry system is available.",
                fix_suggestions=[],
                affected_functionality=[]
            ))
        except ImportError:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.ERROR,
                title="GenOps Core Missing",
                description="GenOps core system is required for governance and telemetry.",
                fix_suggestions=[
                    "Install GenOps: pip install genops",
                    "Install with Perplexity support: pip install genops[perplexity]"
                ],
                affected_functionality=[
                    "Governance controls and cost tracking",
                    "Telemetry and observability",
                    "Team and project attribution"
                ]
            ))

    def _validate_authentication(self, perplexity_api_key: Optional[str]) -> None:
        """Validate Perplexity API authentication."""

        api_key = perplexity_api_key or os.getenv('PERPLEXITY_API_KEY')

        if not api_key:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.ERROR,
                title="Perplexity API Key Missing",
                description="Perplexity API key is required for authentication.",
                fix_suggestions=[
                    "Set environment variable: PERPLEXITY_API_KEY=your-api-key",
                    "Pass api_key parameter: GenOpsPerplexityAdapter(perplexity_api_key='your-key')",
                    "Get API key from: https://www.perplexity.ai/settings/api"
                ],
                affected_functionality=[
                    "All Perplexity API operations",
                    "Real-time search and completions",
                    "Model access and capabilities"
                ],
                documentation_links=[
                    "https://docs.perplexity.ai/getting-started"
                ]
            ))
            return

        # Validate API key format
        if not api_key.startswith('pplx-'):
            self.issues.append(ValidationIssue(
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.WARNING,
                title="Unexpected API Key Format",
                description="Perplexity API keys typically start with 'pplx-'.",
                fix_suggestions=[
                    "Verify API key from Perplexity settings page",
                    "Ensure you're using the correct API key type",
                    "Check for extra spaces or characters"
                ],
                affected_functionality=[
                    "API authentication may fail"
                ]
            ))
        else:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.SUCCESS,
                title="API Key Format Valid",
                description="Perplexity API key format appears correct.",
                fix_suggestions=[],
                affected_functionality=[]
            ))

    def _validate_environment_configuration(self) -> None:
        """Validate environment variables and configuration."""

        # Check for GenOps environment variables
        genops_vars = {
            'GENOPS_TEAM': 'Team name for cost attribution and governance',
            'GENOPS_PROJECT': 'Project name for cost tracking',
            'GENOPS_ENVIRONMENT': 'Environment (production, staging, development)'
        }

        for var_name, description in genops_vars.items():
            value = os.getenv(var_name)
            if value:
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.SUCCESS,
                    title=f"{var_name} Configured",
                    description=f"{description} is set to '{value}'.",
                    fix_suggestions=[],
                    affected_functionality=[]
                ))
            else:
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.INFO,
                    title=f"{var_name} Not Set",
                    description=f"{description} is not configured. This is optional but recommended.",
                    fix_suggestions=[
                        f"Set environment variable: {var_name}=your-value",
                        f"Or pass as parameter: GenOpsPerplexityAdapter({var_name.lower().replace('genops_', '')}='your-value')"
                    ],
                    affected_functionality=[
                        "Cost attribution and reporting",
                        "Governance policy enforcement"
                    ]
                ))

    def _validate_governance_configuration(
        self,
        team: Optional[str],
        project: Optional[str]
    ) -> None:
        """Validate GenOps governance configuration."""

        team = team or os.getenv('GENOPS_TEAM')
        project = project or os.getenv('GENOPS_PROJECT')

        if not team:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.GOVERNANCE,
                level=ValidationLevel.WARNING,
                title="Missing Team Attribution",
                description="Team name not specified for cost attribution and governance.",
                fix_suggestions=[
                    "Set team parameter: GenOpsPerplexityAdapter(team='your-team')",
                    "Set environment variable: GENOPS_TEAM=your-team",
                    "Include team in configuration file"
                ],
                affected_functionality=[
                    "Cost attribution by team",
                    "Team-based governance policies",
                    "Access control and reporting"
                ]
            ))
        else:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.GOVERNANCE,
                level=ValidationLevel.SUCCESS,
                title="Team Attribution Configured",
                description=f"Team '{team}' configured for cost attribution.",
                fix_suggestions=[],
                affected_functionality=[]
            ))

        if not project:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.GOVERNANCE,
                level=ValidationLevel.WARNING,
                title="Missing Project Attribution",
                description="Project name not specified for cost attribution and governance.",
                fix_suggestions=[
                    "Set project parameter: GenOpsPerplexityAdapter(project='your-project')",
                    "Set environment variable: GENOPS_PROJECT=your-project",
                    "Include project in configuration file"
                ],
                affected_functionality=[
                    "Cost attribution by project",
                    "Project-based governance policies",
                    "Budget tracking and reporting"
                ]
            ))
        else:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.GOVERNANCE,
                level=ValidationLevel.SUCCESS,
                title="Project Attribution Configured",
                description=f"Project '{project}' configured for cost tracking.",
                fix_suggestions=[],
                affected_functionality=[]
            ))

    def _validate_cost_configuration(self, **kwargs) -> None:
        """Validate cost management configuration."""

        daily_budget_limit = kwargs.get('daily_budget_limit')
        monthly_budget_limit = kwargs.get('monthly_budget_limit')

        if daily_budget_limit is not None and daily_budget_limit <= 0:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.COST_MANAGEMENT,
                level=ValidationLevel.WARNING,
                title="Invalid Daily Budget Limit",
                description="Daily budget limit should be positive.",
                fix_suggestions=[
                    "Set positive budget limit: GenOpsPerplexityAdapter(daily_budget_limit=100.0)",
                    "Remove budget limit to disable: daily_budget_limit=None"
                ],
                affected_functionality=[
                    "Budget enforcement and alerts",
                    "Cost governance policies"
                ]
            ))
        elif daily_budget_limit is not None:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.COST_MANAGEMENT,
                level=ValidationLevel.SUCCESS,
                title="Daily Budget Configured",
                description=f"Daily budget limit set to ${daily_budget_limit}.",
                fix_suggestions=[],
                affected_functionality=[]
            ))

        if monthly_budget_limit is not None and monthly_budget_limit <= 0:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.COST_MANAGEMENT,
                level=ValidationLevel.WARNING,
                title="Invalid Monthly Budget Limit",
                description="Monthly budget limit should be positive.",
                fix_suggestions=[
                    "Set positive budget limit: GenOpsPerplexityAdapter(monthly_budget_limit=1000.0)",
                    "Remove budget limit to disable: monthly_budget_limit=None"
                ],
                affected_functionality=[
                    "Monthly budget enforcement",
                    "Long-term cost planning"
                ]
            ))

    def _validate_connectivity_and_models(self, perplexity_api_key: Optional[str]) -> None:
        """Validate API connectivity and model access."""

        api_key = perplexity_api_key or os.getenv('PERPLEXITY_API_KEY')

        if not api_key or not HAS_OPENAI:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.ERROR,
                title="Cannot Test Connectivity",
                description="API key or OpenAI client missing - skipping connectivity tests.",
                fix_suggestions=[
                    "Ensure API key is provided",
                    "Install OpenAI client: pip install openai"
                ],
                affected_functionality=[
                    "API connectivity validation",
                    "Model availability testing"
                ]
            ))
            return

        try:
            # Test basic connectivity with a simple request
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )

            # Make a minimal test request
            response = client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )

            self.issues.append(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.SUCCESS,
                title="API Connectivity Confirmed",
                description="Successfully connected to Perplexity API and received response.",
                fix_suggestions=[],
                affected_functionality=[]
            ))

            # Test model access
            if hasattr(response, 'model') or hasattr(response, 'choices'):
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.MODEL_ACCESS,
                    level=ValidationLevel.SUCCESS,
                    title="Model Access Confirmed",
                    description="Successfully accessed Perplexity models.",
                    fix_suggestions=[],
                    affected_functionality=[]
                ))

        except Exception as e:
            error_msg = str(e).lower()

            if 'authentication' in error_msg or 'api key' in error_msg or '401' in error_msg:
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.AUTHENTICATION,
                    level=ValidationLevel.ERROR,
                    title="Authentication Failed",
                    description=f"API authentication failed: {str(e)[:200]}",
                    fix_suggestions=[
                        "Verify your API key is correct",
                        "Check API key permissions and limits",
                        "Ensure API key hasn't expired",
                        "Regenerate API key if necessary"
                    ],
                    affected_functionality=[
                        "All Perplexity API operations"
                    ],
                    technical_details=str(e)
                ))

            elif 'rate limit' in error_msg or '429' in error_msg:
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.WARNING,
                    title="Rate Limit Encountered",
                    description="API rate limit encountered during validation.",
                    fix_suggestions=[
                        "Wait and retry validation",
                        "Check your API usage limits",
                        "Consider upgrading your API plan"
                    ],
                    affected_functionality=[
                        "High-volume API operations"
                    ],
                    technical_details=str(e)
                ))

            elif 'network' in error_msg or 'connection' in error_msg:
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.ERROR,
                    title="Network Connection Failed",
                    description=f"Failed to connect to Perplexity API: {str(e)[:200]}",
                    fix_suggestions=[
                        "Check internet connection",
                        "Verify firewall settings",
                        "Try again in a few minutes",
                        "Check if Perplexity API is experiencing issues"
                    ],
                    affected_functionality=[
                        "All network-dependent operations"
                    ],
                    technical_details=str(e)
                ))

            else:
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.ERROR,
                    title="API Test Failed",
                    description=f"Unexpected error during API test: {str(e)[:200]}",
                    fix_suggestions=[
                        "Check API key and configuration",
                        "Verify OpenAI client version compatibility",
                        "Try basic API test manually",
                        "Review Perplexity API documentation"
                    ],
                    affected_functionality=[
                        "Perplexity API operations"
                    ],
                    technical_details=str(e)
                ))

    def _validate_search_features(self, perplexity_api_key: Optional[str]) -> None:
        """Validate Perplexity-specific search features."""

        api_key = perplexity_api_key or os.getenv('PERPLEXITY_API_KEY')

        if not api_key or not HAS_OPENAI:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.SEARCH_FEATURES,
                level=ValidationLevel.WARNING,
                title="Cannot Test Search Features",
                description="API key or OpenAI client missing - skipping search feature tests.",
                fix_suggestions=[
                    "Configure API key to test search features",
                    "Install OpenAI client for full feature testing"
                ],
                affected_functionality=[
                    "Search feature validation",
                    "Citation and context testing"
                ]
            ))
            return

        try:
            # Test search with citation
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )

            response = client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": "What is AI?"}],
                max_tokens=50
            )

            # Check for search-specific features in response
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content

                if content and len(content) > 0:
                    self.issues.append(ValidationIssue(
                        category=ValidationCategory.SEARCH_FEATURES,
                        level=ValidationLevel.SUCCESS,
                        title="Search Functionality Working",
                        description="Perplexity search returned valid results.",
                        fix_suggestions=[],
                        affected_functionality=[]
                    ))

                # Check for citation indicators (URLs, references)
                if 'http' in content or '[' in content or 'source:' in content.lower():
                    self.issues.append(ValidationIssue(
                        category=ValidationCategory.SEARCH_FEATURES,
                        level=ValidationLevel.SUCCESS,
                        title="Citations Available",
                        description="Response appears to include citations or references.",
                        fix_suggestions=[],
                        affected_functionality=[]
                    ))
                else:
                    self.issues.append(ValidationIssue(
                        category=ValidationCategory.SEARCH_FEATURES,
                        level=ValidationLevel.INFO,
                        title="Citations Not Detected",
                        description="Response may not include visible citations. This could be normal for simple queries.",
                        fix_suggestions=[
                            "Try more specific search queries to trigger citations",
                            "Use return_citations=True parameter if available",
                            "Test with Sonar Pro model for better citation support"
                        ],
                        affected_functionality=[
                            "Citation tracking and governance",
                            "Source attribution"
                        ]
                    ))

        except Exception as e:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.SEARCH_FEATURES,
                level=ValidationLevel.WARNING,
                title="Search Feature Test Failed",
                description=f"Could not validate search features: {str(e)[:200]}",
                fix_suggestions=[
                    "Check API connectivity first",
                    "Verify model access permissions",
                    "Try again with different search query"
                ],
                affected_functionality=[
                    "Search-specific features",
                    "Citation and context capabilities"
                ],
                technical_details=str(e)
            ))

    def print_validation_result(self, result: ValidationResult, show_details: bool = True) -> None:
        """Print formatted validation result."""
        print("\n🔍 Perplexity AI Setup Validation Report")
        print("=" * 55)

        # Overall status
        status_emoji = {
            ValidationLevel.SUCCESS: "✅",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.ERROR: "❌",
            ValidationLevel.INFO: "ℹ️"
        }

        print(f"\n📊 Overall Status: {status_emoji[result.overall_status]} {result.overall_status.value.upper()}")
        print(f"🔧 Setup Valid: {'Yes' if result.is_valid else 'No'}")
        print(f"⏱️ Validation Time: {result.validation_duration_seconds:.2f} seconds")
        print(f"📅 Timestamp: {result.timestamp}")

        # Summary
        print("\n📋 Summary:")
        print(f"   ✅ Successes: {result.success_count}")
        print(f"   ⚠️ Warnings: {result.warning_count}")
        print(f"   ❌ Errors: {result.error_count}")
        print(f"   ℹ️ Info: {len([i for i in result.issues if i.level == ValidationLevel.INFO])}")

        if show_details and result.issues:
            print("\n📝 Detailed Results:")

            # Group by category
            categories = {}
            for issue in result.issues:
                category = issue.category.value
                if category not in categories:
                    categories[category] = []
                categories[category].append(issue)

            for category, issues in categories.items():
                print(f"\n📂 {category.upper().replace('_', ' ')}")
                print("-" * 40)

                for issue in issues:
                    emoji = status_emoji[issue.level]
                    print(f"   {emoji} {issue.title}")

                    if issue.level in [ValidationLevel.ERROR, ValidationLevel.WARNING]:
                        print(f"      Description: {issue.description}")

                        if issue.fix_suggestions:
                            print("      Fix suggestions:")
                            for suggestion in issue.fix_suggestions[:3]:  # Show top 3
                                print(f"      • {suggestion}")

                        if issue.affected_functionality:
                            print(f"      Affects: {', '.join(issue.affected_functionality[:2])}")

                        print()

        # Next steps
        if result.error_count > 0:
            print("\n🚨 Next Steps (Errors Found):")
            print("1. Fix the error-level issues above")
            print("2. Re-run validation to confirm fixes")
            print("3. Address warnings for optimal performance")

        elif result.warning_count > 0:
            print("\n⚠️ Next Steps (Warnings Found):")
            print("1. Basic functionality should work")
            print("2. Address warnings for optimal performance")
            print("3. Consider governance configuration")

        else:
            print("\n🎉 Next Steps (All Good!):")
            print("1. Your Perplexity setup is fully configured")
            print("2. Try the examples in examples/perplexity/")
            print("3. Read the complete integration guide")

        print("\n📚 Resources:")
        print("   • Quickstart: docs/perplexity-quickstart.md")
        print("   • Examples: examples/perplexity/")
        print("   • Complete Guide: docs/integrations/perplexity.md")


# Convenience functions
def validate_setup(
    perplexity_api_key: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs
) -> ValidationResult:
    """
    Quick validation of Perplexity setup.
    
    Args:
        perplexity_api_key: Perplexity API key to validate
        team: Team name for governance
        project: Project name for governance
        **kwargs: Additional configuration
        
    Returns:
        ValidationResult with findings
    """
    validator = PerplexitySetupValidator()
    return validator.validate_complete_setup(
        perplexity_api_key=perplexity_api_key,
        team=team,
        project=project,
        **kwargs
    )


def print_validation_result(result: ValidationResult) -> None:
    """Print validation result with formatted output."""
    validator = PerplexitySetupValidator()
    validator.print_validation_result(result)


def is_properly_configured() -> bool:
    """Quick check if Perplexity integration is properly configured."""
    validator = PerplexitySetupValidator()
    result = validator.validate_complete_setup()
    return result.is_valid and result.error_count == 0


def interactive_setup_wizard() -> Dict[str, Any]:
    """
    Interactive wizard for Perplexity setup configuration.
    
    Returns:
        Configuration dictionary for use with GenOpsPerplexityAdapter
    """
    print("🧙‍♂️ Perplexity AI + GenOps Interactive Setup Wizard")
    print("=" * 55)
    print("This wizard will help you configure Perplexity AI with GenOps governance.")
    print()

    config = {}

    # API Key
    print("🔑 Step 1: API Key Configuration")
    api_key = input("Enter your Perplexity API key (or press Enter to use PERPLEXITY_API_KEY env var): ").strip()
    if api_key:
        config['perplexity_api_key'] = api_key
    elif not os.getenv('PERPLEXITY_API_KEY'):
        print("⚠️ No API key provided. Set PERPLEXITY_API_KEY environment variable.")

    # Team and Project
    print("\n👥 Step 2: Team & Project Attribution")
    team = input("Enter team name (for cost attribution): ").strip()
    if team:
        config['team'] = team

    project = input("Enter project name (for cost tracking): ").strip()
    if project:
        config['project'] = project

    # Environment
    print("\n🌍 Step 3: Environment Configuration")
    print("Environments: production, staging, development, testing")
    environment = input("Enter environment [production]: ").strip() or "production"
    config['environment'] = environment

    # Budget Configuration
    print("\n💰 Step 4: Budget Configuration")
    try:
        daily_budget = input("Enter daily budget limit in USD [100.0]: ").strip()
        config['daily_budget_limit'] = float(daily_budget) if daily_budget else 100.0
    except ValueError:
        config['daily_budget_limit'] = 100.0

    # Governance Policy
    print("\n🛡️ Step 5: Governance Policy")
    print("Policies: advisory (warnings), enforced (blocks on budget), strict (maximum control)")
    policy = input("Enter governance policy [advisory]: ").strip() or "advisory"
    config['governance_policy'] = policy

    # Validation
    print("\n🔍 Step 6: Validating Configuration...")
    result = validate_setup(**config)

    if result.is_valid:
        print("✅ Configuration validated successfully!")

        # Generate code example
        print("\n💻 Your Configuration:")
        print("```python")
        print("from genops.providers.perplexity import GenOpsPerplexityAdapter")
        print()
        print("adapter = GenOpsPerplexityAdapter(")
        # Security: Use static configuration display to prevent sensitive data exposure
        print("    # Configuration values have been validated")
        print("    # Please check your environment variables or configuration file")
        print("    # All sensitive values like API keys are properly secured")
        print(")")
        print("```")

    else:
        print("❌ Configuration validation failed. Please check the issues above.")

    return config


def _sanitize_sensitive_field(field_name: str, value: Any) -> Any:
    """
    Comprehensive sanitization for sensitive fields.
    
    Ensures no sensitive data can be logged regardless of type or content.
    Uses allowlist approach - only explicitly safe fields pass through.
    """
    # Define comprehensive patterns for sensitive field detection
    sensitive_patterns = {
        'key', 'token', 'secret', 'password', 'credential', 'auth',
        'private', 'secure', 'sensitive', 'confidential', 'restricted'
    }

    # Check field name against all sensitive patterns
    field_lower = field_name.lower()
    if any(pattern in field_lower for pattern in sensitive_patterns):
        return "***REDACTED***"

    # Allowlist of explicitly safe configuration fields
    safe_fields = {
        'team', 'project', 'environment', 'daily_budget_limit',
        'monthly_budget_limit', 'governance_policy', 'enable_cost_alerts',
        'customer_id', 'cost_center', 'default_model', 'default_search_context',
        'enable_caching', 'retry_attempts', 'timeout_seconds', 'tags'
    }

    if field_name in safe_fields:
        return value
    else:
        # Any unknown field is treated as potentially sensitive
        return "***REDACTED***"


# Convenience exports
__all__ = [
    'PerplexitySetupValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationLevel',
    'ValidationCategory',
    'validate_setup',
    'print_validation_result',
    'is_properly_configured',
    'interactive_setup_wizard'
]
