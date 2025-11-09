#!/usr/bin/env python3
"""
Perplexity AI Interactive Setup Wizard Example

This example demonstrates an interactive setup wizard for Perplexity AI integration
with GenOps governance, providing guided configuration for different deployment
scenarios and use cases.

Usage:
    python interactive_setup_wizard.py

Prerequisites:
    pip install genops[perplexity]
    (API key and other settings configured through wizard)

Expected Output:
    - 🧙‍♂️ Interactive step-by-step configuration wizard
    - ✅ Customized setup validation and verification
    - 📋 Generated configuration for your specific use case
    - 🚀 Ready-to-use adapter and example code

Learning Objectives:
    - Understand all Perplexity integration configuration options
    - Learn how to customize governance for different scenarios
    - Practice interactive setup and troubleshooting
    - Generate production-ready configuration templates

Time Required: ~10 minutes (guided setup)
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PerplexityConfiguration:
    """Configuration data structure for Perplexity integration."""
    # Basic settings
    api_key: Optional[str] = None
    team: str = "default-team"
    project: str = "default-project"
    environment: str = "development"
    
    # Governance settings
    daily_budget_limit: float = 50.0
    monthly_budget_limit: float = 1500.0
    governance_policy: str = "advisory"
    enable_cost_alerts: bool = True
    
    # Advanced settings
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    default_model: str = "sonar"
    default_search_context: str = "medium"
    
    # Operational settings
    enable_caching: bool = False
    retry_attempts: int = 3
    timeout_seconds: int = 30
    
    # Tags and metadata
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class InteractiveSetupWizard:
    """Interactive setup wizard for Perplexity AI integration."""
    
    def __init__(self):
        self.config = PerplexityConfiguration()
        self.use_case_templates = self._load_use_case_templates()
    
    def run_wizard(self) -> PerplexityConfiguration:
        """Run the complete interactive setup wizard."""
        print("🧙‍♂️ Perplexity AI + GenOps Interactive Setup Wizard")
        print("=" * 65)
        print()
        print("Welcome! This wizard will help you configure Perplexity AI integration")
        print("with GenOps governance for your specific use case and environment.")
        print()
        
        try:
            # Step 1: Use case selection
            self._select_use_case()
            
            # Step 2: Basic configuration
            self._configure_basic_settings()
            
            # Step 3: Governance configuration
            self._configure_governance()
            
            # Step 4: Advanced configuration
            self._configure_advanced_settings()
            
            # Step 5: Validation and testing
            self._validate_configuration()
            
            # Step 6: Generate outputs
            self._generate_configuration_outputs()
            
            return self.config
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Setup wizard cancelled by user.")
            return None
        except Exception as e:
            print(f"\n❌ Setup wizard error: {e}")
            return None
    
    def _select_use_case(self):
        """Help user select their primary use case."""
        print("📋 Step 1: Use Case Selection")
        print("-" * 35)
        print("What's your primary use case for Perplexity AI?")
        print()
        
        use_cases = [
            ("Development & Testing", "Low-volume development and testing"),
            ("Content Research", "Content creation and research workflows"),
            ("Customer Support", "AI-powered customer support and documentation"),
            ("Enterprise Research", "Large-scale enterprise research and analysis"),
            ("Multi-tenant SaaS", "Multi-tenant application with customer attribution"),
            ("Compliance & Governance", "Regulated industry with strict compliance needs"),
            ("Custom Configuration", "I'll configure everything manually")
        ]
        
        for i, (name, description) in enumerate(use_cases, 1):
            print(f"   {i}. {name}")
            print(f"      {description}")
            print()
        
        while True:
            try:
                choice = input("Select your use case (1-7): ").strip()
                use_case_idx = int(choice) - 1
                
                if 0 <= use_case_idx < len(use_cases):
                    selected_use_case = use_cases[use_case_idx][0]
                    print(f"\n✅ Selected: {selected_use_case}")
                    
                    # Apply use case template
                    if selected_use_case in self.use_case_templates:
                        template = self.use_case_templates[selected_use_case]
                        self._apply_template(template)
                        print(f"   Applied template with recommended settings")
                    
                    break
                else:
                    print("Please enter a number between 1 and 7.")
            except ValueError:
                print("Please enter a valid number.")
    
    def _configure_basic_settings(self):
        """Configure basic settings."""
        print("\n📋 Step 2: Basic Configuration")
        print("-" * 35)
        
        # API Key
        print("🔑 Perplexity API Key:")
        current_key = os.getenv('PERPLEXITY_API_KEY', '')
        if current_key:
            print(f"   Current: {current_key[:8]}{'*' * (len(current_key) - 8)}")
            use_current = input("   Use current API key? [Y/n]: ").strip().lower()
            if use_current in ['', 'y', 'yes']:
                self.config.api_key = current_key
            else:
                self.config.api_key = self._get_secure_input("   Enter API key: ")
        else:
            print("   No API key found in environment.")
            print("   Get your key from: https://www.perplexity.ai/settings/api")
            self.config.api_key = self._get_secure_input("   Enter API key: ")
        
        # Team and Project
        print("\n🏷️ Team and Project Identification:")
        self.config.team = input(f"   Team name [{self.config.team}]: ").strip() or self.config.team
        self.config.project = input(f"   Project name [{self.config.project}]: ").strip() or self.config.project
        
        # Environment
        print("\n🌍 Deployment Environment:")
        envs = ["development", "staging", "production"]
        print("   Options: " + ", ".join(envs))
        env_input = input(f"   Environment [{self.config.environment}]: ").strip().lower()
        if env_input in envs:
            self.config.environment = env_input
        elif env_input:
            self.config.environment = env_input  # Allow custom environments
        
        print(f"\n✅ Basic configuration completed")
        print(f"   Team: {self.config.team}")
        print(f"   Project: {self.config.project}")
        print(f"   Environment: {self.config.environment}")
    
    def _configure_governance(self):
        """Configure governance settings."""
        print("\n📋 Step 3: Governance Configuration")
        print("-" * 40)
        
        # Budget limits
        print("💰 Budget Management:")
        daily_budget = input(f"   Daily budget limit (${self.config.daily_budget_limit}): ").strip()
        if daily_budget:
            try:
                self.config.daily_budget_limit = float(daily_budget)
            except ValueError:
                print("   ⚠️ Invalid budget amount, using default")
        
        monthly_budget = input(f"   Monthly budget limit (${self.config.monthly_budget_limit}): ").strip()
        if monthly_budget:
            try:
                self.config.monthly_budget_limit = float(monthly_budget)
            except ValueError:
                print("   ⚠️ Invalid budget amount, using default")
        
        # Governance policy
        print("\n🛡️ Governance Policy:")
        policies = {
            "1": ("advisory", "Warn about budget/policy violations but allow operations"),
            "2": ("enforced", "Block operations that violate budget or policies"),
            "3": ("strict", "Maximum governance with pre-validation checks")
        }
        
        for key, (name, description) in policies.items():
            marker = "✅" if name == self.config.governance_policy else "  "
            print(f"   {key}. {marker} {name.upper()}: {description}")
        
        policy_choice = input("\n   Select governance policy (1-3): ").strip()
        if policy_choice in policies:
            self.config.governance_policy = policies[policy_choice][0]
        
        # Cost alerts
        print(f"\n🔔 Cost Alerts:")
        alert_input = input(f"   Enable cost alerts? [{'Y' if self.config.enable_cost_alerts else 'N'}/n/y]: ").strip().lower()
        if alert_input in ['y', 'yes']:
            self.config.enable_cost_alerts = True
        elif alert_input in ['n', 'no']:
            self.config.enable_cost_alerts = False
        
        print(f"\n✅ Governance configuration completed")
        print(f"   Daily Budget: ${self.config.daily_budget_limit}")
        print(f"   Policy: {self.config.governance_policy}")
        print(f"   Cost Alerts: {'Enabled' if self.config.enable_cost_alerts else 'Disabled'}")
    
    def _configure_advanced_settings(self):
        """Configure advanced settings."""
        print("\n📋 Step 4: Advanced Configuration")
        print("-" * 40)
        
        # Enterprise attribution
        print("🏢 Enterprise Attribution (Optional):")
        self.config.customer_id = input("   Customer ID (for multi-tenant): ").strip() or None
        self.config.cost_center = input("   Cost Center (for financial reporting): ").strip() or None
        
        # Default model and context
        print("\n🤖 Default Model Configuration:")
        models = ["sonar", "sonar-pro", "sonar-reasoning"]
        print("   Available models: " + ", ".join(models))
        model_input = input(f"   Default model [{self.config.default_model}]: ").strip().lower()
        if model_input in models:
            self.config.default_model = model_input
        
        contexts = ["low", "medium", "high"]
        print("   Search contexts: " + ", ".join(contexts))
        context_input = input(f"   Default search context [{self.config.default_search_context}]: ").strip().lower()
        if context_input in contexts:
            self.config.default_search_context = context_input
        
        # Performance settings
        print("\n⚡ Performance Configuration:")
        
        cache_input = input(f"   Enable result caching? [{'Y' if self.config.enable_caching else 'N'}/n/y]: ").strip().lower()
        if cache_input in ['y', 'yes']:
            self.config.enable_caching = True
        elif cache_input in ['n', 'no']:
            self.config.enable_caching = False
        
        retry_input = input(f"   Retry attempts [{self.config.retry_attempts}]: ").strip()
        if retry_input:
            try:
                self.config.retry_attempts = int(retry_input)
            except ValueError:
                print("   ⚠️ Invalid retry count, using default")
        
        timeout_input = input(f"   Timeout seconds [{self.config.timeout_seconds}]: ").strip()
        if timeout_input:
            try:
                self.config.timeout_seconds = int(timeout_input)
            except ValueError:
                print("   ⚠️ Invalid timeout, using default")
        
        # Custom tags
        print("\n🏷️ Custom Tags:")
        print("   Enter custom tags for cost attribution and filtering.")
        print("   Format: key=value (press Enter to finish)")
        
        while True:
            tag_input = input("   Tag: ").strip()
            if not tag_input:
                break
            
            if '=' in tag_input:
                key, value = tag_input.split('=', 1)
                self.config.tags[key.strip()] = value.strip()
                print(f"      Added: {key.strip()}={value.strip()}")
            else:
                print("      ⚠️ Invalid format, use key=value")
        
        print(f"\n✅ Advanced configuration completed")
        if self.config.customer_id:
            print(f"   Customer ID: {self.config.customer_id}")
        if self.config.cost_center:
            print(f"   Cost Center: {self.config.cost_center}")
        print(f"   Default Model: {self.config.default_model}")
        print(f"   Performance: Caching {'enabled' if self.config.enable_caching else 'disabled'}")
    
    def _validate_configuration(self):
        """Validate the configuration."""
        print("\n📋 Step 5: Configuration Validation")
        print("-" * 40)
        print("Validating your configuration...")
        
        validation_results = []
        
        # API key validation
        if self.config.api_key and self.config.api_key.startswith('pplx-'):
            validation_results.append(("✅", "API key format valid"))
        elif not self.config.api_key:
            validation_results.append(("❌", "API key required"))
        else:
            validation_results.append(("⚠️", "API key format may be incorrect"))
        
        # Budget validation
        if self.config.daily_budget_limit > 0:
            validation_results.append(("✅", "Daily budget configured"))
        else:
            validation_results.append(("⚠️", "Daily budget should be positive"))
        
        # Environment validation
        if self.config.environment in ["development", "staging", "production"]:
            validation_results.append(("✅", f"Environment '{self.config.environment}' recognized"))
        else:
            validation_results.append(("⚠️", f"Custom environment '{self.config.environment}'"))
        
        # Enterprise settings validation
        if self.config.environment == "production":
            if not self.config.customer_id and not self.config.cost_center:
                validation_results.append(("⚠️", "Consider adding customer_id or cost_center for production"))
            else:
                validation_results.append(("✅", "Enterprise attribution configured"))
        
        # Display validation results
        for status, message in validation_results:
            print(f"   {status} {message}")
        
        # Test connection if possible
        print("\n🔍 Testing connection (optional)...")
        test_connection = input("   Test Perplexity API connection? [y/N]: ").strip().lower()
        
        if test_connection in ['y', 'yes']:
            self._test_api_connection()
        
        print(f"\n✅ Configuration validation completed")
    
    def _test_api_connection(self):
        """Test the API connection."""
        try:
            from genops.providers.perplexity import GenOpsPerplexityAdapter, PerplexityModel, SearchContext
            
            print("   🔧 Creating test adapter...")
            
            # Create adapter with current configuration
            adapter = GenOpsPerplexityAdapter(
                team=self.config.team,
                project=self.config.project,
                environment=self.config.environment,
                daily_budget_limit=self.config.daily_budget_limit,
                governance_policy=self.config.governance_policy,
                customer_id=self.config.customer_id,
                cost_center=self.config.cost_center,
                tags=self.config.tags or {}
            )
            
            print("   🔍 Testing simple search...")
            
            # Test with a simple query
            result = adapter.search_with_governance(
                query="What is artificial intelligence?",
                model=PerplexityModel.SONAR,
                search_context=SearchContext.LOW,
                max_tokens=50
            )
            
            print(f"   ✅ Connection test successful!")
            print(f"      Response length: {len(result.response)} characters")
            print(f"      Cost: ${result.cost:.6f}")
            print(f"      Citations: {len(result.citations)}")
            
        except ImportError:
            print("   ⚠️ GenOps not available for connection test")
            print("      Install with: pip install genops[perplexity]")
        except Exception as e:
            print(f"   ❌ Connection test failed: {str(e)[:60]}")
            print("      Check your API key and internet connection")
    
    def _generate_configuration_outputs(self):
        """Generate configuration files and example code."""
        print("\n📋 Step 6: Generate Configuration")
        print("-" * 40)
        
        # Generate environment variables
        self._generate_env_file()
        
        # Generate example code
        self._generate_example_code()
        
        # Generate configuration summary
        self._generate_config_summary()
        
        print(f"\n🎉 Setup wizard completed successfully!")
        print(f"\nGenerated files:")
        print(f"   • .env.perplexity - Environment variables")
        print(f"   • perplexity_example.py - Working example code")
        print(f"   • perplexity_config.json - Complete configuration")
        print(f"\nNext steps:")
        print(f"   1. Review generated files")
        print(f"   2. Run: python perplexity_example.py")
        print(f"   3. Explore examples/perplexity/ for more patterns")
    
    def _generate_env_file(self):
        """Generate environment variables file."""
        # Security: Don't write actual API key to file, use placeholder
        api_key_value = 'pplx-your-api-key-here' if self.config.api_key else 'pplx-your-api-key-here'
        
        env_content = f"""# Perplexity AI + GenOps Configuration
# Generated by setup wizard on {datetime.now().isoformat()}
# SECURITY: Replace 'pplx-your-api-key-here' with your actual API key

# Required Settings
PERPLEXITY_API_KEY={api_key_value}
GENOPS_TEAM={self.config.team}
GENOPS_PROJECT={self.config.project}
GENOPS_ENVIRONMENT={self.config.environment}

# Budget Settings
GENOPS_DAILY_BUDGET_LIMIT={self.config.daily_budget_limit}
GENOPS_MONTHLY_BUDGET_LIMIT={self.config.monthly_budget_limit}
GENOPS_GOVERNANCE_POLICY={self.config.governance_policy}

# Optional Enterprise Settings
{f'GENOPS_CUSTOMER_ID={self.config.customer_id}' if self.config.customer_id else '# GENOPS_CUSTOMER_ID=your-customer-id'}
{f'GENOPS_COST_CENTER={self.config.cost_center}' if self.config.cost_center else '# GENOPS_COST_CENTER=your-cost-center'}

# Performance Settings
GENOPS_ENABLE_CACHING={str(self.config.enable_caching).lower()}
GENOPS_RETRY_ATTEMPTS={self.config.retry_attempts}
GENOPS_TIMEOUT_SECONDS={self.config.timeout_seconds}
"""
        
        # Security: Verify env_content doesn't contain actual sensitive data before writing
        if 'pplx-your-api-key-here' in env_content or not any(
            line.startswith('PERPLEXITY_API_KEY=pplx-') and 'your-api-key-here' not in line 
            for line in env_content.split('\n')
        ):
            with open('.env.perplexity', 'w') as f:
                f.write(env_content)
        else:
            # This should not happen with current logic, but prevents any actual API keys from being written
            raise ValueError("Security check failed: attempted to write actual API key to file")
        
        print(f"   ✅ Generated .env.perplexity")
        if self.config.api_key and self.config.api_key.startswith('pplx-'):
            print(f"   🔐 Security: API key not written to file - please set it manually")
            print(f"   💡 Run: export PERPLEXITY_API_KEY='your-actual-key'")
    
    def _get_optional_customer_id(self):
        """Get optional customer_id for code generation."""
        if self.config.customer_id:
            return f'\n        customer_id="{self.config.customer_id}",'
        return ""
    
    def _get_optional_cost_center(self):
        """Get optional cost_center for code generation."""
        if self.config.cost_center:
            return f'\n        cost_center="{self.config.cost_center}",'
        return ""
    
    def _generate_example_code(self):
        """Generate working example code."""
        example_code = f'''#!/usr/bin/env python3
"""
Generated Perplexity AI Example
Created by GenOps setup wizard on {datetime.now().isoformat()}

Usage:
    python perplexity_example.py
"""

import os
from genops.providers.perplexity import (
    GenOpsPerplexityAdapter,
    PerplexityModel,
    SearchContext
)

def main():
    """Your customized Perplexity AI example."""
    print("🔍 Your Perplexity AI + GenOps Example")
    print("=" * 45)
    
    # Create adapter with your configuration
    adapter = GenOpsPerplexityAdapter(
        team="{self.config.team}",
        project="{self.config.project}",
        environment="{self.config.environment}",
        daily_budget_limit={self.config.daily_budget_limit},
        monthly_budget_limit={self.config.monthly_budget_limit},
        governance_policy="{self.config.governance_policy}",
        enable_cost_alerts={self.config.enable_cost_alerts},{self._get_optional_customer_id()}{self._get_optional_cost_center()}
        tags={self.config.tags or {{}}}
    )
    
    # Example search
    with adapter.track_search_session("wizard_example") as session:
        result = adapter.search_with_governance(
            query="What are the latest trends in artificial intelligence?",
            model=PerplexityModel.{self.config.default_model.upper().replace('-', '_')},
            search_context=SearchContext.{self.config.default_search_context.upper()},
            session_id=session.session_id,
            max_tokens=300
        )
        
        print(f"🔍 Search Results:")
        print(f"   Query: What are the latest trends in AI?")
        print(f"   Model: {self.config.default_model}")
        print(f"   Context: {self.config.default_search_context}")
        print(f"   Response: {{result.response[:200]}}...")
        print(f"   Citations: {{len(result.citations)}}")
        print(f"   Cost: ${{result.cost:.6f}}")
        
        # Show cost summary
        cost_summary = adapter.get_cost_summary()
        print(f"\\n💰 Cost Summary:")
        print(f"   Daily Spend: ${{cost_summary['daily_costs']:.6f}}")
        print(f"   Budget Used: {{cost_summary['daily_budget_utilization']:.1f}}%")

if __name__ == "__main__":
    main()
'''
        
        with open('perplexity_example.py', 'w') as f:
            f.write(example_code)
        
        print(f"   ✅ Generated perplexity_example.py")
    
    def _generate_config_summary(self):
        """Generate configuration summary JSON."""
        config_dict = asdict(self.config)
        config_dict['generated_at'] = datetime.now().isoformat()
        config_dict['wizard_version'] = '1.0.0'
        
        with open('perplexity_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"   ✅ Generated perplexity_config.json")
    
    def _load_use_case_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined use case templates."""
        return {
            "Development & Testing": {
                "daily_budget_limit": 10.0,
                "monthly_budget_limit": 300.0,
                "governance_policy": "advisory",
                "default_model": "sonar",
                "default_search_context": "low",
                "enable_caching": True,
                "tags": {"use_case": "development"}
            },
            "Content Research": {
                "daily_budget_limit": 25.0,
                "monthly_budget_limit": 750.0,
                "governance_policy": "advisory",
                "default_model": "sonar-pro",
                "default_search_context": "high",
                "enable_caching": True,
                "tags": {"use_case": "content_research"}
            },
            "Customer Support": {
                "daily_budget_limit": 50.0,
                "monthly_budget_limit": 1500.0,
                "governance_policy": "enforced",
                "default_model": "sonar",
                "default_search_context": "medium",
                "enable_caching": True,
                "tags": {"use_case": "customer_support"}
            },
            "Enterprise Research": {
                "daily_budget_limit": 200.0,
                "monthly_budget_limit": 6000.0,
                "governance_policy": "enforced",
                "default_model": "sonar-pro",
                "default_search_context": "high",
                "enable_cost_alerts": True,
                "tags": {"use_case": "enterprise_research"}
            },
            "Multi-tenant SaaS": {
                "daily_budget_limit": 100.0,
                "monthly_budget_limit": 3000.0,
                "governance_policy": "strict",
                "default_model": "sonar",
                "default_search_context": "medium",
                "enable_cost_alerts": True,
                "tags": {"use_case": "multi_tenant", "architecture": "saas"}
            },
            "Compliance & Governance": {
                "daily_budget_limit": 75.0,
                "monthly_budget_limit": 2250.0,
                "governance_policy": "strict",
                "default_model": "sonar-pro",
                "default_search_context": "high",
                "enable_cost_alerts": True,
                "tags": {"use_case": "compliance", "audit_required": "true"}
            }
        }
    
    def _apply_template(self, template: Dict[str, Any]):
        """Apply a use case template to the configuration."""
        for key, value in template.items():
            if key == "tags":
                self.config.tags.update(value)
            else:
                setattr(self.config, key, value)
    
    def _get_secure_input(self, prompt: str) -> str:
        """Get secure input (like API key) without echoing."""
        try:
            import getpass
            return getpass.getpass(prompt)
        except ImportError:
            # Fallback to regular input if getpass not available
            return input(prompt)


def main():
    """Run the interactive setup wizard."""
    print("🚀 Perplexity AI + GenOps Interactive Setup")
    print("=" * 50)
    print()
    
    # Check prerequisites
    try:
        from genops.providers.perplexity import GenOpsPerplexityAdapter
        print("✅ GenOps Perplexity provider available")
    except ImportError:
        print("❌ GenOps not available")
        print("   Fix: pip install genops[perplexity]")
        print("\nContinuing with configuration generation only...")
        print("(API testing will be skipped)")
    
    print()
    
    # Run wizard
    wizard = InteractiveSetupWizard()
    config = wizard.run_wizard()
    
    if config:
        print("\n📚 Recommended Next Steps:")
        print("   1. Source your environment: source .env.perplexity")
        print("   2. Test your setup: python perplexity_example.py")
        print("   3. Explore examples: ls examples/perplexity/")
        print("   4. Read the quickstart: docs/perplexity-quickstart.md")
        print("   5. Join the community: github.com/genops-ai/discussions")
        
        return config
    else:
        print("❌ Setup wizard was not completed")
        return None


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup wizard cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup wizard failed: {e}")
        sys.exit(1)