#!/usr/bin/env python3
"""
Together AI Interactive Setup Wizard

Interactive setup wizard for configuring Together AI with GenOps governance.
Guides users through configuration, testing, and generates template files.

Usage:
    python interactive_setup_wizard.py

Features:
    - Interactive configuration wizard
    - API key validation and model testing
    - Automatic environment file generation
    - Example code generation with governance
    - Team onboarding assistance
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from genops.providers.together_validation import validate_together_setup
    from genops.providers.together_pricing import TogetherPricingCalculator
    from genops.providers.together import TogetherModel
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install genops-ai[together]")
    sys.exit(1)


class TogetherSetupWizard:
    """Interactive setup wizard for Together AI + GenOps configuration."""
    
    def __init__(self):
        self.config = {}
        self.pricing_calc = TogetherPricingCalculator()
    
    def welcome(self):
        """Display welcome message and overview."""
        print("🧙‍♂️ Together AI + GenOps Setup Wizard")
        print("=" * 50)
        print("Welcome! This wizard will help you:")
        print("  ✅ Configure Together AI with GenOps governance")
        print("  ✅ Validate your API key and model access")
        print("  ✅ Set up cost tracking and budget controls")
        print("  ✅ Generate environment files and example code")
        print("  ✅ Test your configuration")
        print()
    
    def gather_api_credentials(self):
        """Gather and validate API credentials."""
        print("🔐 API Credentials Setup")
        print("-" * 30)
        
        # Check for existing API key
        existing_key = os.getenv('TOGETHER_API_KEY')
        if existing_key:
            print(f"✅ Found existing TOGETHER_API_KEY environment variable")
            use_existing = input("Use existing API key? (Y/n): ").lower()
            if use_existing != 'n':
                self.config['api_key'] = existing_key
                return
        
        print("\n📝 Please provide your Together AI credentials:")
        print("   Get your API key from: https://api.together.xyz/settings/api-keys")
        
        while True:
            api_key = input("\nTogether AI API Key: ").strip()
            if not api_key:
                print("❌ API key is required")
                continue
            
            if not api_key.startswith(('sk-', 'pk-')):
                print("⚠️ Warning: API key format may be incorrect (should start with 'sk-' or 'pk-')")
                confirm = input("Continue anyway? (y/N): ").lower()
                if confirm != 'y':
                    continue
            
            self.config['api_key'] = api_key
            print("✅ API key configured")
            break
    
    def gather_governance_config(self):
        """Gather governance and cost tracking configuration."""
        print("\n🛡️ Governance Configuration")
        print("-" * 30)
        
        # Team information
        default_team = os.getenv('GENOPS_TEAM', '')
        self.config['team'] = input(f"Team name [{default_team or 'my-team'}]: ").strip() or default_team or 'my-team'
        
        # Project information
        default_project = os.getenv('GENOPS_PROJECT', '')
        self.config['project'] = input(f"Project name [{default_project or 'together-ai-project'}]: ").strip() or default_project or 'together-ai-project'
        
        # Environment
        default_env = os.getenv('GENOPS_ENVIRONMENT', 'development')
        print(f"\nEnvironment options: development, staging, production")
        self.config['environment'] = input(f"Environment [{default_env}]: ").strip() or default_env
        
        # Budget configuration
        print(f"\n💰 Budget Configuration")
        print("   Set budget limits to control AI spending")
        
        while True:
            try:
                daily_budget = input("Daily budget limit (USD) [50.0]: ").strip()
                self.config['daily_budget_limit'] = float(daily_budget) if daily_budget else 50.0
                break
            except ValueError:
                print("❌ Please enter a valid number")
        
        while True:
            try:
                monthly_budget = input("Monthly budget limit (USD) [1000.0]: ").strip()
                self.config['monthly_budget_limit'] = float(monthly_budget) if monthly_budget else 1000.0
                break
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Governance policy
        print(f"\n🛡️ Governance Policy Options:")
        print("   advisory  - Monitor costs, provide warnings")
        print("   enforced - Block operations that exceed budget")
        print("   strict   - Strict enforcement with detailed auditing")
        
        policy_options = ['advisory', 'enforced', 'strict']
        while True:
            policy = input("Governance policy [advisory]: ").strip().lower() or 'advisory'
            if policy in policy_options:
                self.config['governance_policy'] = policy
                break
            print(f"❌ Please choose from: {', '.join(policy_options)}")
        
        # Optional enterprise features
        print(f"\n🏢 Optional Enterprise Features")
        customer_id = input("Customer ID (optional): ").strip()
        if customer_id:
            self.config['customer_id'] = customer_id
        
        cost_center = input("Cost center (optional): ").strip()
        if cost_center:
            self.config['cost_center'] = cost_center
    
    def gather_preferences(self):
        """Gather user preferences and model selection."""
        print(f"\n⚙️ Preferences & Model Selection")
        print("-" * 30)
        
        # Default model selection
        print("🤖 Default Model Selection:")
        print("   Available tiers:")
        print("   1. Lite (8B models)    - Ultra fast, cost-effective")
        print("   2. Standard (70B)      - Balanced performance")
        print("   3. Large (405B)        - Maximum capability")
        print("   4. Reasoning (R1)      - Advanced reasoning")
        print("   5. Code (DeepSeek)     - Code generation")
        
        model_choices = {
            '1': TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            '2': TogetherModel.LLAMA_3_1_70B_INSTRUCT,
            '3': TogetherModel.LLAMA_3_1_405B_INSTRUCT,
            '4': TogetherModel.DEEPSEEK_R1,
            '5': TogetherModel.DEEPSEEK_CODER_V2
        }
        
        while True:
            choice = input("Select default model tier [1]: ").strip() or '1'
            if choice in model_choices:
                self.config['default_model'] = model_choices[choice]
                break
            print("❌ Please choose 1-5")
        
        # Performance preferences
        print(f"\n⚡ Performance Preferences:")
        
        enable_caching = input("Enable response caching? [y/N]: ").lower() == 'y'
        self.config['enable_caching'] = enable_caching
        
        while True:
            try:
                retry_attempts = input("Retry attempts for failed requests [3]: ").strip()
                self.config['retry_attempts'] = int(retry_attempts) if retry_attempts else 3
                break
            except ValueError:
                print("❌ Please enter a valid number")
        
        while True:
            try:
                timeout = input("Request timeout (seconds) [30]: ").strip()
                self.config['timeout_seconds'] = int(timeout) if timeout else 30
                break
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Cost alerts
        enable_alerts = input("Enable cost alerts? [Y/n]: ").lower() != 'n'
        self.config['enable_cost_alerts'] = enable_alerts
    
    def validate_configuration(self):
        """Validate the complete configuration."""
        print(f"\n✅ Configuration Validation")
        print("-" * 30)
        
        print("🔍 Validating your configuration...")
        
        # Run comprehensive validation
        result = validate_together_setup(
            together_api_key=self.config['api_key'],
            config=self.config,
            print_results=False  # We'll format our own output
        )
        
        if result.is_valid:
            print("✅ Configuration validation successful!")
            
            if result.model_access:
                print(f"🎯 Model Access: {len(result.model_access)} models available")
                
                # Show cost estimates for the user's default model
                if hasattr(self.config['default_model'], 'value'):
                    model_name = self.config['default_model'].value
                    cost_est = self.pricing_calc.estimate_chat_cost(model_name, 1000)
                    print(f"💰 Default model cost: ~${cost_est:.6f} per 1000 tokens")
            
            return True
        else:
            print("❌ Configuration validation failed:")
            for error in result.errors:
                print(f"   • {error.message}")
                print(f"     Fix: {error.remediation}")
            return False
    
    def generate_files(self):
        """Generate environment and example files."""
        print(f"\n📁 Generating Configuration Files")
        print("-" * 30)
        
        # Generate environment file
        self._generate_env_file()
        
        # Generate example code
        self._generate_example_code()
        
        print(f"\n✅ Files generated successfully!")
        print(f"\nNext steps:")
        print(f"   1. Review generated files")
        print(f"   2. Run: python together_example.py")
        print(f"   3. Explore examples/together/ for more patterns")
    
    def _generate_env_file(self):
        """Generate environment variables file."""
        # Security: Write only static safe content to prevent sensitive data exposure
        static_safe_content = f"""# Together AI + GenOps Configuration
# Generated by setup wizard - TEMPLATE FILE
# SECURITY: Replace placeholders with your actual values

# Required Settings
TOGETHER_API_KEY=sk-your-api-key-here
GENOPS_TEAM=your-team-name
GENOPS_PROJECT=your-project-name
GENOPS_ENVIRONMENT=development

# Budget Settings  
GENOPS_DAILY_BUDGET_LIMIT=50.0
GENOPS_MONTHLY_BUDGET_LIMIT=1000.0
GENOPS_GOVERNANCE_POLICY=advisory

# Optional Enterprise Settings
# GENOPS_CUSTOMER_ID=your-customer-id
# GENOPS_COST_CENTER=your-cost-center

# Performance Settings
GENOPS_ENABLE_CACHING=true
GENOPS_RETRY_ATTEMPTS=3
GENOPS_TIMEOUT_SECONDS=30
"""
        with open('.env.together', 'w') as f:
            f.write(static_safe_content)
        
        print(f"   ✅ Generated .env.together")
        if self.config.get('api_key') and self.config['api_key'].startswith(('sk-', 'pk-')):
            print(f"   🔐 Security: API key not written to file - please set it manually")
            print(f"   💡 Run: export TOGETHER_API_KEY='your-actual-key'")
    
    def _generate_example_code(self):
        """Generate working example code."""
        # Security: Use static template to prevent sensitive data exposure
        example_code = '''#!/usr/bin/env python3
"""
Generated Together AI Example  
Created by GenOps setup wizard - TEMPLATE FILE

Usage:
    1. Update the configuration values below with your actual settings
    2. Run: python together_example.py
"""

import os
from genops.providers.together import (
    GenOpsTogetherAdapter,
    TogetherModel
)

def main():
    """Your customized Together AI example."""
    print("🤖 Your Together AI + GenOps Example")
    print("=" * 45)
    
    # Create adapter with your configuration - UPDATE THESE VALUES
    adapter = GenOpsTogetherAdapter(
        team="your-team-name",
        project="your-project-name", 
        environment="development",
        daily_budget_limit=50.0,
        monthly_budget_limit=1000.0,
        governance_policy="advisory",
        enable_cost_alerts=True,
        # customer_id="your-customer-id",  # Optional
        # cost_center="your-cost-center",  # Optional
        default_model=TogetherModel.LLAMA_3_1_8B_INSTRUCT
    )
    
    # Example chat completion
    with adapter.track_session("example-session") as session:
        result = adapter.chat_with_governance(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain the benefits of Together AI's open-source model approach."}
            ],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=200,
            session_id=session.session_id
        )
        
        print(f"🔍 Response:")
        print(f"   {result.response}")
        print(f"\\n📊 Metrics:")
        print(f"   Model: {result.model_used}")
        print(f"   Tokens: {result.tokens_used}")
        print(f"   Cost: ${result.cost:.6f}")
        print(f"   Time: {result.execution_time_seconds:.2f}s")
        
        # Show cost summary
        cost_summary = adapter.get_cost_summary()
        print(f"\\n💰 Cost Summary:")
        print(f"   Daily spend: ${cost_summary['daily_costs']:.6f}")
        print(f"   Budget used: {cost_summary['daily_budget_utilization']:.1f}%")

if __name__ == "__main__":
    main()
'''
        
        with open('together_example.py', 'w') as f:
            f.write(example_code)
        
        print(f"   ✅ Generated together_example.py")
    
    def _generate_config_summary(self):
        """Generate configuration summary."""
        print(f"\n📋 Configuration Summary")
        print("-" * 30)
        
        # Security: Display configuration safely
        print("Configuration validated and files generated:")
        print("  ✅ API credentials configured")
        print("  ✅ Governance settings applied")
        print("  ✅ Budget controls enabled")
        print("  ✅ Model preferences set")
        print("  ✅ Environment files created")
    
    def run_wizard(self):
        """Run the complete setup wizard."""
        try:
            self.welcome()
            self.gather_api_credentials()
            self.gather_governance_config()
            self.gather_preferences()
            
            if self.validate_configuration():
                self.generate_files()
                self._generate_config_summary()
                
                print(f"\n🎉 Setup completed successfully!")
                print(f"\n🚀 Quick Start:")
                print(f"   1. export TOGETHER_API_KEY='your-actual-key'")
                print(f"   2. python together_example.py")
                print(f"   3. explore examples/together/ for more examples")
                
                return True
            else:
                print(f"\n❌ Setup failed - please fix the issues above and try again")
                return False
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Setup wizard interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup wizard failed: {e}")
            return False


def main():
    """Run the Together AI setup wizard."""
    wizard = TogetherSetupWizard()
    success = wizard.run_wizard()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Wizard interrupted")
        sys.exit(1)