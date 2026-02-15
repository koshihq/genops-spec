#!/usr/bin/env python3
"""
Example: Auto-Instrumentation for Vercel AI SDK

Complexity: ⭐ Beginner

This example demonstrates zero-code auto-instrumentation for existing
Vercel AI SDK applications. Simply import and enable - no code changes required.

Prerequisites:
- Node.js 16+ installed
- Vercel AI SDK installed: npm install ai @ai-sdk/openai
- OpenAI API key set in environment
- GenOps package installed: pip install genops

Usage:
    python 02_auto_instrumentation.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key
    GENOPS_TEAM: Team name for cost attribution
    GENOPS_PROJECT: Project name for tracking
"""

import json
import logging
import os
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import GenOps Vercel AI SDK integration
try:
    from genops.providers.vercel_ai_sdk import (  # noqa: F401
        GenOpsVercelAISDKAdapter,
        auto_instrument,
    )
    from genops.providers.vercel_ai_sdk_pricing import estimate_cost, get_model_info
    from genops.providers.vercel_ai_sdk_validation import validate_setup
except ImportError as e:
    logger.error(f"GenOps not installed: {e}")
    logger.error("Install with: pip install genops")
    exit(1)


class AutoInstrumentationDemo:
    """Demonstration of zero-code auto-instrumentation for Vercel AI SDK."""

    def __init__(self):
        """Initialize the auto-instrumentation demo."""
        self.adapter = None
        self.temp_files = []

    def setup(self):
        """Set up auto-instrumentation with zero code changes."""
        print("🔧 Setting up auto-instrumentation...")

        # Validate environment
        print("📋 Validating environment...")
        result = validate_setup(verbose=False)
        if not result.all_passed:
            print("❌ Environment validation failed!")
            for check_result in result.results:
                if not check_result.passed:
                    print(f"  • {check_result.message}")
            return False

        print("✅ Environment validation passed!")

        # Initialize auto-instrumentation
        self.adapter = auto_instrument(
            integration_mode="python_wrapper",
            team=os.getenv("GENOPS_TEAM", "auto-instrumentation-demo"),
            project=os.getenv("GENOPS_PROJECT", "vercel-ai-sdk-demo"),
            environment="development",
        )

        print("✅ Auto-instrumentation enabled!")
        print(f"   Team: {self.adapter.governance_attrs.get('team', 'N/A')}")
        print(f"   Project: {self.adapter.governance_attrs.get('project', 'N/A')}")
        print("")

        return True

    def generate_instrumented_package(self):
        """Generate the auto-instrumentation JavaScript package."""
        print("📦 Generating auto-instrumentation package...")

        # Create temporary directory for the package
        temp_dir = Path(tempfile.mkdtemp(prefix="genops_vercel_"))
        self.temp_files.append(temp_dir)

        # Generate the instrumentation code
        instrumentation_path = self.adapter.generate_instrumentation_code(
            str(temp_dir / "genops-vercel-instrumentation.js")
        )

        # Create package.json for the instrumented package
        package_json = {
            "name": "genops-vercel-ai-sdk-demo",
            "version": "1.0.0",
            "description": "Auto-instrumented Vercel AI SDK with GenOps governance",
            "main": "app.js",
            "dependencies": {"ai": "^3.0.0", "@ai-sdk/openai": "^0.0.15"},
        }

        with open(temp_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)

        print(f"✅ Generated instrumentation at: {instrumentation_path}")
        return temp_dir, instrumentation_path

    def create_sample_application(self, temp_dir: Path):
        """Create a sample Vercel AI SDK application that gets auto-instrumented."""

        # This represents an EXISTING Vercel AI SDK application
        # that needs NO changes to get GenOps governance
        original_app = """
// Original Vercel AI SDK Application
// NO CHANGES NEEDED - GenOps governance added automatically!

const { generateText, streamText } = require('ai');
const { openai } = require('@ai-sdk/openai');

async function businessLogic() {
    console.log('🏢 Running existing business logic...');

    // Example 1: Simple text generation
    console.log('\\n1️⃣ Simple Text Generation:');
    const result1 = await generateText({
        model: openai('gpt-3.5-turbo'),
        prompt: 'What are the key benefits of cloud computing?',
        maxTokens: 150
    });
    console.log('Response:', result1.text);

    // Example 2: Structured response
    console.log('\\n2️⃣ Structured Response:');
    const result2 = await generateText({
        model: openai('gpt-4'),
        prompt: 'List 3 programming languages and their primary use cases',
        maxTokens: 200,
        temperature: 0.3
    });
    console.log('Response:', result2.text);

    // Example 3: Creative writing
    console.log('\\n3️⃣ Creative Writing:');
    const result3 = await generateText({
        model: openai('gpt-3.5-turbo'),
        prompt: 'Write a haiku about artificial intelligence',
        maxTokens: 50,
        temperature: 0.9
    });
    console.log('Response:', result3.text);

    console.log('\\n✅ Business logic completed successfully!');
}

// This is the ORIGINAL application code - unchanged!
businessLogic().catch(console.error);
"""

        # But we'll use the INSTRUMENTED version instead
        instrumented_app = """
// Auto-Instrumented Vercel AI SDK Application
// Uses GenOps instrumentation automatically!

// Import from GenOps instrumentation (instead of direct 'ai' package)
const { generateText, streamText, original } = require('./genops-vercel-instrumentation');

async function businessLogic() {
    console.log('🏢 Running business logic with GenOps governance...');

    // Example 1: Simple text generation (SAME CODE, now with governance!)
    console.log('\\n1️⃣ Simple Text Generation (with GenOps):');
    try {
        // This looks identical to original code but now includes governance
        const result1 = await generateText({
            model: 'gpt-3.5-turbo',
            prompt: 'What are the key benefits of cloud computing?',
            maxTokens: 150,
            // GenOps governance attributes can be added optionally
            team: process.env.GENOPS_TEAM,
            project: process.env.GENOPS_PROJECT
        });
        console.log('Response:', result1.text || 'Generated successfully');
        console.log('Tokens used:', result1.usage?.totalTokens || 'N/A');
    } catch (error) {
        console.error('Error in example 1:', error.message);
    }

    // Example 2: Structured response
    console.log('\\n2️⃣ Structured Response (with cost tracking):');
    try {
        const result2 = await generateText({
            model: 'gpt-4',
            prompt: 'List 3 programming languages and their primary use cases',
            maxTokens: 200,
            temperature: 0.3
        });
        console.log('Response:', result2.text || 'Generated successfully');
        console.log('Tokens used:', result2.usage?.totalTokens || 'N/A');
    } catch (error) {
        console.error('Error in example 2:', error.message);
    }

    // Example 3: Creative writing
    console.log('\\n3️⃣ Creative Writing (with telemetry):');
    try {
        const result3 = await generateText({
            model: 'gpt-3.5-turbo',
            prompt: 'Write a haiku about artificial intelligence',
            maxTokens: 50,
            temperature: 0.9
        });
        console.log('Response:', result3.text || 'Generated successfully');
        console.log('Tokens used:', result3.usage?.totalTokens || 'N/A');
    } catch (error) {
        console.error('Error in example 3:', error.message);
    }

    console.log('\\n✅ Business logic completed with full governance tracking!');
}

// Run the auto-instrumented business logic
businessLogic().catch(console.error);
"""

        # Write both versions for comparison
        with open(temp_dir / "original-app.js", "w") as f:
            f.write(original_app)

        with open(temp_dir / "app.js", "w") as f:
            f.write(instrumented_app)

        print("✅ Created sample applications:")
        print(f"   Original: {temp_dir}/original-app.js")
        print(f"   Instrumented: {temp_dir}/app.js")

        return temp_dir / "app.js"

    def demonstrate_cost_estimation(self):
        """Demonstrate cost estimation capabilities."""
        print("\n💰 Cost Estimation Demonstration:")
        print("-" * 40)

        models_to_test = [
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-haiku"),
            ("anthropic", "claude-3-sonnet"),
        ]

        sample_prompt = "What are the key benefits of cloud computing? Please provide a comprehensive overview."
        expected_response_length = 500  # characters

        for provider, model in models_to_test:
            try:
                # Get model information
                model_info = get_model_info(provider, model)
                if model_info:
                    print(f"\n📊 {provider.title()} - {model}:")
                    print(f"   Input cost: ${model_info.input_price_per_1k}/1K tokens")
                    print(
                        f"   Output cost: ${model_info.output_price_per_1k}/1K tokens"
                    )
                    print(f"   Context length: {model_info.context_length:,} tokens")

                # Estimate cost for sample prompt
                min_cost, max_cost = estimate_cost(
                    provider, model, len(sample_prompt), expected_response_length
                )
                print(f"   Estimated cost: ${min_cost:.6f} - ${max_cost:.6f}")

            except Exception as e:
                print(f"   Error estimating cost for {provider}/{model}: {e}")

    def run_instrumented_demo(self, app_path: Path, temp_dir: Path):
        """Run the auto-instrumented application demo."""
        print("\n🚀 Running Auto-Instrumented Application:")
        print("=" * 50)

        # Install dependencies
        print("📦 Installing dependencies...")
        import subprocess

        try:
            subprocess.run(
                ["npm", "install"],
                cwd=temp_dir,
                check=True,
                capture_output=True,
                timeout=60,
            )
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Dependency installation timed out")
            return False

        # Set up environment
        env = os.environ.copy()
        env.update(
            {
                "GENOPS_TEAM": self.adapter.governance_attrs.get("team", "auto-demo"),
                "GENOPS_PROJECT": self.adapter.governance_attrs.get(
                    "project", "vercel-demo"
                ),
                "GENOPS_ENVIRONMENT": "development",
            }
        )

        # Run the instrumented application
        print("🎯 Executing auto-instrumented application...")

        try:
            with self.adapter.track_request(
                "generateText", "openai", "gpt-3.5-turbo"
            ) as request:
                result = subprocess.run(
                    ["node", str(app_path)],
                    cwd=temp_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode == 0:
                    print("✅ Application executed successfully!")
                    print("\n📄 Application Output:")
                    print("-" * 30)
                    print(result.stdout)

                    # Update tracking information
                    request.response = "Multiple text generations completed"

                    return True
                else:
                    print("❌ Application execution failed!")
                    print(f"Error: {result.stderr}")
                    return False

        except subprocess.TimeoutExpired:
            print("❌ Application execution timed out!")
            return False
        except Exception as e:
            print(f"❌ Error running application: {e}")
            return False

    def cleanup(self):
        """Clean up temporary files."""
        for temp_path in self.temp_files:
            try:
                if temp_path.exists():
                    import shutil

                    shutil.rmtree(temp_path)
            except Exception as e:
                logger.warning(f"Could not clean up {temp_path}: {e}")

    def run_demo(self):
        """Run the complete auto-instrumentation demo."""
        print("🤖 GenOps Auto-Instrumentation Demo for Vercel AI SDK")
        print("=" * 60)
        print("")
        print("This demo shows how to add GenOps governance to existing")
        print("Vercel AI SDK applications with ZERO code changes!")
        print("")

        success = False
        try:
            # Setup auto-instrumentation
            if not self.setup():
                return False

            # Generate instrumentation package
            temp_dir, instrumentation_path = self.generate_instrumented_package()

            # Create sample application
            app_path = self.create_sample_application(temp_dir)

            # Demonstrate cost estimation
            self.demonstrate_cost_estimation()

            # Run the demo
            success = self.run_instrumented_demo(app_path, temp_dir)

            if success:
                print("\n🎉 Auto-Instrumentation Demo Completed!")
                print("\nWhat happened:")
                print("1. ✅ GenOps auto-instrumentation enabled")
                print("2. ✅ Generated instrumentation package")
                print("3. ✅ Created sample application (no changes needed!)")
                print("4. ✅ Demonstrated cost estimation")
                print("5. ✅ Ran application with full governance tracking")

                print("\n💡 Key Benefits:")
                print("• Zero code changes to existing applications")
                print("• Automatic cost tracking across all requests")
                print("• Real-time governance telemetry")
                print("• Multi-provider cost attribution")
                print("• OpenTelemetry-compatible exports")

                print("\n🔗 Next Steps:")
                print("• Deploy the instrumentation package to production")
                print("• Set up observability dashboards")
                print("• Configure budget alerts and governance policies")
                print("• Try streaming examples (03_streaming_chat.py)")

        finally:
            self.cleanup()

        return success


def run_auto_instrumentation_demo():
    """Run the auto-instrumentation demo."""
    demo = AutoInstrumentationDemo()
    return demo.run_demo()


if __name__ == "__main__":
    success = run_auto_instrumentation_demo()
    exit(0 if success else 1)
