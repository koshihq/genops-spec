#!/usr/bin/env python3
"""
Example: Basic Text Generation with Vercel AI SDK Governance

Complexity: ⭐ Beginner

This example demonstrates the simplest way to add GenOps governance
to Vercel AI SDK text generation operations. Perfect for getting started.

Prerequisites:
- Node.js 16+ installed
- Vercel AI SDK installed: npm install ai @ai-sdk/openai
- OpenAI API key set in environment
- GenOps package installed: pip install genops

Usage:
    python 01_basic_text_generation.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key
    GENOPS_TEAM: Team name for cost attribution (default: vercel-examples)
    GENOPS_PROJECT: Project name for tracking (default: basic-text-generation)
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import GenOps Vercel AI SDK integration
try:
    from genops.providers.vercel_ai_sdk import (
        GenOpsVercelAISDKAdapter,  # noqa: F401
        auto_instrument,
        track_generate_text,  # noqa: F401
    )
    from genops.providers.vercel_ai_sdk_validation import validate_setup
except ImportError as e:
    logger.error(f"GenOps not installed: {e}")
    logger.error("Install with: pip install genops")
    exit(1)


def validate_environment():
    """Validate the environment before running the example."""
    print("🔍 Validating environment...")

    # Quick validation check
    result = validate_setup(
        check_nodejs=True,
        check_python_deps=True,
        check_genops_config=True,
        verbose=False,
    )

    if not result.all_passed:
        print("❌ Environment validation failed!")
        print("\nIssues found:")
        for check_result in result.results:
            if not check_result.passed:
                print(f"  • {check_result.check_name}: {check_result.message}")
                if check_result.fix_suggestion:
                    print(f"    Fix: {check_result.fix_suggestion}")
        print("\nPlease fix the issues above and try again.")
        return False

    print("✅ Environment validation passed!")
    return True


def create_instrumented_js_script(prompt: str, model: str = "gpt-4") -> str:
    """Create a JavaScript script with GenOps instrumentation."""

    # Generate unique script name
    script_name = f"genops_text_generation_{int(time.time())}.js"
    script_path = Path(tempfile.gettempdir()) / script_name

    # JavaScript code with GenOps telemetry hooks
    js_code = f'''
// GenOps Instrumented Vercel AI SDK Example
const {{ generateText }} = require('ai');
const {{ openai }} = require('@ai-sdk/openai');
const http = require('http');

// GenOps Configuration
const GENOPS_CONFIG = {{
    team: process.env.GENOPS_TEAM || 'vercel-examples',
    project: process.env.GENOPS_PROJECT || 'basic-text-generation',
    environment: process.env.GENOPS_ENVIRONMENT || 'development',
    telemetry_endpoint: process.env.GENOPS_TELEMETRY_ENDPOINT || 'http://localhost:8080/telemetry'
}};

// Telemetry helper function
function sendTelemetry(data) {{
    const telemetryData = {{
        timestamp: Date.now(),
        provider: 'vercel-ai-sdk',
        underlying_provider: 'openai',
        operation: 'generateText',
        governance: GENOPS_CONFIG,
        ...data
    }};

    console.log('[GenOps Telemetry]', JSON.stringify(telemetryData, null, 2));
}}

async function main() {{
    const startTime = Date.now();
    const requestId = `req_${{startTime}}_${{Math.random().toString(36).substr(2, 9)}}`;

    try {{
        // Send start telemetry
        sendTelemetry({{
            type: 'request_start',
            requestId: requestId,
            model: '{model}',
            prompt: '{prompt}',
        }});

        console.log('🚀 Generating text with Vercel AI SDK...');
        console.log(`Model: {model}`);
        console.log(`Prompt: "{prompt}"`);
        console.log('');

        // Generate text using Vercel AI SDK
        const result = await generateText({{
            model: openai('{model}'),
            prompt: '{prompt}',
            maxTokens: 200,
            temperature: 0.7,
        }});

        const endTime = Date.now();
        const duration = endTime - startTime;

        // Display results
        console.log('📝 Generated Text:');
        console.log('=' * 50);
        console.log(result.text);
        console.log('=' * 50);
        console.log('');

        // Usage statistics
        console.log('📊 Usage Statistics:');
        console.log(`Input tokens: ${{result.usage?.promptTokens || 'N/A'}}`);
        console.log(`Output tokens: ${{result.usage?.completionTokens || 'N/A'}}`);
        console.log(`Total tokens: ${{result.usage?.totalTokens || 'N/A'}}`);
        console.log(`Duration: ${{duration}}ms`);
        console.log(`Finish reason: ${{result.finishReason || 'N/A'}}`);
        console.log('');

        // Send completion telemetry
        sendTelemetry({{
            type: 'request_complete',
            requestId: requestId,
            success: true,
            duration: duration,
            usage: result.usage,
            finishReason: result.finishReason,
            outputLength: result.text?.length || 0
        }});

        // Return structured result
        const response = {{
            success: true,
            text: result.text,
            usage: result.usage,
            duration: duration,
            finishReason: result.finishReason,
            governance: GENOPS_CONFIG
        }};

        console.log('✅ Text generation completed successfully!');
        console.log(JSON.stringify(response, null, 2));

    }} catch (error) {{
        const endTime = Date.now();
        const duration = endTime - startTime;

        console.error('❌ Error generating text:', error.message);

        // Send error telemetry
        sendTelemetry({{
            type: 'request_error',
            requestId: requestId,
            success: false,
            duration: duration,
            error: error.message
        }});

        process.exit(1);
    }}
}}

// Run the example
main().catch(console.error);
'''

    # Write the JavaScript file
    with open(script_path, "w") as f:
        f.write(js_code)

    return str(script_path)


def run_basic_text_generation_example():
    """Run the basic text generation example with GenOps governance."""

    print("🧠 GenOps Vercel AI SDK - Basic Text Generation Example")
    print("=" * 60)
    print("")

    # Validate environment first
    if not validate_environment():
        return False

    # Get configuration from environment
    team = os.getenv("GENOPS_TEAM", "vercel-examples")
    project = os.getenv("GENOPS_PROJECT", "basic-text-generation")
    model = os.getenv("OPENAI_MODEL", "gpt-4")

    print(f"Team: {team}")
    print(f"Project: {project}")
    print(f"Model: {model}")
    print("")

    # Initialize GenOps adapter
    print("📊 Initializing GenOps governance...")
    adapter = auto_instrument(
        integration_mode="subprocess",
        team=team,
        project=project,
        environment="development",
    )
    print("✅ GenOps adapter initialized")
    print("")

    # Example prompts to try
    prompts = [
        "Explain quantum computing in simple terms",
        "Write a short story about a robot learning to paint",
        "What are the benefits of renewable energy?",
    ]

    # Let user choose a prompt or use default
    print("📝 Choose a prompt:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")
    print(f"  {len(prompts) + 1}. Custom prompt")

    try:
        choice = input(
            f"\nEnter choice (1-{len(prompts) + 1}, or press Enter for #1): "
        ).strip()
        if not choice:
            choice = "1"

        choice_num = int(choice)
        if 1 <= choice_num <= len(prompts):
            selected_prompt = prompts[choice_num - 1]
        elif choice_num == len(prompts) + 1:
            selected_prompt = input("Enter your custom prompt: ").strip()
            if not selected_prompt:
                selected_prompt = prompts[0]  # Fallback
        else:
            print("Invalid choice, using default prompt")
            selected_prompt = prompts[0]
    except (ValueError, KeyboardInterrupt):
        print("Using default prompt")
        selected_prompt = prompts[0]

    print(f'\n🎯 Selected prompt: "{selected_prompt}"')
    print("")

    # Track the request using GenOps
    print("🔄 Starting GenOps tracked request...")
    with adapter.track_request(
        "generateText",
        "openai",
        model,
        prompt=selected_prompt,
        team=team,
        project=project,
    ) as request:
        # Create and run instrumented JavaScript
        js_script_path = create_instrumented_js_script(selected_prompt, model)

        try:
            print("🚀 Executing Vercel AI SDK with governance...")

            # Set environment variables for the subprocess
            env = os.environ.copy()
            env.update(
                {
                    "GENOPS_TEAM": team,
                    "GENOPS_PROJECT": project,
                    "GENOPS_ENVIRONMENT": "development",
                }
            )

            # Run the JavaScript with Node.js
            result = subprocess.run(
                ["node", js_script_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )

            if result.returncode == 0:
                print("✅ JavaScript execution completed successfully!")

                # Parse any JSON output from the script
                try:
                    lines = result.stdout.strip().split("\n")
                    for line in lines:
                        if line.startswith("{") and '"success"' in line:
                            response_data = json.loads(line)

                            # Update request tracking with results
                            if "usage" in response_data and response_data["usage"]:
                                usage = response_data["usage"]
                                request.input_tokens = usage.get("promptTokens", 0)
                                request.output_tokens = usage.get("completionTokens", 0)

                            request.response = response_data.get("text", "")
                            request.duration_ms = response_data.get("duration", 0)

                            break
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON response from JavaScript")

                print("\n📊 Final GenOps Telemetry:")
                print(f"  Request ID: {request.request_id}")
                print(f"  Provider: {request.provider}")
                print(f"  Model: {request.model}")
                print(f"  Input tokens: {request.input_tokens}")
                print(f"  Output tokens: {request.output_tokens}")
                print(f"  Duration: {request.duration_ms}ms")
                if request.cost:
                    print(f"  Estimated cost: ${request.cost}")

            else:
                print("❌ JavaScript execution failed!")
                print(f"Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ JavaScript execution timed out!")
            return False
        except Exception as e:
            print(f"❌ Error executing JavaScript: {e}")
            return False
        finally:
            # Clean up temporary script
            try:
                os.unlink(js_script_path)
            except Exception:
                pass

    print("\n🎉 Example completed successfully!")
    print("\nWhat happened:")
    print("1. ✅ Environment validated (Node.js, packages, API keys)")
    print("2. ✅ GenOps governance initialized")
    print("3. ✅ Vercel AI SDK executed with telemetry")
    print("4. ✅ Cost and usage tracked automatically")
    print("5. ✅ Results displayed with governance context")

    print("\nNext steps:")
    print("• Try example 02_auto_instrumentation.py for zero-code setup")
    print("• Explore streaming with 03_streaming_chat.py")
    print("• Set up observability dashboard to view telemetry")

    return True


if __name__ == "__main__":
    success = run_basic_text_generation_example()
    exit(0 if success else 1)
