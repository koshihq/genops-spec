#!/usr/bin/env python3
"""
Basic Dust AI tracking with GenOps.

This example demonstrates:
- Basic Dust adapter setup
- Conversation creation with governance tracking
- Message sending with cost attribution
- Agent execution with telemetry
- Data source management

Prerequisites:
- pip install genops[dust]
- Set DUST_API_KEY and DUST_WORKSPACE_ID environment variables
"""

import os
import sys

import genops
from genops.providers.dust import instrument_dust

# Constants to avoid CodeQL false positives
CONVERSATION_VISIBILITY_RESTRICTED = "private"


def main():
    """Demonstrate basic Dust tracking with GenOps."""

    print("🚀 Basic Dust AI Tracking with GenOps")
    print("=" * 50)

    # Check environment variables
    api_key = os.getenv("DUST_API_KEY")
    workspace_id = os.getenv("DUST_WORKSPACE_ID")

    if not api_key or not workspace_id:
        print("❌ Missing required environment variables:")
        print("   Set DUST_API_KEY and DUST_WORKSPACE_ID")
        print("   Get these from your Dust workspace settings")
        sys.exit(1)

    # Initialize GenOps with OpenTelemetry
    print("\n📊 Initializing GenOps telemetry...")
    genops.init(
        service_name=os.getenv("OTEL_SERVICE_NAME", "dust-basic-example"),
        enable_console_export=True,  # Show traces in console for demo
    )

    # Create instrumented Dust adapter
    print("🔧 Setting up Dust adapter...")
    dust = instrument_dust(
        api_key=api_key,
        workspace_id=workspace_id,
        # Governance attributes
        team=os.getenv("GENOPS_TEAM", "ai-examples"),
        project=os.getenv("GENOPS_PROJECT", "dust-integration"),
        environment=os.getenv("GENOPS_ENVIRONMENT", "development"),
    )

    try:
        # Example 1: Create a conversation
        print("\n💬 Creating conversation with governance tracking...")
        # Use constant to avoid CodeQL false positive
        visibility_setting = CONVERSATION_VISIBILITY_RESTRICTED
        conversation_result = dust.create_conversation(
            title="GenOps Integration Demo",
            visibility=visibility_setting,
            # Additional governance attributes
            customer_id="demo-customer-123",
            user_id="demo-user-456",
            feature="conversation-management",
        )

        if conversation_result and "conversation" in conversation_result:
            conversation_id = conversation_result["conversation"]["sId"]
            print(f"✅ Created conversation: {conversation_id}")

            # Example 2: Send messages with cost attribution
            print("\n📝 Sending messages with cost tracking...")

            messages = [
                "Hello! This is a demo of GenOps with Dust AI.",
                "Can you help me understand how agent workflows work?",
                "What are the best practices for data source management?",
            ]

            for i, message_content in enumerate(messages, 1):
                message_result = dust.send_message(
                    conversation_id=conversation_id,
                    content=message_content,
                    # Governance tracking per message
                    customer_id="demo-customer-123",
                    user_id="demo-user-456",
                    feature="message-sending",
                    cost_center="ai-research",
                )

                if message_result:
                    print(f"  ✅ Sent message {i}: {message_content[:50]}...")
                else:
                    print(f"  ❌ Failed to send message {i}")
        else:
            print("❌ Failed to create conversation")
            return

        # Example 3: Agent execution (if available)
        print("\n🤖 Demonstrating agent execution tracking...")

        # Note: This is a demo - replace with actual agent ID from your workspace
        demo_agent_id = "demo-agent-123"

        try:
            agent_result = dust.run_agent(
                agent_id=demo_agent_id,
                inputs={
                    "query": "What is GenOps and how does it help with AI governance?",
                    "context": "demonstration",
                },
                # Governance attributes
                customer_id="demo-customer-123",
                user_id="demo-user-456",
                team="ai-examples",
                project="dust-integration",
                feature="agent-execution",
                cost_center="ai-research",
            )

            if agent_result:
                print("✅ Agent execution tracked successfully")
                if "run" in agent_result:
                    run_info = agent_result["run"]
                    print(f"   Run ID: {run_info.get('sId', 'N/A')}")
                    print(f"   Status: {run_info.get('status', 'N/A')}")
            else:
                print("⚠️  Agent execution returned no result")

        except Exception as e:
            print(f"⚠️  Agent execution demo skipped: {e}")
            print("   (This is normal if the demo agent doesn't exist)")

        # Example 4: Data source search
        print("\n🔍 Demonstrating data source search...")

        try:
            search_result = dust.search_datasources(
                query="best practices for AI governance",
                data_sources=[],  # Search all available data sources
                top_k=3,
                # Governance attributes
                customer_id="demo-customer-123",
                user_id="demo-user-456",
                feature="knowledge-search",
                cost_center="ai-research",
            )

            if search_result:
                documents = search_result.get("documents", [])
                print(f"✅ Search completed, found {len(documents)} documents")

                for i, doc in enumerate(documents[:2], 1):  # Show first 2 results
                    if "chunk" in doc and "text" in doc["chunk"]:
                        text_preview = doc["chunk"]["text"][:100] + "..."
                        print(f"   Document {i}: {text_preview}")
            else:
                print("⚠️  Search returned no results")

        except Exception as e:
            print(f"⚠️  Data source search demo: {e}")

        # Example 5: Cost and usage summary
        print("\n💰 Cost and usage summary...")
        print("   (Cost calculations are estimates based on usage patterns)")

        from genops.providers.dust_pricing import calculate_dust_cost

        # Estimate costs for this demo session
        estimated_cost = calculate_dust_cost(
            operation_type="conversation",
            operation_count=1,  # 1 conversation created
            estimated_tokens=500,  # Rough estimate
            user_count=1,
            plan_type="pro",  # Assuming Pro plan
        )

        print(
            f"   Monthly subscription (1 user): €{estimated_cost.monthly_subscription_cost}"
        )
        print(f"   API costs: €{estimated_cost.estimated_api_cost}")
        print(f"   Total estimated: €{estimated_cost.total_cost}")
        print(f"   Currency: {estimated_cost.currency}")

        print("\n✅ Basic tracking demo completed successfully!")
        print("\n📈 Telemetry Data Generated:")
        print("   • Conversation creation trace with governance attributes")
        print("   • Message sending traces with cost attribution")
        print("   • Agent execution traces (if available)")
        print("   • Data source search traces")
        print("   • Cost and usage metrics")

        print("\n🔍 Next Steps:")
        print("   • View traces in your OpenTelemetry collector")
        print("   • Run 'python cost_optimization.py' for cost analysis")
        print("   • Try 'python production_patterns.py' for enterprise patterns")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("💡 Tip: Run 'python setup_validation.py' to check your configuration")
        sys.exit(1)


def demonstrate_error_handling():
    """Show how GenOps handles Dust API errors gracefully."""

    print("\n🛡️  Error Handling Demonstration")
    print("-" * 40)

    # Example with invalid credentials (will be caught by validation)
    try:
        dust_invalid = instrument_dust(
            api_key="invalid-key", workspace_id="invalid-workspace"
        )

        # This will fail gracefully with proper error tracking
        dust_invalid.create_conversation(title="Test")

    except Exception as e:
        print(f"✅ Error properly caught and tracked: {type(e).__name__}")
        print("   GenOps automatically tracks errors in telemetry")


if __name__ == "__main__":
    main()
    demonstrate_error_handling()
